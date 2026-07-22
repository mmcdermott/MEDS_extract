"""Integration tests for :mod:`MEDS_extract.download` using httpx.MockTransport.

Doctests throughout the module cover most pure-Python machinery — spec dispatch, URL
normalization, hash helpers, ``RemoteFile`` validation, SHA256SUMS parsing, manifest
filtering, and the :meth:`Source.download_all`
skip / overwrite / path-traversal / duplicate-dest paths via the doctest in
``source.py``. This file covers what doctests can't: the ``_resumable_stream``
HTTP primitive's wire-level behavior (Range resume, 416/206 mismatch handling,
identity content-coding), streaming retry, the ``Source._fetch_one`` staging
pipeline, end-to-end ``download_all`` against ``MockTransport``-backed
:class:`HTTPSource` / :class:`PhysioNetSource` (sequential and pooled), the CLI
subprocess flows (success and failure exits), and the SIGINT-cancellation
regression that needs a real signal.

MockTransport intercepts at the httpx level below the client, so retry/timeout/Range
behavior is all exercised against the real client code path.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import httpx
import pytest
from tenacity import wait_fixed

if TYPE_CHECKING:
    from pathlib import Path

from MEDS_extract.download import ChecksumError, HTTPSource, PhysioNetSource

# ``_resumable_stream`` lives on HTTPSource as a staticmethod — alias it for brevity.
_resumable_stream = HTTPSource._resumable_stream


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _mock_client(handler):
    """Build an :class:`httpx.Client` whose transport is a user-supplied handler.

    ``handler(request) -> httpx.Response`` is the same signature ``MockTransport`` expects.
    """
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=10.0)


# ``_resumable_stream`` writes bytes from a URL into a target path. These tests cover
# its wire-level behavior (Range resume, 416/206 mismatch handling). SHA verify and
# atomic rename live on ``Source._fetch_one`` and are covered by
# ``test_fetch_one_*`` further down.


def test_resumable_stream_writes_target(tmp_path: Path):
    body = b"hello world"
    url = "https://example.com/x.csv"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/x.csv"
        return httpx.Response(200, content=body)

    client = _mock_client(handler)
    target = tmp_path / "x.csv.part"
    _resumable_stream(client, url, target)
    assert target.read_bytes() == body


def test_resumable_stream_range_resume_appends(tmp_path: Path):
    """If ``target`` exists, send Range: bytes=N- and append to the existing bytes."""
    full_body = b"abcdefghij"
    url = "https://example.com/x.csv"
    target = tmp_path / "x.csv.part"
    target.write_bytes(full_body[:4])  # "abcd"

    seen_range: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_range.append(request.headers.get("Range"))
        if request.headers.get("Range") == "bytes=4-":
            return httpx.Response(206, content=full_body[4:], headers={"Content-Range": "bytes 4-9/10"})
        return httpx.Response(200, content=full_body)

    client = _mock_client(handler)
    _resumable_stream(client, url, target)
    assert target.read_bytes() == full_body
    assert seen_range == ["bytes=4-"]


def test_resumable_stream_range_ignored_restarts_from_scratch(tmp_path: Path):
    """Server returning 200 despite Range header → client restarts from byte 0."""
    body = b"fresh contents"
    url = "https://example.com/x.csv"
    target = tmp_path / "x.csv.part"
    target.write_bytes(b"stale partial")

    def handler(request):
        # Ignore Range, always serve full body with 200.
        return httpx.Response(200, content=body)

    client = _mock_client(handler)
    _resumable_stream(client, url, target)
    assert target.read_bytes() == body


def test_resumable_stream_mismatched_content_range_restarts(tmp_path: Path):
    """Server returning 206 with a Content-Range that doesn't start at resume_from must not silently corrupt
    ``target`` — the client restarts from byte 0."""
    full_body = b"abcdefghij"
    url = "https://example.com/x.csv"
    target = tmp_path / "x.csv.part"
    target.write_bytes(full_body[:4])  # "abcd"

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        rng = request.headers.get("Range")
        seen.append(rng)
        if rng == "bytes=4-":
            # Lying server: returns 206 but starts at byte 2 instead of 4.
            return httpx.Response(
                206,
                content=full_body[2:],
                headers={"Content-Range": "bytes 2-9/10"},
            )
        # Second attempt: no Range → full body.
        return httpx.Response(200, content=full_body)

    client = _mock_client(handler)
    _resumable_stream(client, url, target)
    assert target.read_bytes() == full_body
    assert seen == ["bytes=4-", None]


def test_resumable_stream_416_restarts(tmp_path: Path):
    """Server 416 (Range Not Satisfiable) — remote file shrank — must restart from byte 0."""
    full_body = b"short"
    url = "https://example.com/x.csv"
    target = tmp_path / "x.csv.part"
    target.write_bytes(b"stale longer content")  # larger than current full body

    seen_statuses = []

    def handler(request: httpx.Request) -> httpx.Response:
        rng = request.headers.get("Range")
        if rng:
            seen_statuses.append(416)
            return httpx.Response(416)
        seen_statuses.append(200)
        return httpx.Response(200, content=full_body)

    client = _mock_client(handler)
    _resumable_stream(client, url, target)
    assert target.read_bytes() == full_body
    assert seen_statuses == [416, 200]


# ── Source._fetch_one pipeline (.part staging + sha verify + atomic rename) ─────────


def test_fetch_one_discards_stale_part_when_no_sha(tmp_path: Path):
    """``_fetch_one``: a stale ``.part`` from a prior failed run can't be safely resumed when the manifest has
    no SHA to catch silent corruption.

    The
    orchestrator unlinks it before calling ``_pull``, which then starts fresh
    (no ``Range`` header sent).
    """
    body = b"the real full content"
    url = "https://example.com/x.csv"
    dest = tmp_path / "x.csv"
    (tmp_path / "x.csv.part").write_bytes(b"stale partial")

    seen_range: list[str | None] = []

    def handler(request):
        seen_range.append(request.headers.get("Range"))
        return httpx.Response(200, content=body)

    client = _mock_client(handler)
    src = HTTPSource(urls=[url], client=client)  # plain string → no sha
    [item] = src.files
    src._fetch_one(item, tmp_path, do_overwrite=False)
    assert dest.read_bytes() == body
    assert seen_range == [None]  # no Range header → started fresh


def test_fetch_one_writes_dest_when_sha_matches(tmp_path: Path):
    """End-to-end ``_fetch_one``: ``_pull`` produces bytes via ``.part``, base verifies sha, atomic-renames
    into ``dest``.

    No ``.part`` remains.
    """
    body = b"hello world"
    url = "https://example.com/x.csv"

    def handler(request):
        return httpx.Response(200, content=body)

    client = _mock_client(handler)
    src = HTTPSource(urls=[{"url": url, "sha256": _sha(body)}], client=client)
    [item] = src.files
    src._fetch_one(item, tmp_path, do_overwrite=False)
    dest = tmp_path / "x.csv"
    assert dest.read_bytes() == body
    assert not dest.with_name(dest.name + ".part").exists()


def test_fetch_one_raises_checksum_error_on_sha_mismatch(tmp_path: Path):
    """``_fetch_one``: if ``_pull`` writes content that doesn't match ``remote.sha256``, the orchestrator
    raises ``ChecksumError`` and cleans up the staged ``.part`` — ``dest`` is never created."""
    body = b"hello world"
    wrong_digest = "0" * 64
    url = "https://example.com/x.csv"

    def handler(request):
        return httpx.Response(200, content=body)

    client = _mock_client(handler)
    src = HTTPSource(urls=[{"url": url, "sha256": wrong_digest}], client=client)
    [item] = src.files
    dest = tmp_path / "x.csv"
    with pytest.raises(ChecksumError):
        src._fetch_one(item, tmp_path, do_overwrite=False)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


# ── HTTPSource end-to-end through Source.download_all ────────────────────────────────


def test_http_source_fetches_multiple_urls(tmp_path: Path):
    bodies = {
        "https://example.com/a.csv": b"a,b,c\n1,2,3",
        "https://example.com/b.csv": b"x,y\n4,5",
    }

    def handler(request):
        body = bodies.get(str(request.url))
        return httpx.Response(200 if body else 404, content=body or b"")

    client = _mock_client(handler)
    src = HTTPSource(urls=list(bodies.keys()), client=client)
    src.download_all(tmp_path)

    assert (tmp_path / "a.csv").read_bytes() == bodies["https://example.com/a.csv"]
    assert (tmp_path / "b.csv").read_bytes() == bodies["https://example.com/b.csv"]


def test_http_source_honors_rel_path_override(tmp_path: Path):
    body = b"shared metadata"

    def handler(request):
        return httpx.Response(200, content=body)

    client = _mock_client(handler)
    src = HTTPSource(
        urls=[{"url": "https://example.com/lookups.csv", "rel_path": "concept_map/lookups.csv"}],
        client=client,
    )
    src.download_all(tmp_path)
    assert (tmp_path / "concept_map" / "lookups.csv").read_bytes() == body


def test_http_source_checksum_mismatch_fails(tmp_path: Path):
    body = b"contents"
    wrong_sha = "0" * 64

    def handler(request):
        return httpx.Response(200, content=body)

    client = _mock_client(handler)
    src = HTTPSource(
        urls=[{"url": "https://example.com/x.csv", "sha256": wrong_sha}],
        client=client,
    )
    with pytest.raises(ChecksumError):
        src.download_all(tmp_path)


# ── PhysioNetSource: SHA256SUMS.txt parsing + fetch flow ─────────────────────────────


def test_physionet_source_end_to_end(tmp_path: Path):
    files = {
        "patients.csv": b"subject_id\n1\n",
        "labs/vitals.csv": b"sid,hr\n1,80\n",
    }
    manifest = "\n".join(f"{_sha(body)}  {rel}" for rel, body in files.items()) + "\n"

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path  # e.g. "/files/demo/SHA256SUMS.txt"
        if path.endswith("/SHA256SUMS.txt"):
            return httpx.Response(200, text=manifest)
        for rel, body in files.items():
            if path.endswith("/" + rel):
                return httpx.Response(200, content=body)
        return httpx.Response(404)

    client = _mock_client(handler)
    src = PhysioNetSource(base_url="https://physionet.org/files/demo/1.0", client=client)
    src.download_all(tmp_path)

    assert (tmp_path / "patients.csv").read_bytes() == files["patients.csv"]
    assert (tmp_path / "labs" / "vitals.csv").read_bytes() == files["labs/vitals.csv"]


def test_physionet_source_trailing_slash_normalization():
    """Base URL is normalized to include a trailing slash so URL concatenation is clean."""
    client = _mock_client(lambda r: httpx.Response(404))
    a = PhysioNetSource(base_url="https://example.com/files/x", client=client)
    b = PhysioNetSource(base_url="https://example.com/files/x/", client=client)
    assert a._base_url == b._base_url == "https://example.com/files/x/"


@pytest.mark.integration
def test_physionet_source_lists_mimic_demo_manifest():
    """Network integration test against the public MIMIC-IV demo.

    Gated behind the ``integration`` marker — doesn't run in the default ``pytest``
    invocation, only when explicitly requested via ``pytest -m integration``. Validates
    that the ``SHA256SUMS.txt``-driven manifest machinery works end-to-end against the
    live PhysioNet host, not just against mock responses.
    """
    with PhysioNetSource(base_url="https://physionet.org/files/mimic-iv-demo/2.2") as src:
        files = src.files
    # Demo currently has ~34 files; threshold is a robust "not empty / not truncated"
    # floor — if the demo layout changes dramatically, this surfaces it.
    assert len(files) >= 20, f"demo manifest surprisingly small ({len(files)} files)"
    # SHA256SUMS.txt is the whole point — every entry must carry a digest.
    assert all(f.sha256 and len(f.sha256) == 64 for f in files)
    # Sanity: a known stable file in the demo tree (its removal would itself be a signal
    # worth investigating, which is what this assertion gives us).
    assert any("patients" in f.rel_path for f in files)


def test_physionet_source_rejects_half_credentials():
    """Username without password (or vice versa) is a clear user error — fail fast."""
    with pytest.raises(ValueError, match="must be supplied together"):
        PhysioNetSource(base_url="https://example.com/files/x", username="u", password=None)
    with pytest.raises(ValueError, match="must be supplied together"):
        PhysioNetSource(base_url="https://example.com/files/x", username=None, password="p")


def test_download_all_refuses_to_overwrite_unverifiable_file(tmp_path: Path):
    """An existing dest with no manifest sha raises ``FileExistsError`` rather than overwrite.

    Without a sha to verify against, the orchestrator can't tell if the file on disk matches the manifest, so
    the safe move is to refuse. The previous "silently overwrite" behavior masked stale or partially-flushed
    local copies.
    """
    full_body = b"correct content"
    url = "https://example.com/x.csv"
    dest = tmp_path / "x.csv"
    dest.write_bytes(b"stale")  # any prior content; no manifest sha to verify against

    def handler(request):
        return httpx.Response(200, content=full_body)

    client = _mock_client(handler)
    src = HTTPSource(urls=[url], client=client)  # plain string entry → no sha
    with pytest.raises(FileExistsError, match="does not verify"):
        src.download_all(tmp_path)
    # Stale local copy was not touched.
    assert dest.read_bytes() == b"stale"

    # ``do_overwrite=True`` clears the stale dest and refetches.
    src.download_all(tmp_path, do_overwrite=True)
    assert dest.read_bytes() == full_body


def test_download_all_continue_on_error_collects_failures_into_group(tmp_path: Path):
    """``continue_on_error=True`` collects per-file errors into an ``ExceptionGroup``."""
    body = b"ok"

    def handler(request):
        if str(request.url).endswith("bad.csv"):
            return httpx.Response(500, text="server error")
        return httpx.Response(200, content=body)

    client = _mock_client(handler)
    src = HTTPSource(
        urls=["https://example.com/good.csv", "https://example.com/bad.csv"],
        client=client,
        max_attempts=1,  # a 500 is retried by _pull; one attempt keeps the test instant
    )
    with pytest.raises(ExceptionGroup) as exc_info:
        src.download_all(tmp_path, continue_on_error=True)
    # Exactly one of the two files failed — the good one still landed.
    assert len(exc_info.value.exceptions) == 1
    assert (tmp_path / "good.csv").read_bytes() == body


def test_download_all_first_failure_reraises(tmp_path: Path):
    """Default mode re-raises the first per-file failure."""

    def handler(request):
        return httpx.Response(500, text="server error")

    client = _mock_client(handler)
    src = HTTPSource(urls=["https://example.com/x.csv"], client=client, max_attempts=1)
    with pytest.raises(httpx.HTTPStatusError):
        src.download_all(tmp_path)


def test_download_all_fail_fast_cancels_queued_futures(tmp_path: Path):
    """In pooled mode, the first failure cancels still-queued work — "fail fast" must actually halt the run,
    not let the rest of the bundle drain in the pool.

    With a single-worker pool, the failing item is processed first; the remaining
    items sit queued. When ``download_all`` re-raises, ``_attempts``' ``finally``
    cancels them, so most never run. (The one that may already be in-flight when
    the failure surfaces is the small race margin — hence ``< n_items``, not
    ``== 1``.)
    """
    from concurrent.futures import ThreadPoolExecutor

    from MEDS_extract.download import RemoteFile, Source

    n_items = 20
    fetched: list[str] = []

    class FailFirstSource(Source):
        def _list_files(self):
            # The first item carries a "bad" source_path so its _pull raises;
            # everything else is "ok". Numeric prefixes keep the first item
            # first under any stable iteration order.
            return [RemoteFile("00_bad.txt", "bad")] + [
                RemoteFile(f"{i:02d}_ok.txt", "ok") for i in range(1, n_items)
            ]

        def _pull(self, source_path, target):
            if source_path == "bad":
                raise RuntimeError("transport boom")
            fetched.append(target.name)
            target.write_text("ok")

    with ThreadPoolExecutor(max_workers=1) as pool, pytest.raises(RuntimeError, match="transport boom"):
        FailFirstSource().download_all(tmp_path, pool=pool)

    # Without the cancel-on-early-exit ``finally`` in ``_attempts``, all 19 "ok"
    # items would drain through the single worker before the pool shut down.
    assert len(fetched) < n_items - 1, f"expected queued futures cancelled, but {len(fetched)} ran"


def test_download_all_force_overwrite_refetches_complete_file(tmp_path: Path):
    """``do_overwrite=True`` re-fetches a file that's already complete on disk."""
    body = b"hello"
    digest = _sha(body)
    (tmp_path / "x.csv").write_bytes(body)

    n_calls = 0

    def handler(request):
        nonlocal n_calls
        n_calls += 1
        return httpx.Response(200, content=body)

    client = _mock_client(handler)
    src = HTTPSource(
        urls=[{"url": "https://example.com/x.csv", "sha256": digest}],
        client=client,
    )
    # Without overwrite — skipped (no HTTP call).
    src.download_all(tmp_path)
    assert n_calls == 0
    # With overwrite — re-fetched (one HTTP call).
    src.download_all(tmp_path, do_overwrite=True)
    assert n_calls == 1


def test_download_all_force_overwrite_discards_stale_part_when_dest_missing(tmp_path: Path):
    """Regression: ``do_overwrite=True`` must clear ``.part`` even when ``dest`` doesn't
    exist (e.g. a prior failed run left only a partial). Otherwise the next fetch would
    silently Range-resume from the stale prefix, and with sha set we'd get a
    ``ChecksumError`` instead of the intended "force a clean refetch."""
    body = b"clean fresh content here"
    digest = _sha(body)
    url = "https://example.com/x.csv"
    # No ``dest`` — but a stale ``.part`` from a prior failed run.
    (tmp_path / "x.csv.part").write_bytes(b"stale partial bytes")

    seen_range: list[str | None] = []

    def handler(request):
        seen_range.append(request.headers.get("Range"))
        return httpx.Response(200, content=body)

    client = _mock_client(handler)
    src = HTTPSource(urls=[{"url": url, "sha256": digest}], client=client)
    src.download_all(tmp_path, do_overwrite=True)

    assert (tmp_path / "x.csv").read_bytes() == body
    # The stale ``.part`` was cleared before ``_pull``, so no Range header was sent.
    assert seen_range == [None]


def test_http_backend_raises_without_extras(monkeypatch):
    """Importing ``backends.http`` without ``httpx``/``tenacity`` surfaces a clear hint.

    Coverage test for the top-level ``try: import httpx; from tenacity import ...`` guard.
    We blackhole ``httpx`` in ``sys.modules`` and force a reimport so the guard actually
    runs (the module is already cached at test-run time since the extras *are* installed
    for this suite). ``tenacity`` needs the same treatment because the guard short-circuits
    on the first failed import, and we don't want to rely on which package is listed first.
    """
    import importlib
    import sys

    mod_name = "MEDS_extract.download.backends.http"
    monkeypatch.delitem(sys.modules, mod_name, raising=False)
    monkeypatch.setitem(sys.modules, "httpx", None)
    with pytest.raises(ImportError, match=r"MEDS_extract\[download\]"):
        importlib.import_module(mod_name)


def test_http_source_closes_owned_client(tmp_path: Path):
    """``HTTPSource`` owns a client when none is injected and closes it on ``close()``."""
    src = HTTPSource(urls=["https://example.com/a.csv"])
    assert src._owns_client is True
    assert src._client.is_closed is False
    src.close()
    assert src._client.is_closed is True
    # Idempotent: a second close() is a no-op (httpx is re-close-safe).
    src.close()


def test_http_source_does_not_close_injected_client(tmp_path: Path):
    """An injected client belongs to the caller; ``HTTPSource.close()`` leaves it open."""
    client = _mock_client(lambda r: httpx.Response(200, content=b""))
    src = HTTPSource(urls=["https://example.com/a.csv"], client=client)
    assert src._owns_client is False
    src.close()
    assert client.is_closed is False
    client.close()  # caller cleans up


def test_http_source_context_manager_closes_on_exit():
    """``with HTTPSource(...) as src:`` closes owned client on exit."""
    with HTTPSource(urls=["https://example.com/a.csv"]) as src:
        inner = src._client
        assert inner.is_closed is False
    assert inner.is_closed is True


# ── End-to-end CLI demonstration ─────────────────────────────────────────────────────


def test_meds_extract_download_cli_end_to_end(tmp_path: Path):
    """End-to-end ``meds-extract-download`` CLI against a local ``fsspec`` source.

    Runs the ``meds-extract-download`` console entry point as a subprocess, mimicking
    how a downstream ETL would shell out to populate ``raw_input_dir`` before handing
    off to the MEDS_extract stage pipeline. Uses a local directory as the source so the
    test needs no network and still exercises the full Hydra → ``sources_from_spec`` →
    ``Source.download_all`` → ``FsspecSource._pull`` path.
    """
    import subprocess

    # 1. Build a local "release" directory that stands in for a PhysioNet/cloud mirror.
    source_dir = tmp_path / "upstream_mirror"
    source_dir.mkdir()
    (source_dir / "patients.csv").write_text("patient_id,dob\n1,2000-01-01\n2,1990-05-05\n")
    (source_dir / "labs").mkdir()
    (source_dir / "labs" / "vitals.csv").write_text("pid,time,hr\n1,2024-01-01 08:00,82\n")

    # 2. Write a MESSY-style spec with a ``sources:`` block (the CLI only reads that
    # block; the rest of a real MESSY file is irrelevant to the download stage).
    spec_fp = tmp_path / "event_configs.yaml"
    spec_fp.write_text(
        f"""
sources:
  dataset:
    - type: fsspec
      root: {source_dir}
"""
    )

    # 3. Invoke the CLI binary as a subprocess, resolving ``spec`` and ``raw_input_dir``
    # through Hydra's dotlist override syntax — exactly how users will run it.
    raw_input_dir = tmp_path / "raw"
    result = subprocess.run(
        [
            "meds-extract-download",
            f"spec={spec_fp}",
            f"raw_input_dir={raw_input_dir}",
            "hydra.run.dir=" + str(tmp_path / ".hydra"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"CLI failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    # 4. Every file from the upstream mirror landed under ``raw_input_dir`` at its
    # expected relative path. No ``.part`` files remain.
    assert (raw_input_dir / "patients.csv").read_text().startswith("patient_id,dob")
    assert (raw_input_dir / "labs" / "vitals.csv").read_text().startswith("pid,time,hr")
    assert not any(raw_input_dir.rglob("*.part"))

    # 5. Re-running with ``do_overwrite=false`` (default) is idempotent — a same-size,
    # same-content file on disk is taken as already-complete and skipped. We verify by
    # re-running the CLI and confirming the file mtime didn't change (skipped, not
    # rewritten).
    patients_fp = raw_input_dir / "patients.csv"
    mtime_before = patients_fp.stat().st_mtime
    subprocess.run(
        [
            "meds-extract-download",
            f"spec={spec_fp}",
            f"raw_input_dir={raw_input_dir}",
            "hydra.run.dir=" + str(tmp_path / ".hydra2"),
        ],
        check=True,
        capture_output=True,
    )
    assert patients_fp.stat().st_mtime == mtime_before, "skipped file should not be rewritten"

    # 6. ``do_overwrite=true`` forces a re-fetch even when the on-disk content matches —
    # mtime changes and any stale local modification is overwritten by the upstream copy.
    patients_fp.write_text("local_edits_that_should_be_blown_away" + "X" * 100)
    subprocess.run(
        [
            "meds-extract-download",
            f"spec={spec_fp}",
            f"raw_input_dir={raw_input_dir}",
            "do_overwrite=true",
            "hydra.run.dir=" + str(tmp_path / ".hydra3"),
        ],
        check=True,
        capture_output=True,
    )
    assert patients_fp.read_text().startswith("patient_id,dob"), "do_overwrite should re-fetch"


def test_cli_only_resolves_sources_subtree(tmp_path: Path):
    """Regression for the symmetric OmegaConf-resolution problem on the CLI side.

    ``MessyConfig.parse`` strips ``sources`` before ``resolve=True`` so the pipeline
    doesn't need download-only env vars set. This is the mirror: the download CLI must
    resolve ONLY the ``sources:`` subtree, so an unrelated ``${oc.env:...}``
    interpolation in the event-conversion section of the combined MESSY file does not
    break ``meds-extract-download``.

    We verify end-to-end with a real console-script subprocess: combined MESSY with
    event-conversion ``${oc.env:UNRELATED_UNSET}`` that is never set anywhere. If the
    CLI were still resolving the whole file, this would fail with an
    ``InterpolationResolutionError``.
    """
    import os
    import subprocess

    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "hello.csv").write_text("a,b\n1,2\n")

    spec_fp = tmp_path / "messy.yaml"
    spec_fp.write_text(
        f"""
sources:
  dataset:
    - type: fsspec
      root: {mirror}

# event-conversion side references an env var that is never set — the CLI must not
# try to resolve this when loading sources.
_defaults:
  subject_id: $patient_id

patients:
  dob:
    code: DOB
    time: ${{oc.env:UNRELATED_UNSET}}
"""
    )

    raw_input_dir = tmp_path / "raw"
    env = {k: v for k, v in os.environ.items() if k != "UNRELATED_UNSET"}
    result = subprocess.run(
        [
            "meds-extract-download",
            f"spec={spec_fp}",
            f"raw_input_dir={raw_input_dir}",
            "hydra.run.dir=" + str(tmp_path / ".hydra"),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        "CLI failed; likely resolved the whole MESSY file instead of only the "
        f"``sources:`` subtree.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert (raw_input_dir / "hello.csv").read_text().startswith("a,b")


# ── Robustness: loop bounds, SIGINT escape ─────────────────────────────────────────


def test_resumable_stream_bounded_restart(tmp_path):
    """Defense-in-depth: if the ``resume_from = 0`` restart invariant ever breaks, the
    range-resume loop must surface a ``RuntimeError`` instead of looping forever.

    We exercise the cap by (a) lowering ``_MAX_RESUME_ATTEMPTS`` to 1 and (b) seeding a
    pre-existing ``target`` file (so ``resume_from > 0`` and the 416 branch fires on
    the first iteration). The single allowed iteration hits the 416 restart path
    (``continue``) and the loop falls through to the RuntimeError branch before a
    second iteration could produce output.
    """
    from MEDS_extract.download.backends import http as http_mod

    url = "https://example.com/x.csv"
    target = tmp_path / "x.csv.part"
    target.write_bytes(b"stale")  # makes resume_from > 0

    def handler(request):
        # Always 416 — forces the restart branch inside the loop body on every iteration.
        return httpx.Response(416)

    client = _mock_client(handler)
    original = http_mod._MAX_RESUME_ATTEMPTS
    http_mod._MAX_RESUME_ATTEMPTS = 1
    try:
        with pytest.raises(RuntimeError, match=r"exhausted .* restart attempts"):
            _resumable_stream(client, url, target)
    finally:
        http_mod._MAX_RESUME_ATTEMPTS = original


def test_download_all_sigint_cancels_queued_work(tmp_path: Path):
    """Regression for the ``ThreadPoolExecutor(wait=True)`` SIGINT-blocks-for-hours trap.

    Without the fix, Ctrl+C during a slow parallel run would wait for *every* queued
    + in-flight worker to complete — a multi-GiB PhysioNet download at ~30 KB/s per
    connection is literally hours. With ``shutdown(wait=False, cancel_futures=True)``,
    only the in-flight batch remains (and those are daemon threads that get
    abandoned at interpreter teardown).

    The child script is in ``tests/_fetcher_sigint_child.py`` — see that file for
    the design reasoning (why subprocess, why file-count as the signal rather than
    wall-clock, etc.). A subprocess is genuinely necessary here: pytest itself
    catches ``KeyboardInterrupt`` and ends the session, so there's no way to
    observe real SIGINT semantics from inside a pytest worker.
    """
    import subprocess
    import sys as _sys

    if _sys.platform == "win32":
        pytest.skip("SIGINT semantics differ on Windows")

    import pathlib as _pathlib

    child_script = _pathlib.Path(__file__).parent / "_fetcher_sigint_child.py"
    result = subprocess.run(
        [_sys.executable, str(child_script), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,  # hard ceiling in case the fix regresses
    )

    assert result.returncode == 0, (
        f"subprocess did not exit cleanly after SIGINT "
        f"(returncode={result.returncode}):\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # With the fix, at most ``max_workers`` files are in flight when SIGINT fires; a
    # handful more can race to completion before the thread pool winds down. 20 is a
    # comfortable upper bound still far below the child's 100 queued files. Without
    # the fix, n_written ≈ 100 (the full queue drains before exit).
    n_written = len(list(tmp_path.rglob("file_*.txt")))
    assert n_written < 20, (
        f"expected fewer than 20 files written after SIGINT (got {n_written} of 100) — "
        f"pool.shutdown(wait=True) looks like it drained the whole queue instead of "
        f"cancelling it.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ── Streaming retry + content-coding ─────────────────────────────────────────────────


def test_pull_retries_transient_5xx_then_succeeds(tmp_path: Path):
    """A transient 503 on the streaming request is retried (same policy as manifest GETs) rather than failing
    the file on first occurrence."""
    body = b"payload bytes"
    calls: list[int] = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503, text="try later")
        return httpx.Response(200, content=body)

    src = HTTPSource(
        urls=[{"url": "https://example.com/x.csv", "sha256": _sha(body)}],
        transport=httpx.MockTransport(handler),
        max_attempts=3,
        retry_wait=wait_fixed(0),
    )
    src.download_all(tmp_path)
    assert (tmp_path / "x.csv").read_bytes() == body
    assert len(calls) == 3


def test_pull_does_not_retry_4xx(tmp_path: Path):
    """4xx on the streaming request fails immediately — retrying a bad URL or bad auth makes things worse, not
    better."""
    calls: list[int] = []

    def handler(request):
        calls.append(1)
        return httpx.Response(404)

    src = HTTPSource(
        urls=["https://example.com/missing.csv"],
        transport=httpx.MockTransport(handler),
        max_attempts=5,
        retry_wait=wait_fixed(0),
    )
    with pytest.raises(httpx.HTTPStatusError):
        src.download_all(tmp_path)
    assert len(calls) == 1


def test_pull_retry_resumes_from_partial_bytes(tmp_path: Path):
    """A mid-body failure leaves the enlarged ``.part`` in place, so the retried attempt resumes via ``Range``
    instead of restarting from byte 0.

    The truncation point must exceed ``_resumable_stream``'s 1 MiB chunk size —
    httpx buffers smaller partials internally, so bytes only reach disk once at
    least one full chunk has been yielded.
    """
    chunk = 1024 * 1024
    body = b"x" * chunk + b"the tail that only arrives on the second attempt"
    state = {"first": True}
    seen_range: list[str | None] = []

    def handler(request):
        seen_range.append(request.headers.get("Range"))
        if state["first"]:
            state["first"] = False
            # First attempt: serve exactly one full chunk then die mid-stream.
            return httpx.Response(
                200,
                content=_TruncatingStream(body[:chunk]),
            )
        start = int(request.headers["Range"].removeprefix("bytes=").rstrip("-"))
        return httpx.Response(
            206,
            content=body[start:],
            headers={"Content-Range": f"bytes {start}-{len(body) - 1}/{len(body)}"},
        )

    src = HTTPSource(
        urls=[{"url": "https://example.com/x.bin", "sha256": _sha(body)}],
        transport=httpx.MockTransport(handler),
        max_attempts=3,
        retry_wait=wait_fixed(0),
    )
    src.download_all(tmp_path)
    assert (tmp_path / "x.bin").read_bytes() == body
    # Second request resumed from the one full chunk the first attempt wrote.
    assert seen_range == [None, f"bytes={chunk}-"]


class _TruncatingStream(httpx.SyncByteStream):
    """Yields a prefix of the body, then raises a transient transport error."""

    def __init__(self, prefix: bytes):
        self._prefix = prefix

    def __iter__(self):
        yield self._prefix
        raise httpx.RemoteProtocolError("connection dropped mid-body")


def test_resumable_stream_requests_identity_encoding(tmp_path: Path):
    """Streaming requests must send ``Accept-Encoding: identity``: transparent gzip would decouple on-disk
    byte counts from the Range/Content-Range byte space and corrupt resume offsets."""
    body = b"plain bytes"
    seen: list[str | None] = []

    def handler(request):
        seen.append(request.headers.get("Accept-Encoding"))
        return httpx.Response(200, content=body)

    client = _mock_client(handler)
    target = tmp_path / "x.csv.part"
    _resumable_stream(client, "https://example.com/x.csv", target)
    assert target.read_bytes() == body
    assert seen == ["identity"]


def test_download_from_gzip_capable_server(tmp_path: Path):
    """End-to-end against a server that would gzip-encode if invited: the stored
    bytes and the manifest sha must both describe the identity representation."""
    import gzip

    body = b"a,b,c\n" * 2000  # compressible

    def handler(request):
        if "gzip" in request.headers.get("Accept-Encoding", ""):
            return httpx.Response(200, content=gzip.compress(body), headers={"Content-Encoding": "gzip"})
        return httpx.Response(200, content=body)

    src = HTTPSource(
        urls=[{"url": "https://example.com/x.csv", "sha256": _sha(body)}],
        client=_mock_client(handler),
    )
    src.download_all(tmp_path)
    assert (tmp_path / "x.csv").read_bytes() == body


# ── .part promotion + manifest filtering + auth plumbing ─────────────────────────────


def test_complete_part_promoted_without_refetch(tmp_path: Path):
    """A leftover ``.part`` that already verifies against the manifest sha (prior run died between last byte
    and rename) is promoted to ``dest`` with zero HTTP calls — not deleted by a 416 bounce and re-
    downloaded."""
    body = b"the whole file, fully written"
    (tmp_path / "x.csv.part").write_bytes(body)
    n_calls = 0

    def handler(request):
        nonlocal n_calls
        n_calls += 1
        return httpx.Response(200, content=body)

    src = HTTPSource(
        urls=[{"url": "https://example.com/x.csv", "sha256": _sha(body)}],
        client=_mock_client(handler),
    )
    src.download_all(tmp_path)
    assert (tmp_path / "x.csv").read_bytes() == body
    assert not (tmp_path / "x.csv.part").exists()
    assert n_calls == 0


def test_physionet_include_filter_fetches_subset(tmp_path: Path):
    """``include=`` globs subset a SHA256SUMS manifest — only matching files are listed or fetched."""
    files = {
        "hosp/patients.csv.gz": b"p",
        "hosp/labevents.csv.gz": b"l",
        "waveforms/w0001.dat": b"w" * 64,
    }
    manifest = "\n".join(f"{_sha(b)}  {rel}" for rel, b in files.items()) + "\n"
    fetched: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/SHA256SUMS.txt"):
            return httpx.Response(200, text=manifest)
        for rel, b in files.items():
            if path.endswith("/" + rel):
                fetched.append(rel)
                return httpx.Response(200, content=b)
        return httpx.Response(404)

    src = PhysioNetSource(
        base_url="https://physionet.org/files/demo/1.0",
        client=_mock_client(handler),
        include=["hosp/*"],
    )
    src.download_all(tmp_path)
    assert sorted(fetched) == ["hosp/labevents.csv.gz", "hosp/patients.csv.gz"]
    assert not (tmp_path / "waveforms").exists()


def test_physionet_basic_auth_sent_on_wire(tmp_path: Path):
    """Credentials passed as ``username=``/``password=`` must surface as an ``Authorization: Basic`` header on
    both the manifest GET and the file streams.

    Constructed via the public ``transport=`` kwarg (NOT ``client=``) so the real
    auth-construction path in ``_make_client`` is exercised.
    """
    import base64

    body = b"data"
    manifest = f"{_sha(body)}  data.csv\n"
    seen_auth: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("Authorization"))
        if request.url.path.endswith("/SHA256SUMS.txt"):
            return httpx.Response(200, text=manifest)
        return httpx.Response(200, content=body)

    with PhysioNetSource(
        base_url="https://physionet.org/files/restricted/1.0",
        username="alice",
        password="s3cret",
        transport=httpx.MockTransport(handler),
    ) as src:
        src.download_all(tmp_path)

    expected = "Basic " + base64.b64encode(b"alice:s3cret").decode()
    assert len(seen_auth) == 2  # manifest GET + one file stream
    assert all(h == expected for h in seen_auth)


def test_fsspec_source_memory_protocol(tmp_path: Path):
    """FsspecSource against a non-local protocol (in-process ``memory://``): locks the ``source_path=str(p)``
    → ``UPath(source_path)`` round-trip that every remote root depends on, without needing a network."""
    from upath import UPath

    from MEDS_extract.download import FsspecSource

    root = UPath("memory://mirror")
    (root / "sub").mkdir(parents=True, exist_ok=True)
    (root / "patients.csv").write_bytes(b"patient_id\n1\n")
    (root / "sub" / "vitals.csv").write_bytes(b"pid,hr\n1,80\n")

    FsspecSource(root="memory://mirror").download_all(tmp_path)
    assert (tmp_path / "patients.csv").read_bytes() == b"patient_id\n1\n"
    assert (tmp_path / "sub" / "vitals.csv").read_bytes() == b"pid,hr\n1,80\n"


def test_download_all_pooled_multiworker_end_to_end(tmp_path: Path):
    """Real multi-worker parallelism through a real backend: concurrent
    ``_fetch_one`` staging (shared client, sibling-dir mkdir races, per-file
    ``.part`` + rename) must produce exactly the manifest, with no leftovers."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    n = 24
    bodies = {f"https://example.com/sub{i % 3}/f{i:02d}.bin": f"body {i}".encode() for i in range(n)}
    lock = threading.Lock()
    served: list[str] = []

    def handler(request):
        with lock:
            served.append(str(request.url))
        return httpx.Response(200, content=bodies[str(request.url)])

    src = HTTPSource(
        urls=[
            {"url": u, "sha256": _sha(b), "rel_path": u.removeprefix("https://example.com/")}
            for u, b in bodies.items()
        ],
        client=_mock_client(handler),
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        src.download_all(tmp_path, pool=pool)

    assert sorted(served) == sorted(bodies)
    for u, b in bodies.items():
        assert (tmp_path / u.removeprefix("https://example.com/")).read_bytes() == b
    assert not list(tmp_path.rglob("*.part"))


def test_download_all_pooled_continue_on_error_collects_all(tmp_path: Path):
    """Pooled ``continue_on_error=True`` is a distinct path through ``_attempts``:

    errors surface via ``fut.result`` in completion order, the loop must drain fully,
    and the unconditional ``finally: fut.cancel()`` must be a harmless no-op.
    """
    from concurrent.futures import ThreadPoolExecutor

    good = {f"https://example.com/ok{i}.csv": f"ok {i}".encode() for i in range(4)}
    bad = ["https://example.com/bad0.csv", "https://example.com/bad1.csv"]

    def handler(request):
        body = good.get(str(request.url))
        if body is None:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, content=body)

    src = HTTPSource(
        urls=[{"url": u, "sha256": _sha(b)} for u, b in good.items()] + bad,
        client=_mock_client(handler),
        max_attempts=1,
    )
    with (
        ThreadPoolExecutor(max_workers=2) as pool,
        pytest.raises(ExceptionGroup) as exc_info,
    ):
        src.download_all(tmp_path, pool=pool, continue_on_error=True)

    assert len(exc_info.value.exceptions) == 2
    # Every failure carries its fetch-context note.
    assert all(any("while fetching" in n for n in e.__notes__) for e in exc_info.value.exceptions)
    for u, b in good.items():
        assert (tmp_path / u.rsplit("/", 1)[1]).read_bytes() == b
    assert not list(tmp_path.rglob("*.part"))


# ── CLI failure paths ────────────────────────────────────────────────────────────────


def _run_cli(tmp_path: Path, spec_body: str, *args: str, hydra_dir: str = ".hydra"):
    """Run the ``meds-extract-download`` console script against an inline spec."""
    import subprocess

    spec_fp = tmp_path / "spec.yaml"
    spec_fp.write_text(spec_body)
    raw_input_dir = tmp_path / "raw"
    return (
        subprocess.run(
            [
                "meds-extract-download",
                f"spec={spec_fp}",
                f"raw_input_dir={raw_input_dir}",
                "hydra.run.dir=" + str(tmp_path / hydra_dir),
                *args,
            ],
            capture_output=True,
            text=True,
            check=False,
        ),
        raw_input_dir,
    )


def test_cli_failure_exits_nonzero(tmp_path: Path):
    """Regression for the Hydra-discards-return-value bug: a failed download must
    exit non-zero (the fix is an explicit ``sys.exit(1)``), so scripted callers
    (``meds-extract-download && next-step``) stop on failure."""
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "a.csv").write_text("upstream content\n")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.csv").write_text("conflicting local content\n")  # sha mismatch → FileExistsError

    result, _ = _run_cli(tmp_path, f"sources:\n  dataset:\n    - type: fsspec\n      root: {mirror}\n")
    assert result.returncode != 0, f"expected failure exit:\n{result.stdout}\n{result.stderr}"
    assert (raw / "a.csv").read_text() == "conflicting local content\n"  # untouched


def test_cli_unknown_key_exits_nonzero(tmp_path: Path):
    """A typo'd ``key=`` must be an error listing the available buckets — not a silent no-op success
    (``common`` is always appended, so a bad key would otherwise quietly fetch the wrong subset)."""
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "a.csv").write_text("x\n")

    result, raw = _run_cli(
        tmp_path,
        f"sources:\n  dataset:\n    - type: fsspec\n      root: {mirror}\n",
        "key=dtaaset",
    )
    assert result.returncode != 0
    assert "dtaaset" in result.stdout + result.stderr
    assert "dataset" in result.stdout + result.stderr  # available buckets listed
    assert not raw.exists()  # nothing was staged


def test_cli_cross_source_collision_exits_before_any_fetch(tmp_path: Path):
    """Two sources listing the same rel_path into one shared raw_input_dir is a config error caught up-front —
    not a mid-download race/FileExistsError."""
    m1 = tmp_path / "m1"
    m2 = tmp_path / "m2"
    for m in (m1, m2):
        m.mkdir()
        (m / "x.csv").write_text(f"from {m.name}\n")

    result, raw = _run_cli(
        tmp_path,
        f"""sources:
  dataset:
    - type: fsspec
      root: {m1}
    - type: fsspec
      root: {m2}
""",
    )
    assert result.returncode != 0
    assert "Duplicate destination across sources" in result.stdout + result.stderr
    assert not raw.exists() or not any(raw.iterdir())  # failed before any fetch


def test_cli_fail_fast_skips_remaining_sources(tmp_path: Path):
    """With the default ``continue_on_error=false``, a failing source stops the whole run — later sources are
    not attempted.

    With ``continue_on_error=true``, they are.
    """
    m1 = tmp_path / "m1"
    m2 = tmp_path / "m2"
    m1.mkdir()
    m2.mkdir()
    (m1 / "a.csv").write_text("upstream a\n")
    (m2 / "b.csv").write_text("upstream b\n")
    spec = f"""sources:
  dataset:
    - type: fsspec
      root: {m1}
    - type: fsspec
      root: {m2}
"""
    # Sabotage source 1: a conflicting pre-existing dest for a.csv.
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.csv").write_text("conflicting\n")

    result, _ = _run_cli(tmp_path, spec)
    assert result.returncode != 0
    assert not (raw / "b.csv").exists(), "fail-fast must not proceed to source 2"

    result, _ = _run_cli(tmp_path, spec, "continue_on_error=true", hydra_dir=".hydra2")
    assert result.returncode != 0  # source 1 still failed
    assert (raw / "b.csv").read_text() == "upstream b\n"  # but source 2 was attempted
