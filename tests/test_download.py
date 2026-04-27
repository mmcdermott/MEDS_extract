"""Integration tests for :mod:`MEDS_extract.download` using httpx.MockTransport.

Doctests throughout the module cover most pure-Python machinery — dispatch, URL
normalization, hash helpers, SHA256SUMS parsing, and the :meth:`Source.download_all`
skip / overwrite / path-traversal / duplicate-dest paths via the doctest in
``source.py``. This file covers what doctests can't: the ``_resumable_download``
HTTP primitive's wire-level behavior (Range resume, 416/206 mismatch handling,
checksum verify), end-to-end ``download_all`` against ``MockTransport``-backed
:class:`HTTPSource` / :class:`PhysioNetSource`, the CLI subprocess flow, and the
SIGINT-cancellation regression that needs a real signal.

MockTransport intercepts at the httpx level below the client, so retry/timeout/Range
behavior is all exercised against the real client code path.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import httpx
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from MEDS_extract.download import HTTPSource, PhysioNetSource
from MEDS_extract.download.source import ChecksumError

# ``_resumable_download`` lives on HTTPSource as a staticmethod — alias it here so
# the test body reads the same as before.
_resumable_download = HTTPSource._resumable_download


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _mock_client(handler):
    """Build an :class:`httpx.Client` whose transport is a user-supplied handler.

    ``handler(request) -> httpx.Response`` is the same signature ``MockTransport`` expects.
    """
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=10.0)


# ── _resumable_download: core primitive ──────────────────────────────────────────────


def test_resumable_download_writes_file_and_verifies_sha256(tmp_path: Path):
    body = b"hello world"
    digest = _sha(body)
    url = "https://example.com/x.csv"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/x.csv"
        return httpx.Response(200, content=body)

    client = _mock_client(handler)
    dest = tmp_path / "x.csv"
    _resumable_download(client, url, dest, expected_sha256=digest)
    assert dest.read_bytes() == body
    assert not dest.with_name(dest.name + ".part").exists()


def test_resumable_download_checksum_mismatch_raises_and_cleans_part(tmp_path: Path):
    body = b"hello world"
    wrong_digest = "0" * 64
    url = "https://example.com/x.csv"

    def handler(request):
        return httpx.Response(200, content=body)

    client = _mock_client(handler)
    dest = tmp_path / "x.csv"
    with pytest.raises(ChecksumError):
        _resumable_download(client, url, dest, expected_sha256=wrong_digest)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_resumable_download_range_resume_appends(tmp_path: Path):
    """If a ``.part`` exists, send Range: bytes=N- and append to the existing bytes."""
    full_body = b"abcdefghij"
    digest = _sha(full_body)
    url = "https://example.com/x.csv"
    dest = tmp_path / "x.csv"
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(full_body[:4])  # "abcd"

    seen_range: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_range.append(request.headers.get("Range"))
        if request.headers.get("Range") == "bytes=4-":
            return httpx.Response(206, content=full_body[4:], headers={"Content-Range": "bytes 4-9/10"})
        return httpx.Response(200, content=full_body)

    client = _mock_client(handler)
    _resumable_download(client, url, dest, expected_sha256=digest)
    assert dest.read_bytes() == full_body
    assert seen_range == ["bytes=4-"]


def test_resumable_download_range_ignored_restarts_from_scratch(tmp_path: Path):
    """Server returning 200 despite Range header → client restarts from byte 0."""
    body = b"fresh contents"
    digest = _sha(body)
    url = "https://example.com/x.csv"
    dest = tmp_path / "x.csv"
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(b"stale partial")

    def handler(request):
        # Ignore Range, always serve full body with 200.
        return httpx.Response(200, content=body)

    client = _mock_client(handler)
    _resumable_download(client, url, dest, expected_sha256=digest)
    assert dest.read_bytes() == body


# Note: the "skip if dest exists + matches sha" optimization moved out of
# ``_resumable_download`` and onto ``Source._verifies`` (covered by the
# ``Source.download_all`` doctest in src/MEDS_extract/download/source.py). The
# primitive now trusts its caller to have cleared ``dest`` — overwrite semantics
# live in the orchestration loop, not here.


def test_resumable_download_mismatched_content_range_restarts(tmp_path: Path):
    """Server returning 206 with a Content-Range that doesn't start at resume_from must not silently corrupt
    `.part` — the client restarts from byte 0."""
    full_body = b"abcdefghij"
    digest = _sha(full_body)
    url = "https://example.com/x.csv"
    dest = tmp_path / "x.csv"
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(full_body[:4])  # "abcd"

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
    _resumable_download(client, url, dest, expected_sha256=digest)
    assert dest.read_bytes() == full_body
    assert seen == ["bytes=4-", None]


def test_resumable_download_416_restarts(tmp_path: Path):
    """Server 416 (Range Not Satisfiable) — remote file shrank — must restart from byte 0."""
    full_body = b"short"
    digest = _sha(full_body)
    url = "https://example.com/x.csv"
    dest = tmp_path / "x.csv"
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(b"stale longer content")  # larger than current full body

    seen_statuses = []

    def handler(request: httpx.Request) -> httpx.Response:
        rng = request.headers.get("Range")
        if rng:
            seen_statuses.append(416)
            return httpx.Response(416)
        seen_statuses.append(200)
        return httpx.Response(200, content=full_body)

    client = _mock_client(handler)
    _resumable_download(client, url, dest, expected_sha256=digest)
    assert dest.read_bytes() == full_body
    assert seen_statuses == [416, 200]


def test_resumable_download_stale_dest_rewritten(tmp_path: Path):
    """Existing ``dest`` with wrong SHA-256 is deleted and refetched."""
    correct = b"correct"
    digest = _sha(correct)
    url = "https://example.com/x.csv"
    dest = tmp_path / "x.csv"
    dest.write_bytes(b"stale")

    def handler(request):
        return httpx.Response(200, content=correct)

    client = _mock_client(handler)
    _resumable_download(client, url, dest, expected_sha256=digest)
    assert dest.read_bytes() == correct


def test_resumable_download_discards_part_when_no_sha(tmp_path: Path):
    """Without ``expected_sha256``, a pre-existing ``.part`` is unsafe to resume from — the stale prefix could
    no longer match the current remote content.

    The primitive must discard the
    ``.part`` and fetch from byte 0.
    """
    body = b"the real full content"
    url = "https://example.com/x.csv"
    dest = tmp_path / "x.csv"
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(b"stale partial")  # from a prior run, no checksum to verify it

    seen_range: list[str | None] = []

    def handler(request):
        seen_range.append(request.headers.get("Range"))
        return httpx.Response(200, content=body)

    client = _mock_client(handler)
    _resumable_download(client, url, dest, expected_sha256=None)
    assert dest.read_bytes() == body
    # No Range header was sent — we started fresh.
    assert seen_range == [None]
    assert not part.exists()


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
        files = src.list_files()
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


def test_download_all_refuses_to_overwrite_wrong_size_file(tmp_path: Path):
    """A wrong-size file at dest with ``do_overwrite=False`` raises ``FileExistsError``.

    The pre-fix behavior was to silently overwrite — that path masked stale or
    partially-flushed local copies. The strict semantic: only verified-match skips,
    only ``do_overwrite=True`` overwrites; everything else errors.
    """
    full_body = b"correct content (20 bytes).."
    url = "https://example.com/x.csv"
    dest = tmp_path / "x.csv"
    dest.write_bytes(b"wrong_contents")  # different size from full_body

    def handler(request):
        return httpx.Response(200, content=full_body)

    client = _mock_client(handler)
    src = HTTPSource(
        urls=[{"url": url, "size": len(full_body)}],  # size specified, no sha
        client=client,
    )
    with pytest.raises(FileExistsError, match="does not verify"):
        src.download_all(tmp_path)
    # Stale local copy was not touched.
    assert dest.read_bytes() == b"wrong_contents"

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
    src = HTTPSource(urls=["https://example.com/x.csv"], client=client)
    with pytest.raises(httpx.HTTPStatusError):
        src.download_all(tmp_path)


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
        urls=[{"url": "https://example.com/x.csv", "sha256": digest, "size": len(body)}],
        client=client,
    )
    # Without overwrite — skipped (no HTTP call).
    src.download_all(tmp_path)
    assert n_calls == 0
    # With overwrite — re-fetched (one HTTP call).
    src.download_all(tmp_path, do_overwrite=True)
    assert n_calls == 1


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
    ``Source.download_all`` → ``FsspecSource._fetch`` path.
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


def test_cli_python_m_propagates_nonzero_exit(tmp_path: Path):
    """Regression: ``python -m MEDS_extract.download.cli`` must return the Hydra-main
    function's exit code, not swallow it.

    Setuptools wraps console-script entry points in ``sys.exit(func())`` automatically,
    but the ``__main__`` block in ``cli.py`` has to do that manually for the
    ``python -m`` path. Earlier iterations called bare ``main()`` there, which silently
    exited 0 on every failure and masked real errors in subprocess-style tests.
    """
    import subprocess
    import sys as _sys

    missing_spec = tmp_path / "does_not_exist.yaml"
    # Pointing at a non-existent spec reliably triggers FileNotFoundError inside main,
    # which Hydra surfaces as a non-zero exit. Pre-fix, this would have exited 0.
    result = subprocess.run(
        [
            _sys.executable,
            "-m",
            "MEDS_extract.download.cli",
            f"spec={missing_spec}",
            f"raw_input_dir={tmp_path / 'raw'}",
            "hydra.run.dir=" + str(tmp_path / ".hydra"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, (
        f"expected non-zero exit when spec is missing; got returncode=0.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ── Robustness: loop bounds, SIGINT escape ─────────────────────────────────────────


def test_resumable_download_bounded_restart(tmp_path):
    """Defense-in-depth: if the ``resume_from = 0`` restart invariant ever breaks, the
    range-resume loop must surface a ``RuntimeError`` instead of looping forever.

    We exercise the cap by (a) lowering ``_MAX_RESUME_ATTEMPTS`` to 1 and (b) seeding a
    ``.part`` file with a known SHA-256 (so the no-sha branch doesn't pre-clear it),
    then serving 416 on every Range request. The single allowed iteration hits the 416
    restart path (``continue``) and the ``for ... else:`` fires with our RuntimeError
    before a second iteration could produce output.
    """
    from MEDS_extract.download.backends import http as http_mod

    body = b"hello world"
    digest = _sha(body)
    url = "https://example.com/x.csv"
    dest = tmp_path / "x.csv"
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(b"stale")  # makes resume_from > 0

    def handler(request):
        # Always 416 — forces the restart branch inside the loop body on every iteration.
        return httpx.Response(416)

    client = _mock_client(handler)
    original = http_mod._MAX_RESUME_ATTEMPTS
    http_mod._MAX_RESUME_ATTEMPTS = 1
    try:
        with pytest.raises(RuntimeError, match=r"exhausted .* restart attempts"):
            _resumable_download(client, url, dest, expected_sha256=digest)
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
