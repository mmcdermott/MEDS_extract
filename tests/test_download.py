"""Integration tests for :mod:`MEDS_extract.download` using httpx.MockTransport.

Doctests throughout the module cover most pure-Python machinery — spec dispatch, URL
normalization, hash helpers, ``RemoteFile`` validation, SHA256SUMS parsing, manifest
filtering, client lifecycle (``HTTPSource.close``), and the whole
:meth:`Source._fetch_one` / :meth:`Source.download_all` policy surface
(skip / overwrite / ``.part`` staging + promotion / checksum / failure-collection /
path-traversal / duplicate-dest) via the doctests in ``source.py``. This file covers
what doctests can't express cleanly: the ``_resumable_stream`` HTTP primitive's
wire-level behavior (Range resume, 416/206 mismatch handling, gzip-vs-identity
content-coding), streaming retry with multi-request handler state machines,
end-to-end ``download_all`` against ``MockTransport``-backed :class:`HTTPSource` /
:class:`PhysioNetSource` (sequential and pooled), and the SIGINT-cancellation
regression that needs a real signal. CLI subprocess flows (success, failure exit
codes, key validation, collision rejection) live in ``tests/test_download_fsspec.py``
so the no-extras CI job can run them.

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
# its wire-level behavior (Range resume, 416/206 mismatch handling); the basic
# stream-to-target contract and the ``Accept-Encoding: identity`` invariant are its
# own doctest. SHA verify and atomic rename live on ``Source._fetch_one`` and are
# covered by the doctests in ``source.py``.


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


# ── HTTPSource end-to-end through Source.download_all ────────────────────────────────
# The ``_fetch_one`` staging pipeline itself (stale-.part discard, sha verify, atomic
# rename, complete-.part promotion) is covered by stub-Source doctests in ``source.py``;
# these tests exercise it end-to-end through the real HTTP ``_pull``.


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


def test_physionet_rel_path_is_percent_encoded_on_wire(tmp_path: Path):
    """Regression for the ``quote(rel_path, safe='/')`` fix: a manifest rel_path containing ``#`` (parsed as a
    fragment) or a space must be percent-encoded on the wire, or the request silently fetches the wrong
    resource."""
    body = b"odd payload"
    manifest = f"{_sha(body)}  odd dir/file#1.csv\n"
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        if str(request.url).endswith("/SHA256SUMS.txt"):
            return httpx.Response(200, text=manifest)
        return httpx.Response(200, content=body)

    src = PhysioNetSource(base_url="https://physionet.org/files/demo/1.0", client=_mock_client(handler))
    src.download_all(tmp_path)

    assert any("odd%20dir/file%231.csv" in u for u in seen_urls), seen_urls
    assert (tmp_path / "odd dir" / "file#1.csv").read_bytes() == body


def test_physionet_manifest_get_retries_5xx_with_injected_client():
    """Manifest-GET retry must apply to injected clients too — the policy lives on the source
    (``HTTPSource._get``), not on a monkeypatched ``client.get``, so ``client=`` injection cannot silently
    lose retries."""
    body = b"x"
    manifest = f"{_sha(body)}  a.csv\n"
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(503, text="try later")
        return httpx.Response(200, text=manifest)

    src = PhysioNetSource(
        base_url="https://physionet.org/files/demo/1.0",
        client=_mock_client(handler),
        retry_wait=wait_fixed(0),
    )
    assert [f.rel_path for f in src.files] == ["a.csv"]
    assert len(attempts) == 3


def test_physionet_manifest_get_does_not_retry_4xx():
    """A 4xx on the manifest GET fails fast (one attempt) — same never-retry-4xx semantics as every other
    request path."""
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(404)

    src = PhysioNetSource(
        base_url="https://physionet.org/files/demo/1.0",
        client=_mock_client(handler),
        retry_wait=wait_fixed(0),
    )
    with pytest.raises(httpx.HTTPStatusError):
        src.files  # noqa: B018 — the cached property does the manifest GET
    assert len(attempts) == 1


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
    only the in-flight batch remains (the non-daemon workers are still joined at
    interpreter teardown, but with the queue cancelled that join covers at most
    the in-flight batch).

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


@pytest.mark.parametrize("exc_type", [httpx.RemoteProtocolError, httpx.ReadError])
def test_pull_retry_resumes_from_partial_bytes(tmp_path: Path, exc_type):
    """A mid-body failure leaves the enlarged ``.part`` in place, so the retried attempt resumes via ``Range``
    instead of restarting from byte 0.

    Parametrized over both mid-body failure shapes: a clean premature FIN
    (``RemoteProtocolError``) and a TCP reset (``ReadError`` — the common
    real-world transient, which must also be in the retry set).

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
                content=_TruncatingStream(body[:chunk], exc_type),
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


def test_failed_download_leaves_part_for_future_resume(tmp_path: Path):
    """A fully-exhausted transport failure must leave the ``.part`` on disk — the documented head start for a
    future run's Range-resume.

    (Success and checksum-failure paths clean it up; transport failure deliberately does not.)
    """

    def handler(request):
        return httpx.Response(200, content=_TruncatingStream(b"x" * 1024 * 1024, httpx.ReadError))

    src = HTTPSource(
        urls=["https://example.com/big.bin"],
        transport=httpx.MockTransport(handler),
        max_attempts=2,
        retry_wait=wait_fixed(0),
    )
    with pytest.raises(httpx.ReadError):
        src.download_all(tmp_path)
    assert (tmp_path / "big.bin.part").exists(), ".part must survive for cross-run resume"
    assert not (tmp_path / "big.bin").exists()


class _TruncatingStream(httpx.SyncByteStream):
    """Yields a prefix of the body, then raises a transient transport error."""

    def __init__(self, prefix: bytes, exc_type: type[Exception] = httpx.RemoteProtocolError):
        self._prefix = prefix
        self._exc_type = exc_type

    def __iter__(self):
        yield self._prefix
        raise self._exc_type("connection dropped mid-body")


def test_gzip_capable_server_resume_not_corrupted(tmp_path: Path):
    """Resume against an RFC-compliant server that gzip-encodes when invited.

    ``Range`` applies to the *encoded* representation, while ``.part`` sizes count
    *decoded* bytes — so if the client invited gzip (i.e. the identity fix were
    reverted), the resume request would fetch a mid-stream slice of the gzip
    representation whose ``Content-Range`` start still matches, and decoding
    would fail (or corrupt). With ``Accept-Encoding: identity``, byte spaces
    coincide and the resume completes byte-perfect.
    """
    import gzip

    body = b"a,b,c\n" * 2000  # compressible
    resume_at = 1000
    (tmp_path / "x.csv.part").write_bytes(body[:resume_at])  # prior run's partial

    def handler(request):
        accepts_gzip = "gzip" in request.headers.get("Accept-Encoding", "")
        representation = gzip.compress(body) if accepts_gzip else body
        headers = {"Content-Encoding": "gzip"} if accepts_gzip else {}
        rng = request.headers.get("Range")
        if rng:
            start = int(rng.removeprefix("bytes=").rstrip("-"))
            return httpx.Response(
                206,
                content=representation[start:],
                headers={
                    **headers,
                    "Content-Range": f"bytes {start}-{len(representation) - 1}/{len(representation)}",
                },
            )
        return httpx.Response(200, content=representation, headers=headers)

    src = HTTPSource(
        urls=[{"url": "https://example.com/x.csv", "sha256": _sha(body)}],
        client=_mock_client(handler),
        max_attempts=1,  # no second chances: the first (resume) attempt must be clean
    )
    src.download_all(tmp_path)
    assert (tmp_path / "x.csv").read_bytes() == body


# ── Manifest filtering + auth plumbing ───────────────────────────────────────────────


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
