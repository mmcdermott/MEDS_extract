"""Integration tests for :mod:`MEDS_extract.download` using httpx.MockTransport.

Doctests throughout the module cover the small pure functions (dispatch, URL
normalization, hash helpers, SHA256SUMS parsing). This file covers the pieces that
need a real httpx round-trip: the ``_resumable_download`` primitive, the ``HTTPSource``
+ ``PhysioNetSource`` flow through ``Fetcher``, and the atomic-rename / checksum-verify
/ Range-resume invariants.

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

from MEDS_extract.download import (
    Fetcher,
    HTTPSource,
    PhysioNetSource,
    RemoteFile,
)
from MEDS_extract.download.backends.fsspec import FsspecSource
from MEDS_extract.download.source import ChecksumError

# ``_resumable_download`` moved onto HTTPSource as a staticmethod — alias it here so the
# test body reads the same as before.
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


def test_resumable_download_skips_existing_correct_file(tmp_path: Path):
    """If ``dest`` already exists and matches the sha, no HTTP call is made."""
    body = b"already here"
    digest = _sha(body)
    url = "https://example.com/x.csv"
    dest = tmp_path / "x.csv"
    dest.write_bytes(body)

    calls = []

    def handler(request):
        calls.append(request.url)
        return httpx.Response(200, content=body)

    client = _mock_client(handler)
    _resumable_download(client, url, dest, expected_sha256=digest)
    assert calls == []  # no transport call


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


# ── HTTPSource end-to-end through Fetcher ────────────────────────────────────────────


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
    report = Fetcher(tmp_path).fetch_all(src)

    assert report.n_downloaded == 2
    assert report.n_failed == 0
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
    Fetcher(tmp_path).fetch_all(src)
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
        Fetcher(tmp_path).fetch_all(src)


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
    report = Fetcher(tmp_path).fetch_all(src)

    assert report.n_downloaded == 2
    assert report.n_failed == 0
    assert (tmp_path / "patients.csv").read_bytes() == files["patients.csv"]
    assert (tmp_path / "labs" / "vitals.csv").read_bytes() == files["labs/vitals.csv"]


def test_physionet_source_trailing_slash_normalization():
    """Base URL is normalized to include a trailing slash so URL concatenation is clean."""
    client = _mock_client(lambda r: httpx.Response(404))
    a = PhysioNetSource(base_url="https://example.com/files/x", client=client)
    b = PhysioNetSource(base_url="https://example.com/files/x/", client=client)
    assert a._base_url == b._base_url == "https://example.com/files/x/"


# ── Fetcher: integration with RemoteFile.size / SHA256 skip paths ────────────────────


def test_fetcher_skips_when_size_and_sha256_match(tmp_path: Path):
    body = b"already downloaded"
    digest = _sha(body)

    # Pre-populate the file.
    (tmp_path / "x.txt").write_bytes(body)

    class Src:
        def list_files(self):
            return [RemoteFile(rel_path="x.txt", size=len(body), sha256=digest)]

        def fetch(self, remote, dest):
            raise RuntimeError("fetch should not be called — file is already complete.")

    report = Fetcher(tmp_path).fetch_all(Src())
    assert report.n_skipped == 1
    assert report.n_downloaded == 0


def test_fetcher_rejects_path_traversal(tmp_path: Path):
    """A malformed/malicious manifest pointing outside ``dest_dir`` must be rejected."""

    class EscapingSource:
        def list_files(self):
            return [RemoteFile(rel_path="../../etc/passwd")]

        def fetch(self, remote, dest):
            dest.write_text("never reached")

    with pytest.raises(ValueError, match="escapes dest_dir"):
        Fetcher(tmp_path).fetch_all(EscapingSource())


def test_fetcher_rejects_absolute_path(tmp_path: Path):
    class AbsSource:
        def list_files(self):
            return [RemoteFile(rel_path="/etc/passwd")]

        def fetch(self, remote, dest):
            dest.write_text("never reached")

    with pytest.raises(ValueError, match="must be relative"):
        Fetcher(tmp_path).fetch_all(AbsSource())


def test_fsspec_source_verifies_sha256(tmp_path: Path):
    """FsspecSource must honor remote.sha256 the same way HTTP-backed sources do."""
    body = b"hello fsspec"
    correct_digest = _sha(body)

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "x.txt").write_bytes(body)

    # Correct hash: succeeds silently.
    source = FsspecSource(root=str(src_dir))
    (tmp_path / "out_good").mkdir()
    source.fetch(
        RemoteFile("x.txt", sha256=correct_digest, extra={"upath": src_dir / "x.txt"}),
        tmp_path / "out_good" / "x.txt",
    )
    assert (tmp_path / "out_good" / "x.txt").read_bytes() == body

    # Wrong hash: ChecksumError, dest is not created, .part cleaned up.
    (tmp_path / "out_bad").mkdir()
    dest = tmp_path / "out_bad" / "x.txt"
    with pytest.raises(ChecksumError):
        source.fetch(RemoteFile("x.txt", sha256="0" * 64, extra={"upath": src_dir / "x.txt"}), dest)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_physionet_source_rejects_half_credentials():
    """Username without password (or vice versa) is a clear user error — fail fast."""
    with pytest.raises(ValueError, match="must be supplied together"):
        PhysioNetSource(base_url="https://example.com/files/x", username="u", password=None)
    with pytest.raises(ValueError, match="must be supplied together"):
        PhysioNetSource(base_url="https://example.com/files/x", username=None, password="p")


def test_fetcher_refetches_wrong_size_file_without_sha(tmp_path: Path):
    """Regression: when ``remote.size`` is set but ``sha256`` isn't, a wrong-size local
    file must be deleted + refetched, not left in place.

    The underlying bug: ``_resumable_download`` early-returns when ``dest`` exists and
    ``expected_sha256`` is ``None``, so size-only validation relied on the caller to
    clean the stale file first.
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
    report = Fetcher(tmp_path).fetch_all(src)

    assert report.n_downloaded == 1
    assert report.n_skipped == 0
    assert dest.read_bytes() == full_body  # actually refetched, not left stale


def test_fetcher_continue_on_error_captures_failures(tmp_path: Path):
    """With ``continue_on_error=True``, per-file failures are captured in the report."""
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
    report = Fetcher(tmp_path, continue_on_error=True).fetch_all(src)
    assert report.n_downloaded == 1
    assert report.n_failed == 1
    assert not report.ok
    # The good one still landed:
    assert (tmp_path / "good.csv").read_bytes() == body
