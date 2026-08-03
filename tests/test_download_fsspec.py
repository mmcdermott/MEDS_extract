"""Extras-free ``FsspecSource`` / ``Source``-level download tests.

Everything here runs without the ``download`` extra (no ``httpx``/``tenacity``
import anywhere in the chain), so the "core (no extras)" CI job exercises the
fsspec-only download path — exactly the path a user on a base install takes when
re-running against a pre-downloaded local mirror. Generic ``meds-extract-download``
CLI behavior tests live in ``test_download_cli.py``; HTTP-backed tests
(MockTransport wire behavior, retry, Range-resume) live in ``test_download.py``,
which the no-extras environment skips.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.skipif(sys.platform == "win32", reason="symlink semantics differ on Windows")
def test_fetch_refuses_symlink_escape(tmp_path: Path):
    """``_resolve_dest`` is the runtime security boundary: a rel_path that is clean at the string level
    (``evil/x.txt``) but whose first component is a symlink pointing outside ``dest_dir`` must be rejected at
    fetch time, with nothing written outside."""
    from MEDS_extract.download import RemoteFile, Source

    outside = tmp_path / "outside"
    outside.mkdir()
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "evil").symlink_to(outside)

    class EvilSource(Source):
        def _list_files(self):
            return [RemoteFile("evil/x.txt", "")]

        def _pull(self, source_path, target):
            target.write_text("escaped!")

    with pytest.raises(ValueError, match="escapes dest_dir"):
        EvilSource().download_all(dest)
    assert list(outside.iterdir()) == [], "nothing may be written outside dest_dir"


def test_fsspec_include_filters_before_hashing(tmp_path: Path, monkeypatch):
    """``include=`` on FsspecSource must filter *before* hashing — the documented cost mitigation for cloud
    mirrors is that excluded bytes are never read.

    (Source.files would re-filter anyway, so only a hash recorder can catch this regressing.)
    """
    import MEDS_extract.download.backends.fsspec as fs_mod

    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "keep.csv").write_text("kept\n")
    (mirror / "drop.csv").write_text("dropped\n")

    hashed: list[str] = []
    real_sha256_of = fs_mod.sha256_of

    def recording_sha256_of(p):
        hashed.append(p.name)
        return real_sha256_of(p)

    monkeypatch.setattr(fs_mod, "sha256_of", recording_sha256_of)

    dst = tmp_path / "dst"
    fs_mod.FsspecSource(root=str(mirror), include=["keep*"]).download_all(dst)

    assert hashed == ["keep.csv"], "excluded files must never be read/hashed"
    assert (dst / "keep.csv").exists()
    assert not (dst / "drop.csv").exists()


def test_pooled_dispatch_completes_manifests_larger_than_window(tmp_path: Path, monkeypatch):
    """The bounded sliding-window submission in ``Source._attempts`` must drain manifests
    larger than the window: every item beyond the initial batch is submitted as prior
    futures complete, and all of them finish."""
    from concurrent.futures import ThreadPoolExecutor

    import MEDS_extract.download.source as source_mod
    from MEDS_extract.download import RemoteFile, Source

    monkeypatch.setattr(source_mod, "_MAX_PENDING_SUBMITS", 4)
    n = 25  # > 6x the patched window

    class ManySource(Source):
        def _list_files(self):
            return [RemoteFile(f"f{i:03d}.txt", "") for i in range(n)]

        def _pull(self, source_path, target):
            target.write_text("ok")

    with ThreadPoolExecutor(max_workers=3) as pool:
        ManySource().download_all(tmp_path, pool=pool)

    assert len(list(tmp_path.glob("f*.txt"))) == n


def test_fsspec_source_memory_protocol(tmp_path: Path):
    """FsspecSource against a non-local protocol (in-process ``memory://``): locks the ``source_path=str(p)``
    → ``UPath(source_path)`` round-trip that every remote root depends on, without needing a network."""
    import uuid

    from upath import UPath

    from MEDS_extract.download import FsspecSource

    # The memory filesystem is process-global — use a unique root and clean it up
    # so nothing leaks into other tests (or later parametrizations) this session.
    root = UPath(f"memory://mirror-{uuid.uuid4().hex}")
    try:
        (root / "sub").mkdir(parents=True, exist_ok=True)
        (root / "patients.csv").write_bytes(b"patient_id\n1\n")
        (root / "sub" / "vitals.csv").write_bytes(b"pid,hr\n1,80\n")

        FsspecSource(root=str(root)).download_all(tmp_path)
        assert (tmp_path / "patients.csv").read_bytes() == b"patient_id\n1\n"
        assert (tmp_path / "sub" / "vitals.csv").read_bytes() == b"pid,hr\n1,80\n"
    finally:
        root.fs.rm(root.path, recursive=True)
