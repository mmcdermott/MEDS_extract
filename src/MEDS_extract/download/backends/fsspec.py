"""Fsspec-backed :class:`Source` for local and cloud-bucket re-runs."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from upath import UPath

from ..source import ChecksumError, RemoteFile, Source, sha256_of

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class FsspecSource(Source):
    """A :class:`Source` backed by an fsspec-compatible root via :class:`upath.UPath`.

    Accepts any protocol UPath supports: ``file://`` and local paths (re-run against a
    pre-downloaded copy), ``s3://`` / ``gs://`` / ``azure://`` (mirrors the user keeps on
    cloud storage), and so on. The ``fsspec`` extras for cloud protocols (``s3fs``,
    ``gcsfs``, …) are NOT declared as dependencies here — users install them themselves
    following the standard fsspec pattern.

    Examples:
        ``download_all`` walks the tree and copies every file under ``dest_dir``,
        preserving the relative layout:

        >>> spec = '''
        ... patients.csv: |
        ...   patient_id,dob
        ...   1,2000-01-01
        ... labs:
        ...   vitals.csv: |
        ...     pid,hr
        ...     1,80
        ... '''
        >>> with yaml_disk(spec) as src_dir, tempfile.TemporaryDirectory() as dst:
        ...     dst = Path(dst)
        ...     FsspecSource(root=str(src_dir)).download_all(dst)
        ...     print_directory(dst)
        ├── labs
        │   └── vitals.csv
        └── patients.csv

        :meth:`_fetch` honors ``remote.sha256``: a mismatch raises
        :class:`~MEDS_extract.download.source.ChecksumError`, the staged ``.part``
        is cleaned up, and the dest is not created. Important for local-mirror
        re-runs where a silent drift between the mirror and the authoritative
        manifest should fail loudly rather than feed corrupt bytes downstream:

        >>> import hashlib
        >>> with yaml_disk("x.txt: hello fsspec") as src, tempfile.TemporaryDirectory() as out:
        ...     out = Path(out)
        ...     source = FsspecSource(root=str(src))
        ...     remote = RemoteFile("x.txt", sha256="0" * 64, extra={"upath": src / "x.txt"})
        ...     try:
        ...         source._fetch(remote, out / "x.txt")
        ...     except ChecksumError as e:
        ...         print(f"raised: {type(e).__name__}")
        ...     print(f"dest exists: {(out / 'x.txt').exists()}")
        ...     print(f"part exists: {(out / 'x.txt.part').exists()}")
        raised: ChecksumError
        dest exists: False
        part exists: False
    """

    def __init__(self, root: str):
        self._root = UPath(root)

    def list_files(self) -> Iterable[RemoteFile]:
        for p in self._root.rglob("*"):
            if not p.is_file():
                continue
            yield RemoteFile(
                rel_path=p.relative_to(self._root).as_posix(),
                size=p.stat().st_size,
                extra={"upath": p},
            )

    def _fetch(self, remote: RemoteFile, dest: Path) -> None:
        upath = remote.extra["upath"]
        part = dest.with_name(dest.name + ".part")
        try:
            with upath.open("rb") as src, part.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            # Honor remote.sha256 the same way HTTP-backed Sources do — a mismatch is a
            # hard error regardless of transport. Important for local-mirror re-runs:
            # if the mirror is silently out of sync with the authoritative manifest, we
            # want to fail loudly rather than feed corrupt data downstream.
            if remote.sha256 is not None:
                actual = sha256_of(part)
                if actual != remote.sha256:
                    part.unlink()
                    raise ChecksumError(str(upath), remote.sha256, actual)
            part.replace(dest)
        finally:
            if part.exists():
                part.unlink()
