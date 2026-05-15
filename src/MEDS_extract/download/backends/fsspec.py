"""Fsspec-backed :class:`Source` for local and cloud-bucket re-runs."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from upath import UPath

from ..source import RemoteFile, Source, sha256_of

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

    Each :class:`RemoteFile` carries a SHA-256 computed from the source file at
    ``_list_files`` time, so re-runs verify the on-disk copy against the same hash and
    skip if it matches. For local roots the hash cost is trivial; for cloud-bucket roots
    it costs one source-side read per file the first time the manifest is built (cached
    thereafter via :attr:`Source.files`).

    The transport implementation is ``_pull`` only — the base class handles
    ``.part`` staging, SHA verification, and atomic rename.

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
    """

    def __init__(self, root: str):
        self._root = UPath(root)

    def _list_files(self) -> Iterable[RemoteFile]:
        for p in self._root.rglob("*"):
            if not p.is_file():
                continue
            yield RemoteFile(
                rel_path=p.relative_to(self._root).as_posix(),
                sha256=sha256_of(p),
                source_path=str(p),
            )

    def _pull(self, remote: RemoteFile, target: Path) -> None:
        with UPath(remote.source_path).open("rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
