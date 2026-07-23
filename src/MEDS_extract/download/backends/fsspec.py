"""Fsspec-backed :class:`Source` for local and cloud-bucket re-runs."""

from __future__ import annotations

import logging
import posixpath
import shutil
from typing import TYPE_CHECKING

from upath import UPath

from ..source import RemoteFile, Source, sha256_of

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

logger = logging.getLogger(__name__)


class FsspecSource(Source):
    """A :class:`Source` backed by an fsspec-compatible root via :class:`upath.UPath`.

    Accepts any protocol UPath supports: ``file://`` and local paths (re-run against a
    pre-downloaded copy), ``s3://`` / ``gs://`` / ``azure://`` (mirrors the user keeps on
    cloud storage), and so on. The ``fsspec`` extras for cloud protocols (``s3fs``,
    ``gcsfs``, …) are NOT declared as dependencies here — users install them themselves
    following the standard fsspec pattern.

    Each :class:`RemoteFile` carries a SHA-256 computed from the source file at
    ``_list_files`` time, so re-runs verify the on-disk copy against the same hash and
    skip if it matches. **Cost note**: computing those hashes reads every selected
    source file in full, serially, in the manifest-building thread — *before* any file
    is fetched and *on every run* (the manifest is cached per ``Source`` instance, not
    across processes). For local roots that read is cheap; for cloud-bucket roots it
    means a complete remote read of the (filtered) dataset per invocation, so use
    ``include=`` / ``exclude=`` to subset large mirrors, or prefer a local mirror for
    iterated re-runs.

    Args:
        root: The tree to copy from — a local path or any UPath-supported URL.
            Must exist; a nonexistent root raises :class:`FileNotFoundError` at
            manifest time rather than silently yielding an empty file list.
        include, exclude: Optional :mod:`fnmatch` globs applied to the manifest —
            see :class:`~MEDS_extract.download.source.Source`.

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

        A nonexistent root is a config error, not an empty dataset:

        >>> FsspecSource(root="/no/such/dir/anywhere").files
        Traceback (most recent call last):
            ...
        FileNotFoundError: FsspecSource root does not exist: /no/such/dir/anywhere
    """

    def __init__(self, root: str, include: list[str] | None = None, exclude: list[str] | None = None):
        super().__init__(include=include, exclude=exclude)
        self._root = UPath(root)

    def _list_files(self) -> Iterable[RemoteFile]:
        if not self._root.exists():
            raise FileNotFoundError(f"{type(self).__name__} root does not exist: {self._root}")
        logger.info(
            f"Building manifest for {self._root}: hashing each selected source file "
            "(this reads the selected files in full)"
        )
        n_total = 0
        n_skipped = 0
        for p in self._root.rglob("*"):
            if not p.is_file():
                continue
            rel_path = p.relative_to(self._root).as_posix()
            n_total += 1
            # Apply the manifest filters *before* hashing — the whole point of
            # ``include=`` on a cloud mirror is to not read the excluded bytes.
            if not self._selected_path(posixpath.normpath(rel_path)):
                n_skipped += 1
                continue
            logger.debug(f"Hashing {rel_path}")
            yield RemoteFile(
                rel_path=rel_path,
                sha256=sha256_of(p),
                source_path=str(p),
            )
        if n_skipped:
            logger.info(
                f"include/exclude filters skipped {n_skipped}/{n_total} files under "
                f"{self._root} before hashing"
            )

    def _pull(self, source_path: str, target: Path) -> None:
        with UPath(source_path).open("rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
