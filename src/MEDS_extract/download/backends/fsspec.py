"""Fsspec-backed :class:`Source` for local and cloud-bucket re-runs."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from upath import UPath

from ..source import RemoteFile

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class FsspecSource:
    """A :class:`Source` backed by an fsspec-compatible root via :class:`upath.UPath`.

    Accepts any protocol UPath supports: ``file://`` and local paths (re-run against a
    pre-downloaded copy), ``s3://`` / ``gs://`` / ``azure://`` (mirrors the user keeps on
    cloud storage), and so on. The ``fsspec`` extras for cloud protocols (``s3fs``,
    ``gcsfs``, …) are NOT declared as dependencies here — users install them themselves
    following the standard fsspec pattern.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as src_dir:
        ...     src_dir = Path(src_dir)
        ...     _ = (src_dir / "patients.csv").write_text("patient_id,dob\\n1,2000-01-01")
        ...     (src_dir / "labs").mkdir()
        ...     _ = (src_dir / "labs" / "vitals.csv").write_text("pid,hr\\n1,80")
        ...     source = FsspecSource(root=str(src_dir))
        ...     rel_paths = sorted(r.rel_path for r in source.list_files())
        >>> rel_paths
        ['labs/vitals.csv', 'patients.csv']

        Fetching copies to ``dest`` atomically via a ``.part`` file:

        >>> with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        ...     src, dst = Path(src), Path(dst)
        ...     _ = (src / "x.txt").write_text("hello")
        ...     source = FsspecSource(root=str(src))
        ...     [remote] = list(source.list_files())
        ...     dest = dst / remote.rel_path
        ...     source.fetch(remote, dest)
        ...     dest.read_text()
        'hello'
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

    def fetch(self, remote: RemoteFile, dest: Path) -> None:
        upath = remote.extra["upath"]
        part = dest.with_name(dest.name + ".part")
        try:
            with upath.open("rb") as src, part.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            part.replace(dest)
        finally:
            if part.exists():
                part.unlink()
