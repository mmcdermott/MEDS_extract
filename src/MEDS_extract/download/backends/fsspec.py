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
        ``list_files()`` yields one :class:`RemoteFile` per file in the tree, with ``size``
        set from :meth:`~pathlib.Path.stat`:

        >>> spec = '''
        ... patients.csv: |
        ...   patient_id,dob
        ...   1,2000-01-01
        ... labs:
        ...   vitals.csv: |
        ...     pid,hr
        ...     1,80
        ... '''
        >>> with yaml_disk(spec) as src_dir:
        ...     source = FsspecSource(root=str(src_dir))
        ...     for r in sorted(source.list_files(), key=lambda r: r.rel_path):
        ...         print(r)
        labs/vitals.csv size=11
        patients.csv size=27

        Fetching copies to ``dest`` atomically via a ``.part`` file:

        >>> with yaml_disk("x.txt: hello") as src, tempfile.TemporaryDirectory() as dst:
        ...     dst = Path(dst)
        ...     source = FsspecSource(root=str(src))
        ...     [remote] = list(source.list_files())
        ...     source.fetch(remote, dst / remote.rel_path)
        ...     (dst / "x.txt").read_text()
        ...     print_directory(dst)
        'hello'
        └── x.txt
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
