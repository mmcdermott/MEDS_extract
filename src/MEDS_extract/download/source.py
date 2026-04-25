"""The :class:`Source` ABC and its companion :class:`RemoteFile` / :class:`ChecksumError`.

A :class:`Source` is anywhere raw data comes from — a PhysioNet dataset release, an explicit
list of HTTP URLs, an S3 / GCS / local-filesystem tree. Concrete sources inherit from this
ABC and implement two methods:

- :meth:`Source.list_files` — enumerate what files the source offers.
- :meth:`Source._fetch` — move one file's bytes from the source to a local path.

The public :meth:`Source.fetch` method is defined here on the base class: it enforces the
"``dest`` must not already exist unless ``do_overwrite=True``" precondition (raising
:class:`FileExistsError` on violation) and clears stale ``dest`` / ``.part`` state when
``do_overwrite=True``. That way every concrete backend has identical overwrite semantics
without reimplementing the guard.

See :mod:`MEDS_extract.download.fetcher` for the orchestrator that drives a :class:`Source`
through a bounded-concurrency download plan, and :mod:`MEDS_extract.download.backends` for
the concrete implementations.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class ChecksumError(ValueError):
    """Raised when a downloaded file's SHA-256 doesn't match the expected digest.

    Every :class:`Source` that honors ``remote.sha256`` raises this on mismatch (not just the
    HTTP-backed ones) so callers can catch a single exception type regardless of transport.
    """

    def __init__(self, source_id: str, expected: str, actual: str):
        self.source_id = source_id
        self.expected = expected
        self.actual = actual
        super().__init__(f"SHA-256 mismatch for {source_id}: expected {expected}, got {actual}")


def sha256_of(fp: Path) -> str:
    """Compute the SHA-256 of ``fp``, streaming 1 MiB at a time.

    Examples:
        >>> with tempfile.NamedTemporaryFile(delete=False) as tmp:
        ...     _ = tmp.write(b"hello world")
        ...     fp = Path(tmp.name)
        >>> sha256_of(fp)
        'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
        >>> fp.unlink()
    """
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class RemoteFile:
    """One file addressable within a :class:`Source`.

    Attributes:
        rel_path: Where the file lands under the fetcher's ``dest_dir``. Must use forward
            slashes; path semantics mirror ``pathlib.PurePosixPath``.
        size: Expected content-length in bytes, if the source's manifest provides it. Used
            by :class:`~MEDS_extract.download.fetcher.Fetcher` as a cheap "is this file
            already fully downloaded" check before falling back to the (more expensive)
            SHA-256 verify.
        sha256: Expected SHA-256 digest (lowercase hex), if the source's manifest provides it.
            Checked by every transport after write; a mismatch is always a hard error.
        unarchive: Optional post-fetch unpack format. ``None`` means no unpack (default).
            ``"zip"``, ``"tar"``, ``"tar.gz"`` / ``"tgz"`` dispatch to the matching
            :class:`~MEDS_extract.download.unarchive.ArchiveFormat`. ``"auto"`` infers
            the format from ``rel_path``'s extension — useful when a single source lists
            both archive and non-archive files, since ``"auto"`` is a no-op on anything
            that doesn't end in a recognized archive extension.
        cleanup_archive: Tri-state controlling whether the source archive file is
            removed after a successful extraction. ``None`` (default) means "use the
            mode-implied default": when ``unarchive == "auto"`` the archive is removed
            (the one-arg "fetch + extract + cleanup" flow); when ``unarchive`` names
            an explicit format the archive is kept (matches the prior default —
            re-runs stay cheap and the SHA-256 verify remains reproducible). Set
            ``True`` to force cleanup, ``False`` to force keep, regardless of mode.
            Has no effect when ``unarchive`` is ``None``.
        extra: Transport-specific fields. HTTP-backed sources stash the absolute URL here;
            fsspec-backed sources stash the :class:`~upath.UPath`. Users shouldn't touch this.

    Examples:
        >>> r = RemoteFile(rel_path="patients.csv.gz", size=1234, sha256="abc123def456ffff")
        >>> r.rel_path
        'patients.csv.gz'
        >>> r.size
        1234
        >>> r.sha256
        'abc123def456ffff'
        >>> r.unarchive is None
        True
        >>> r.cleanup_archive is None
        True
        >>> r.extra
        {}

        ``__str__`` is compact and doctest-friendly — rel_path plus any set manifest fields:

        >>> print(RemoteFile("x.csv"))
        x.csv
        >>> print(RemoteFile("x.csv", size=1234))
        x.csv size=1234
        >>> print(r)
        patients.csv.gz size=1234 sha256=abc123def456...
        >>> print(RemoteFile("AUMCdb.zip", unarchive="auto"))
        AUMCdb.zip unarchive=auto
        >>> print(RemoteFile("AUMCdb.zip", unarchive="zip", cleanup_archive=True))
        AUMCdb.zip unarchive=zip cleanup_archive=True

        ``RemoteFile`` is frozen — mutating it raises:

        >>> r.rel_path = "other"
        Traceback (most recent call last):
            ...
        dataclasses.FrozenInstanceError: cannot assign to field 'rel_path'
    """

    rel_path: str
    size: int | None = None
    sha256: str | None = None
    unarchive: str | None = None
    cleanup_archive: bool | None = None
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [self.rel_path]
        if self.size is not None:
            parts.append(f"size={self.size}")
        if self.sha256 is not None:
            parts.append(f"sha256={self.sha256[:12]}...")
        if self.unarchive is not None:
            parts.append(f"unarchive={self.unarchive}")
        if self.cleanup_archive is not None:
            parts.append(f"cleanup_archive={self.cleanup_archive}")
        return " ".join(parts)


class Source(ABC):
    """A place raw data comes from.

    Subclasses implement :meth:`list_files` (what files this source offers) and
    :meth:`_fetch` (how to move one file's bytes to a local path). :meth:`fetch` is
    concrete on the base class and wraps ``_fetch`` with the overwrite-precondition check
    so every backend has identical ``do_overwrite`` semantics.

    Invariants implementations must uphold:

    - :meth:`list_files` is idempotent across calls. Re-enumerating must produce the same
      set of :class:`RemoteFile` objects (in the same order when possible).
    - :meth:`_fetch` writes to ``dest.with_name(dest.name + ".part")`` then atomic-renames
      into place, so a partial write never leaves a corrupt or truncated final ``dest``
      on disk. The staged ``.part`` file MAY remain after a transport failure — that is
      intentional, since it enables range-resume on a subsequent attempt. Callers who
      want a clean slate pass ``do_overwrite=True`` to :meth:`fetch`.
    - :meth:`_fetch` honors ``remote.sha256`` when set: verify after write, raise on
      mismatch, delete the ``.part``.
    - :meth:`_fetch` raises on transport errors rather than completing the rename into
      ``dest``.
    - :meth:`_fetch` is called by the base ``fetch`` only after the dest-exists guard has
      cleared the path, so implementations can assume ``dest`` does not exist on entry.
    """

    @abstractmethod
    def list_files(self) -> Iterable[RemoteFile]:
        """Enumerate the files this source offers.

        Implementations MAY stream via a generator for large manifests. Callers should not
        assume the result is re-iterable — materialize into a ``list`` when needed.
        """

    def fetch(self, remote: RemoteFile, dest: Path, do_overwrite: bool = False) -> None:
        """Fetch ``remote`` to ``dest``, enforcing the overwrite precondition.

        When ``remote.unarchive`` is set, the archive is unpacked into ``dest.parent``
        after ``_fetch`` (and after the SHA-256 verify the transport performs). If
        ``remote.cleanup_archive`` is also set, the archive file is removed once
        extraction succeeds. Extraction uses
        :func:`~MEDS_extract.download.unarchive.safe_extract`, which rejects zip-slip /
        tar-slip attempts before touching the filesystem.

        Args:
            remote: The file to fetch.
            dest: Final destination path. Parent directories must already exist.
            do_overwrite: If ``True``, any pre-existing ``dest`` and ``.part`` are deleted
                before fetching — forces a fresh copy. If ``False`` (default), an existing
                ``dest`` is a precondition violation and raises :class:`FileExistsError`.
                ``do_overwrite=True`` does NOT clean previously-extracted archive members
                out of ``dest.parent``; callers that need a pristine target directory must
                clear it manually.

        Raises:
            FileExistsError: If ``dest`` exists and ``do_overwrite`` is ``False``.
            ValueError: If ``remote.unarchive`` names an unsupported format or the
                archive contains an unsafe member (zip-slip / tar-slip).

        Examples:
            ``remote.unarchive`` triggers a post-fetch unpack. ``cleanup_archive=True``
            removes the archive once extraction completes — the AUMCdb / HIRID pattern,
            where the archive itself is just a transit format:

            >>> import io, zipfile as _zipfile
            >>> class ZipSource(Source):
            ...     '''Stub source that "fetches" by synthesizing a zip on disk.'''
            ...     def __init__(self, payload):
            ...         self._payload = payload  # list of (name, bytes)
            ...     def list_files(self):
            ...         return [RemoteFile("bundle.zip", unarchive="auto", cleanup_archive=True)]
            ...     def _fetch(self, remote, dest):
            ...         buf = io.BytesIO()
            ...         with _zipfile.ZipFile(buf, "w") as zf:
            ...             for name, data in self._payload:
            ...                 zf.writestr(name, data)
            ...         dest.write_bytes(buf.getvalue())
            >>>
            >>> src = ZipSource([("a.csv", b"col\\n1"), ("sub/b.csv", b"col\\n2")])
            >>> [remote] = list(src.list_files())
            >>> with tempfile.TemporaryDirectory() as d:
            ...     d = Path(d)
            ...     src.fetch(remote, d / "bundle.zip")
            ...     files = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
            ...     bundle_removed = not (d / "bundle.zip").exists()
            >>> files  # archive removed via cleanup_archive=True; members materialized
            ['a.csv', 'sub/b.csv']
            >>> bundle_removed
            True
        """
        part = dest.with_name(dest.name + ".part")
        if do_overwrite:
            if dest.exists():
                dest.unlink()
            if part.exists():
                part.unlink()
        elif dest.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file {dest}. "
                f"Pass do_overwrite=True to force a refetch, or delete the file first."
            )
        self._fetch(remote, dest)
        if remote.unarchive:
            # Import locally so the ABC stays importable without the optional extras —
            # ``unarchive.py`` uses only stdlib, but keeping the import lazy matches the
            # pattern the rest of this module uses for transport-specific deps.
            from .unarchive import ArchiveFormat, resolve_format, safe_extract

            fmt = resolve_format(remote.unarchive, dest)
            if fmt is not None:
                safe_extract(dest, dest.parent, fmt)
                # Tri-state cleanup: ``None`` defers to the mode-implied default —
                # AUTO opts into the full "fetch + extract + cleanup" flow (the
                # one-arg way to drop the archive), explicit formats keep it.
                # Explicit True/False from the user always wins.
                if remote.cleanup_archive is None:
                    cleanup = ArchiveFormat(remote.unarchive) is ArchiveFormat.AUTO
                else:
                    cleanup = remote.cleanup_archive
                if cleanup:
                    dest.unlink()

    @abstractmethod
    def _fetch(self, remote: RemoteFile, dest: Path) -> None:
        """Transport-specific fetch implementation.

        See :class:`Source` invariants.
        """

    def close(self) -> None:  # noqa: B027 — intentional no-op default; subclasses override when needed
        """Release transport resources held by this source.

        Default is a no-op. Subclasses that own network clients / file handles / connection pools override
        this to close them. Safe to call multiple times; safe to call on sources that own nothing.
        """

    def __enter__(self) -> Source:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
