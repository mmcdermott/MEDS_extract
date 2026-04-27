"""Manifest types shared between :mod:`source` and :mod:`fetcher`.

Pulled into a third module so the two top-level modules can import each other's
public classes (:class:`Source` and :class:`Fetcher`) without a circular import.
The only types that live here are the ones referenced by both modules:

- :class:`RemoteFile` — manifest row, the return type of ``Source._list_files``.
- :class:`ChecksumError` — raised by every transport on SHA-256 mismatch.
- :func:`sha256_of` — streaming SHA-256 helper used by transports + the fetcher's
  ``_already_complete`` cache check.

Nothing in here is part of the user-facing public API. The download package's
top-level :mod:`__init__` exports :class:`Source` and :class:`Fetcher` (and the
concrete backends), not anything from here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class ChecksumError(ValueError):
    """Raised when a downloaded file's SHA-256 doesn't match the expected digest.

    Every :class:`Source` that honors ``remote.sha256`` raises this on mismatch
    (not just the HTTP-backed ones) so callers can catch a single exception type
    regardless of transport.
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
    """One manifest row from a :class:`~MEDS_extract.download.source.Source`.

    Internal-ish: the only legitimate construction sites are inside a backend's
    ``_list_files`` and inside test stub sources. Users never see one passed in or
    out of the public API; ``Source.download_all`` is the only fetch entry point.

    Attributes:
        rel_path: Where the file lands under ``download_all``'s ``dest_dir``. Must
            use forward slashes; path semantics mirror ``pathlib.PurePosixPath``.
        size: Expected content-length in bytes, if the source's manifest provides
            it. Used by :class:`~MEDS_extract.download.fetcher.Fetcher` as a cheap
            "is this file already fully downloaded" check before falling back to
            the (more expensive) SHA-256 verify.
        sha256: Expected SHA-256 digest (lowercase hex), if the source's manifest
            provides it. Checked by every transport after write; a mismatch is
            always a hard error.
        unarchive: Optional post-fetch unpack format. ``None`` means no unpack
            (default). ``"zip"``, ``"tar"``, ``"tar.gz"`` / ``"tgz"`` dispatch to
            the matching :class:`~MEDS_extract.download.unarchive.ArchiveFormat`.
            ``"auto"`` infers the format from ``rel_path``'s extension — useful
            when a single source lists both archive and non-archive files, since
            ``"auto"`` is a no-op on anything that doesn't end in a recognized
            archive extension.
        cleanup_archive: Tri-state controlling whether the source archive file
            is removed after a successful extraction. ``None`` (default) means
            "use the mode-implied default": ``"auto"`` removes the archive (the
            one-arg "fetch + extract + cleanup" flow); explicit formats keep it.
            Set ``True`` to force cleanup, ``False`` to force keep, regardless of
            mode. Has no effect when ``unarchive`` is ``None``.
        extra: Transport-specific stash. HTTP-backed sources put the absolute URL
            here; fsspec-backed sources put the :class:`~upath.UPath`. Read only
            by the originating source's ``_fetch``.

    Examples:
        >>> r = RemoteFile(rel_path="patients.csv.gz", size=1234, sha256="abc123def456ffff")
        >>> r.rel_path
        'patients.csv.gz'
        >>> r.size, r.sha256
        (1234, 'abc123def456ffff')
        >>> r.unarchive is None and r.cleanup_archive is None
        True
        >>> r.extra
        {}

        ``__str__`` is compact and doctest-friendly — rel_path plus any set
        manifest fields:

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
