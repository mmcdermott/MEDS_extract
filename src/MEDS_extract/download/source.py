"""The :class:`Source` protocol and its companion dataclasses.

A :class:`Source` is anywhere raw data comes from — a PhysioNet dataset release, an explicit
list of HTTP URLs, an S3 / GCS / local-filesystem tree. It enumerates files and fetches them.
Everything else in :mod:`MEDS_extract.download` is built on this one abstraction.

See :mod:`MEDS_extract.download.fetcher` for the orchestrator that drives a :class:`Source`
through a bounded-concurrency download plan, and
:mod:`MEDS_extract.download.backends` for the concrete implementations.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

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
        >>> import tempfile
        >>> from pathlib import Path
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
        >>> r.extra
        {}

        ``__str__`` is compact and doctest-friendly — rel_path plus any set manifest fields:

        >>> print(RemoteFile("x.csv"))
        x.csv
        >>> print(RemoteFile("x.csv", size=1234))
        x.csv size=1234
        >>> print(r)
        patients.csv.gz size=1234 sha256=abc123def456...

        ``RemoteFile`` is frozen — mutating it raises:

        >>> r.rel_path = "other"
        Traceback (most recent call last):
            ...
        dataclasses.FrozenInstanceError: cannot assign to field 'rel_path'
    """

    rel_path: str
    size: int | None = None
    sha256: str | None = None
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [self.rel_path]
        if self.size is not None:
            parts.append(f"size={self.size}")
        if self.sha256 is not None:
            parts.append(f"sha256={self.sha256[:12]}...")
        return " ".join(parts)


@dataclass(frozen=True)
class FetchResult:
    """The outcome of fetching one :class:`RemoteFile`.

    ``status`` is one of ``"downloaded"`` (written this run), ``"skipped"`` (already
    present and verified), or ``"failed"`` (transport raised; ``error`` has the exception).

    Examples:
        >>> from pathlib import Path
        >>> r = RemoteFile(rel_path="x.csv")
        >>> ok = FetchResult(remote=r, dest=Path("/tmp/x.csv"), status="downloaded")
        >>> ok.status
        'downloaded'
        >>> ok.error is None
        True
    """

    remote: RemoteFile
    dest: Path
    status: str
    error: Exception | None = None


@dataclass(frozen=True)
class FetchReport:
    """Summary of one :meth:`~MEDS_extract.download.fetcher.Fetcher.fetch_all` call.

    Examples:
        >>> from pathlib import Path
        >>> report = FetchReport(
        ...     results=[
        ...         FetchResult(RemoteFile("a.csv"), Path("/tmp/a.csv"), "downloaded"),
        ...         FetchResult(RemoteFile("b.csv"), Path("/tmp/b.csv"), "skipped"),
        ...     ]
        ... )
        >>> report.n_downloaded
        1
        >>> report.n_skipped
        1
        >>> report.n_failed
        0
        >>> report.ok
        True
    """

    results: list[FetchResult]

    @property
    def n_downloaded(self) -> int:
        return sum(1 for r in self.results if r.status == "downloaded")

    @property
    def n_skipped(self) -> int:
        return sum(1 for r in self.results if r.status == "skipped")

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if r.status == "failed")

    @property
    def ok(self) -> bool:
        return self.n_failed == 0


@runtime_checkable
class Source(Protocol):
    """A place raw data comes from.

    Implementations declare *what* files they offer (:meth:`list_files`) and *how* to fetch
    one (:meth:`fetch`). The :class:`~MEDS_extract.download.fetcher.Fetcher` handles
    everything else — directory creation, concurrency, per-file skip logic, reporting.

    Invariants implementations must uphold:

    - :meth:`list_files` is idempotent across calls. Re-enumerating must produce the same
      set of :class:`RemoteFile` objects (in the same order when possible).
    - :meth:`fetch` writes to ``dest.with_name(dest.name + ".part")`` then atomic-renames
      into place, so partial writes never leave a corrupt ``dest`` in place.
    - :meth:`fetch` honors ``remote.sha256`` when set: verify after write, raise on
      mismatch, delete the ``.part``.
    - :meth:`fetch` raises on transport errors rather than writing a partial file.
    - :meth:`fetch` with ``do_overwrite=True`` clears any pre-existing ``dest`` and
      ``.part`` before fetching; with ``do_overwrite=False`` (default), existing state
      may short-circuit the fetch when the destination is already valid.
    """

    def list_files(self) -> Iterable[RemoteFile]:  # pragma: no cover — Protocol
        """Enumerate the files this source offers.

        Implementations MAY stream via a generator for large manifests. Callers should not
        assume the result is re-iterable — materialize into a ``list`` when needed.
        """
        ...

    def fetch(
        self, remote: RemoteFile, dest: Path, do_overwrite: bool = False
    ) -> None:  # pragma: no cover — Protocol
        """Fetch ``remote`` to ``dest``.

        Args:
            remote: The file to fetch.
            dest: Final destination path.
            do_overwrite: Force a fresh download, ignoring any cached ``dest`` / ``.part``
                state. The :class:`~MEDS_extract.download.fetcher.Fetcher` threads its own
                ``do_overwrite`` attribute through; direct callers can override per-file.

        See invariants in the class docstring.
        """
        ...
