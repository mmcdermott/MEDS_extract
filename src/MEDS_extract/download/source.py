"""The :class:`Source` ABC, its manifest type :class:`RemoteFile`, and the orchestration loop.

A :class:`Source` is anywhere raw data comes from — a PhysioNet dataset release, an
explicit list of HTTP URLs, an S3 / GCS / local-filesystem tree. Concrete sources
inherit from this ABC and implement two private hooks:

- :meth:`Source._list_files` — enumerate what files the source offers.
- :meth:`Source._fetch` — move one file's bytes from the source to a local path.

The single public fetch entry point is :meth:`Source.download_all`. It optionally
takes a :class:`ThreadPoolExecutor` (so multiple sources in one CLI invocation can
share a pool) and a :class:`DownloadPolicy` (skip-check / continue-on-error /
overwrite knobs). When neither is given, ``download_all`` builds a private pool +
default policy, runs the bundle, and tears down — the simple-case caller writes
``src.download_all(dest)`` and gets sensible behavior.

Why this shape — single-file, free-function orchestration, no separate ``Fetcher``
class:

- Per-file fetch is not a public concern. Every consumer downloads the bundle.
- Concurrency is a property of a thread pool, not a property of a "fetcher."
- Skip / overwrite / continue-on-error are configuration; that's a frozen dataclass
  (:class:`DownloadPolicy`), not a class with methods.
- The orchestration logic — submit each manifest item to the pool, run the
  per-file skip-and-fetch, collect the report — is a free function that takes a
  source, a pool, and a policy. It doesn't need a class.

See https://github.com/mmcdermott/MEDS_extract/pull/96 for the design discussion.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


# ── manifest types ────────────────────────────────────────────────────────────


class ChecksumError(ValueError):
    """Raised when a downloaded file's SHA-256 doesn't match the expected digest.

    Every :class:`Source` that honors ``remote.sha256`` raises this on mismatch (not
    just the HTTP-backed ones) so callers can catch a single exception type
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
    """One manifest row from a :class:`Source`.

    Internal-ish: the only legitimate construction sites are inside a backend's
    ``_list_files`` and inside test stub sources. Users never see one passed in or
    out of the public API; :meth:`Source.download_all` is the only fetch entry point.

    Attributes:
        rel_path: Where the file lands under ``download_all``'s ``dest_dir``. Must
            use forward slashes; path semantics mirror ``pathlib.PurePosixPath``.
        size: Expected content-length in bytes, if the source's manifest provides
            it. Used as a cheap "is this file already fully downloaded" check
            before falling back to the (more expensive) SHA-256 verify.
        sha256: Expected SHA-256 digest (lowercase hex), if the source's manifest
            provides it. Checked by every transport after write; a mismatch is
            always a hard error.
        extra: Transport-specific stash. HTTP-backed sources put the absolute URL
            here; fsspec-backed sources put the :class:`~upath.UPath`. Read only
            by the originating source's ``_fetch``.

    Examples:
        >>> r = RemoteFile(rel_path="patients.csv.gz", size=1234, sha256="abc123def456ffff")
        >>> r.rel_path
        'patients.csv.gz'
        >>> r.size, r.sha256
        (1234, 'abc123def456ffff')
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


# ── policy + report types ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class DownloadPolicy:
    """Frozen config bag for :meth:`Source.download_all`.

    Concurrency is **not** here — that's a property of the
    :class:`~concurrent.futures.ThreadPoolExecutor` you pass to ``download_all``,
    not a property of the policy. This dataclass holds only the per-file
    decisions: skip-when-already-complete, continue-on-error, force-overwrite.

    Attributes:
        continue_on_error: If ``True``, per-file transport exceptions are captured
            as :class:`FetchResult` with ``status="failed"`` and the run proceeds.
            If ``False`` (default), the first failure is re-raised. Either way, any
            still-queued transfers are cancelled on exit via
            ``pool.shutdown(wait=False, cancel_futures=True)`` (when
            ``download_all`` owns the pool).
        do_overwrite: If ``True``, skip the "already complete" short-circuit and
            force a re-fetch of every file. Existing ``dest`` / ``.part`` state is
            cleared before each fetch. If ``False`` (default), existing files that
            match the manifest's ``size`` / ``sha256`` are skipped.

    Examples:
        >>> DownloadPolicy()
        DownloadPolicy(continue_on_error=False, do_overwrite=False)
        >>> DownloadPolicy(continue_on_error=True).continue_on_error
        True
    """

    continue_on_error: bool = False
    do_overwrite: bool = False


@dataclass(frozen=True)
class FetchResult:
    """The outcome of fetching one :class:`RemoteFile`.

    ``status`` is one of ``"downloaded"`` (written this run), ``"skipped"`` (already
    present and verified), or ``"failed"`` (transport raised; ``error`` has the exception).

    Examples:
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
    """Summary of one :meth:`Source.download_all` call.

    Examples:
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


# ── orchestration helpers (free functions) ────────────────────────────────────


def _resolve_dest(dest_dir: Path, rel_path: str) -> Path:
    """Resolve ``rel_path`` under ``dest_dir``, rejecting any escape attempts.

    A malformed manifest could ship an absolute path or one containing ``..``
    segments that would land the fetched file outside ``dest_dir``. Both are
    rejected eagerly before we touch the filesystem.

    Examples:
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     _resolve_dest(d, "sub/ok.txt").relative_to(d.resolve()).as_posix()
        'sub/ok.txt'

        >>> with tempfile.TemporaryDirectory() as d:
        ...     _resolve_dest(Path(d), "/etc/passwd")
        Traceback (most recent call last):
            ...
        ValueError: rel_path must be relative, got absolute: '/etc/passwd'

        >>> with tempfile.TemporaryDirectory() as d:
        ...     _resolve_dest(Path(d), "../../etc/passwd")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: rel_path '../../etc/passwd' escapes dest_dir ...
    """
    rp = Path(rel_path)
    if rp.is_absolute():
        raise ValueError(f"rel_path must be relative, got absolute: {rel_path!r}")
    dest_root = Path(dest_dir).resolve()
    resolved = (dest_root / rp).resolve()
    try:
        resolved.relative_to(dest_root)
    except ValueError as e:
        raise ValueError(
            f"rel_path {rel_path!r} escapes dest_dir {dest_root} (resolved to {resolved})."
        ) from e
    return resolved


def _already_complete(dest: Path, item: RemoteFile) -> bool:
    """True if ``dest`` already has the right content per ``item``'s manifest info.

    The cheaper size check gates the expensive SHA-256 check. When neither is
    known, the file's presence is taken as sufficient — callers can force a
    re-fetch by deleting the local copy or constructing :class:`DownloadPolicy`
    with ``do_overwrite=True``.

    Examples:
        >>> import hashlib
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     fp = d / "x.txt"
        ...     _ = fp.write_bytes(b"abc")
        ...     _already_complete(fp, RemoteFile("x.txt"))  # no manifest info
        True
        >>> with tempfile.TemporaryDirectory() as d:
        ...     fp = Path(d) / "missing.txt"
        ...     _already_complete(fp, RemoteFile("missing.txt"))
        False
        >>> with tempfile.TemporaryDirectory() as d:
        ...     fp = Path(d) / "wrong_size.txt"
        ...     _ = fp.write_bytes(b"abc")
        ...     _already_complete(fp, RemoteFile("x.txt", size=100))
        False
        >>> with tempfile.TemporaryDirectory() as d:
        ...     fp = Path(d) / "bad_hash.txt"
        ...     _ = fp.write_bytes(b"abc")
        ...     _already_complete(fp, RemoteFile("x.txt", sha256="deadbeef"))
        False
        >>> with tempfile.TemporaryDirectory() as d:
        ...     fp = Path(d) / "good.txt"
        ...     _ = fp.write_bytes(b"abc")
        ...     digest = hashlib.sha256(b"abc").hexdigest()
        ...     _already_complete(fp, RemoteFile("x.txt", size=3, sha256=digest))
        True
    """
    if not dest.exists():
        return False
    if item.size is not None and dest.stat().st_size != item.size:
        return False
    return not (item.sha256 is not None and sha256_of(dest) != item.sha256)


def _fetch_one(source: Source, item: RemoteFile, dest_dir: Path, policy: DownloadPolicy) -> FetchResult:
    """Apply skip/overwrite policy to one item and call its transport hook."""
    dest = _resolve_dest(dest_dir, item.rel_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not policy.do_overwrite and _already_complete(dest, item):
        logger.debug(f"Skipping {item.rel_path}: already complete.")
        return FetchResult(item, dest, "skipped")
    # Force overwrite if either the caller requested it OR the on-disk copy
    # failed the completeness check. Both cases are "the file is wrong and
    # needs a clean refetch"; clear stale ``dest`` / ``.part`` before
    # handing off to the transport hook.
    force = policy.do_overwrite or dest.exists()
    part = dest.with_name(dest.name + ".part")
    if force:
        if dest.exists():
            dest.unlink()
        if part.exists():
            part.unlink()
    try:
        source._fetch(item, dest)
        return FetchResult(item, dest, "downloaded")
    except Exception as e:
        if not policy.continue_on_error:
            raise
        logger.error(f"Failed to fetch {item.rel_path}: {e}")
        return FetchResult(item, dest, "failed", error=e)


def _drive(source: Source, dest_dir: Path, pool: ThreadPoolExecutor, policy: DownloadPolicy) -> FetchReport:
    """Submit every manifest item from ``source`` to ``pool`` under ``policy``.

    Pool ownership: ``pool`` is borrowed, never owned. The caller (typically
    :meth:`Source.download_all` when it builds its own pool, or the CLI when it
    builds a shared pool with ``with ThreadPoolExecutor(...) as pool:``) is
    responsible for ``pool.shutdown``.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    items = list(source._list_files())

    # Reject collisions up-front — two items with the same resolved dest would
    # race on the same ``.part`` / dest under concurrent workers and silently
    # corrupt the output. Resolve now (cheap) before submitting work.
    seen: dict[Path, RemoteFile] = {}
    for item in items:
        dest = _resolve_dest(dest_dir, item.rel_path)
        if dest in seen:
            raise ValueError(
                f"Duplicate destination {dest}: rel_path {item.rel_path!r} collides "
                f"with {seen[dest].rel_path!r}. Each item from a source's "
                "_list_files() must resolve to a unique dest_dir-relative path."
            )
        seen[dest] = item

    logger.info(f"Fetching {len(items)} files to {dest_dir}")
    futures = {pool.submit(_fetch_one, source, item, dest_dir, policy): item for item in items}
    results = [fut.result() for fut in as_completed(futures)]
    report = FetchReport(results=results)
    logger.info(
        f"Fetch complete: {report.n_downloaded} downloaded, {report.n_skipped} skipped, "
        f"{report.n_failed} failed."
    )
    return report


# ── Source ABC ────────────────────────────────────────────────────────────────


class Source(ABC):
    """A place raw data comes from.

    Subclasses implement two private hooks:

    - :meth:`_list_files` — enumerate what files the source offers, as
      :class:`RemoteFile` rows.
    - :meth:`_fetch` — move one file's bytes from the source to a local path.

    The base class supplies :meth:`download_all` as the public fetch entry point.
    Users never call the private hooks directly.

    Invariants implementations must uphold:

    - :meth:`_list_files` is idempotent across calls. Re-enumerating must produce
      the same set of :class:`RemoteFile` rows (in the same order when possible).
    - :meth:`_fetch` writes to ``dest.with_name(dest.name + ".part")`` then
      atomic-renames into place, so a partial write never leaves a corrupt or
      truncated final ``dest`` on disk. The staged ``.part`` file MAY remain after
      a transport failure — that is intentional, since it enables range-resume on
      a subsequent attempt. The base class clears stale ``.part`` / ``dest``
      before invoking ``_fetch`` when a refetch is needed.
    - :meth:`_fetch` honors ``remote.sha256`` when set: verify after write, raise
      on mismatch, delete the ``.part``.
    - :meth:`_fetch` raises on transport errors rather than completing the rename
      into ``dest``.
    - When :meth:`_fetch` is called by ``download_all``, ``dest`` does not exist —
      the skip/overwrite logic has already run.

    Examples:
        Simple case — no pool or policy passed, ``download_all`` builds and tears
        down its own private pool:

        >>> class StubSource(Source):
        ...     def _list_files(self):
        ...         return [RemoteFile("a.txt"), RemoteFile("sub/b.txt")]
        ...     def _fetch(self, remote, dest):
        ...         dest.write_text(f"contents of {remote.rel_path}")
        >>>
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     report = StubSource().download_all(d)
        ...     files = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
        >>> report.n_downloaded, report.n_skipped, report.n_failed
        (2, 0, 0)
        >>> files
        ['a.txt', 'sub/b.txt']

        Multi-source case — caller owns one :class:`ThreadPoolExecutor` and one
        :class:`DownloadPolicy`, hands them to every source:

        >>> from concurrent.futures import ThreadPoolExecutor
        >>> with tempfile.TemporaryDirectory() as d, ThreadPoolExecutor(max_workers=4) as pool:
        ...     d = Path(d)
        ...     policy = DownloadPolicy(continue_on_error=True)
        ...     for src in [StubSource(), StubSource()]:
        ...         _ = src.download_all(d, pool=pool, policy=policy)
        ...     n_files = len(list(d.rglob("*.txt")))
        >>> n_files
        2
    """

    def download_all(
        self,
        dest_dir: Path,
        *,
        pool: ThreadPoolExecutor | None = None,
        policy: DownloadPolicy | None = None,
    ) -> FetchReport:
        """Download every file this source lists into ``dest_dir``.

        Args:
            dest_dir: Where files land. Created if missing.
            pool: Optional :class:`ThreadPoolExecutor` to submit work to. When
                provided, the caller owns the pool's lifetime — useful when one
                CLI invocation drives multiple sources sequentially and wants to
                share a single pool. When ``None`` (default), this call builds a
                private 4-worker pool with the SIGINT-safe ``cancel_futures``
                shutdown semantics, runs the bundle, and tears the pool down.
            policy: Optional :class:`DownloadPolicy`. ``None`` builds a default
                (no continue-on-error, no force-overwrite).

        Returns:
            A :class:`FetchReport` with per-file outcomes.
        """
        policy = policy or DownloadPolicy()
        if pool is not None:
            return _drive(self, dest_dir, pool, policy)
        # Own the pool for this call. ``shutdown(wait=False, cancel_futures=True)``
        # is critical for SIGINT — the default ``__exit__`` calls
        # ``shutdown(wait=True)``, which would block Ctrl+C until every queued
        # future drains. For a multi-GiB PhysioNet pull that's literal hours.
        # With cancel_futures, queued submissions die immediately and running
        # daemon worker threads are abandoned at interpreter teardown; the OS
        # tears down their sockets on process exit (no main-thread
        # ``httpx.Client.close()`` needed — that path deadlocks on OpenSSL's
        # per-socket lock when called from another thread mid-``SSL_read``).
        owned_pool = ThreadPoolExecutor(max_workers=4)
        try:
            return _drive(self, dest_dir, owned_pool, policy)
        finally:
            owned_pool.shutdown(wait=False, cancel_futures=True)

    @abstractmethod
    def _list_files(self) -> Iterable[RemoteFile]:
        """Enumerate the files this source offers.

        Implementations MAY stream via a generator for large manifests. Callers
        should not assume the result is re-iterable — :meth:`download_all`
        materializes into a ``list`` once.
        """

    @abstractmethod
    def _fetch(self, remote: RemoteFile, dest: Path) -> None:
        """Transport-specific fetch implementation.

        See :class:`Source` invariants. Called by the orchestration loop, not
        directly by users.
        """

    def close(self) -> None:  # noqa: B027 — intentional no-op default; subclasses override when needed
        """Release transport resources held by this source.

        Default is a no-op. Subclasses that own network clients / file handles / connection pools override
        this. Safe to call multiple times; safe to call on sources that own nothing.
        """

    def __enter__(self) -> Source:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
