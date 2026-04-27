"""The :class:`Source` ABC, its manifest type :class:`RemoteFile`, and the orchestration loop.

A :class:`Source` is anywhere raw data comes from — a PhysioNet dataset release, an
explicit list of HTTP URLs, an S3 / GCS / local-filesystem tree. Concrete sources
inherit from this ABC and implement two methods:

- :meth:`Source.list_files` — enumerate what files the source offers.
- :meth:`Source._fetch` — move one file's bytes from the source to a local path.

The single public fetch entry point is :meth:`Source.download_all`. It optionally takes
a :class:`ThreadPoolExecutor` (so multiple sources in one CLI invocation can share a
pool) and ``continue_on_error`` / ``do_overwrite`` per-file flags. When no pool is
given, ``download_all`` builds a private 4-worker pool, runs the bundle, and tears
down — the simple-case caller writes ``src.download_all(dest)`` and gets sensible
behavior.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


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


class RemoteFile(NamedTuple):
    """One manifest row from a :class:`Source`.

    Internal: the only legitimate construction sites are inside a backend's
    ``list_files`` and inside test stub sources. Users never see one passed in or
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
    """

    rel_path: str
    size: int | None = None
    sha256: str | None = None
    extra: dict = {}  # noqa: RUF012 — read-only; backends pass fresh dicts


class Source(ABC):
    """A place raw data comes from.

    Subclasses implement two hooks:

    - :meth:`list_files` — enumerate what files the source offers, as
      :class:`RemoteFile` rows.
    - :meth:`_fetch` — move one file's bytes from the source to a local path.

    The base class supplies :meth:`download_all` as the public fetch entry point.

    Invariants implementations must uphold:

    - :meth:`list_files` is idempotent across calls. Re-enumerating must produce
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
        Simple case — no pool passed, ``download_all`` builds and tears down its
        own private pool. A successful run returns ``None``; failure raises:

        >>> class StubSource(Source):
        ...     def list_files(self):
        ...         return [RemoteFile("a.txt"), RemoteFile("sub/b.txt")]
        ...     def _fetch(self, remote, dest):
        ...         dest.write_text(f"contents of {remote.rel_path}")
        >>>
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     StubSource().download_all(d)
        ...     files = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
        >>> files
        ['a.txt', 'sub/b.txt']

        Multi-source case — caller owns one :class:`ThreadPoolExecutor`, hands it
        to every source so the worker cap is global:

        >>> from concurrent.futures import ThreadPoolExecutor
        >>> with tempfile.TemporaryDirectory() as d, ThreadPoolExecutor(max_workers=4) as pool:
        ...     d = Path(d)
        ...     for src in [StubSource(), StubSource()]:
        ...         src.download_all(d, pool=pool)
        ...     n_files = len(list(d.rglob("*.txt")))
        >>> n_files
        2

        Already-complete files are skipped — ``_fetch`` is not invoked for any
        :class:`RemoteFile` whose on-disk copy already matches the manifest's
        ``size`` and (if set) ``sha256``:

        >>> import hashlib
        >>> body = b"abc"
        >>> digest = hashlib.sha256(body).hexdigest()
        >>>
        >>> class SkipSource(Source):
        ...     def list_files(self):
        ...         return [RemoteFile("x.txt", size=len(body), sha256=digest)]
        ...     def _fetch(self, remote, dest):
        ...         raise RuntimeError("must not be called — file is already complete")
        >>>
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     _ = (d / "x.txt").write_bytes(body)
        ...     SkipSource().download_all(d)  # no exception → already-complete skip worked

        With ``continue_on_error=True``, per-file failures are collected and a
        single :class:`ExceptionGroup` is raised at the end so the caller still
        sees every error. ``do_overwrite=True`` forces a re-fetch even when the
        local copy matches.

        Path-traversal manifests are rejected up-front before any I/O:

        >>> class EscapingSource(Source):
        ...     def list_files(self):
        ...         return [RemoteFile("../escape.txt")]
        ...     def _fetch(self, remote, dest):
        ...         dest.write_text("never reached")
        >>>
        >>> with tempfile.TemporaryDirectory() as d:
        ...     EscapingSource().download_all(Path(d))
        Traceback (most recent call last):
            ...
        ValueError: rel_path '../escape.txt' escapes dest_dir ...

        Two manifest entries that resolve to the same dest collide before any
        worker runs — a duplicate ``rel_path`` would otherwise race on the same
        ``.part`` file under concurrent workers:

        >>> class DupSource(Source):
        ...     def list_files(self):
        ...         return [RemoteFile("a.txt"), RemoteFile("a.txt")]
        ...     def _fetch(self, remote, dest):
        ...         dest.write_text("never reached")
        >>>
        >>> with tempfile.TemporaryDirectory() as d:
        ...     DupSource().download_all(Path(d))
        Traceback (most recent call last):
            ...
        ValueError: Duplicate destination ...
    """

    def download_all(
        self,
        dest_dir: Path,
        *,
        pool: ThreadPoolExecutor | None = None,
        continue_on_error: bool = False,
        do_overwrite: bool = False,
    ) -> None:
        """Download every file this source lists into ``dest_dir``.

        Args:
            dest_dir: Where files land. Created if missing.
            pool: Optional :class:`ThreadPoolExecutor` to submit work to. When
                provided, the caller owns the pool's lifetime — useful when one
                CLI invocation drives multiple sources sequentially and wants to
                share a single pool. When ``None`` (default), this call builds a
                private 4-worker pool with the SIGINT-safe ``cancel_futures``
                shutdown semantics, runs the bundle, and tears the pool down.
            continue_on_error: If ``False`` (default), the first per-file failure
                propagates. If ``True``, per-file errors are collected and raised
                as a single :class:`ExceptionGroup` at the end so the caller sees
                every failure, not just the first.
            do_overwrite: If ``True``, skip the "already complete" short-circuit
                and re-fetch every file. Existing ``dest`` / ``.part`` state is
                cleared before each fetch.

        Raises:
            Exception: From the transport layer on any per-file failure when
                ``continue_on_error=False``.
            ExceptionGroup: When ``continue_on_error=True`` and at least one
                file failed.
        """
        if pool is not None:
            return self._drive(dest_dir, pool, continue_on_error, do_overwrite)
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
            return self._drive(dest_dir, owned_pool, continue_on_error, do_overwrite)
        finally:
            owned_pool.shutdown(wait=False, cancel_futures=True)

    @abstractmethod
    def list_files(self) -> Iterable[RemoteFile]:
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

    @staticmethod
    def _resolve_dest(dest_dir: Path, rel_path: str) -> Path:
        """Resolve ``rel_path`` under ``dest_dir``, rejecting any escape attempts.

        A malformed manifest could ship an absolute path or one containing ``..``
        segments that would land the fetched file outside ``dest_dir``. Both are
        rejected eagerly before we touch the filesystem.
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

    @staticmethod
    def _already_complete(dest: Path, item: RemoteFile) -> bool:
        """True if ``dest`` already has the right content per ``item``'s manifest info.

        The cheaper size check gates the expensive SHA-256 check. When neither is
        known, the file's presence is taken as sufficient — callers can force a
        re-fetch by deleting the local copy or passing ``do_overwrite=True`` to
        :meth:`download_all`.
        """
        if not dest.exists():
            return False
        if item.size is not None and dest.stat().st_size != item.size:
            return False
        return not (item.sha256 is not None and sha256_of(dest) != item.sha256)

    def _fetch_one(self, item: RemoteFile, dest_dir: Path, do_overwrite: bool) -> None:
        """Apply skip/overwrite policy to one item and call its transport hook."""
        dest = self._resolve_dest(dest_dir, item.rel_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not do_overwrite and self._already_complete(dest, item):
            logger.debug(f"Skipping {item.rel_path}: already complete.")
            return
        # Force overwrite if either the caller requested it OR the on-disk copy
        # failed the completeness check. Both cases are "the file is wrong and
        # needs a clean refetch"; clear stale ``dest`` / ``.part`` before
        # handing off to the transport hook.
        part = dest.with_name(dest.name + ".part")
        if dest.exists():
            dest.unlink()
        if do_overwrite and part.exists():
            part.unlink()
        self._fetch(item, dest)

    def _drive(
        self,
        dest_dir: Path,
        pool: ThreadPoolExecutor,
        continue_on_error: bool,
        do_overwrite: bool,
    ) -> None:
        """Submit every manifest item to ``pool``; collect or re-raise failures.

        Pool ownership: ``pool`` is borrowed, never owned. The caller (typically
        :meth:`download_all` when it builds its own pool, or the CLI when it
        builds a shared pool with ``with ThreadPoolExecutor(...) as pool:``) is
        responsible for ``pool.shutdown``.
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        items = list(self.list_files())

        # Reject collisions up-front — two items with the same resolved dest would
        # race on the same ``.part`` / dest under concurrent workers and silently
        # corrupt the output. Resolve now (cheap) before submitting work.
        seen: dict[Path, RemoteFile] = {}
        for item in items:
            dest = self._resolve_dest(dest_dir, item.rel_path)
            if dest in seen:
                raise ValueError(
                    f"Duplicate destination {dest}: rel_path {item.rel_path!r} collides "
                    f"with {seen[dest].rel_path!r}. Each item from a source's "
                    "list_files() must resolve to a unique dest_dir-relative path."
                )
            seen[dest] = item

        logger.info(f"Fetching {len(items)} files to {dest_dir}")
        futures = {pool.submit(self._fetch_one, item, dest_dir, do_overwrite): item for item in items}

        errors: list[Exception] = []
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                if not continue_on_error:
                    raise
                item = futures[fut]
                logger.error(f"Failed to fetch {item.rel_path}: {e}")
                errors.append(e)
        if errors:
            raise ExceptionGroup(f"{len(errors)} of {len(items)} files failed to download", errors)
