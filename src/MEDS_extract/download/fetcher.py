"""Shared download-policy holder + per-item drive loop.

A :class:`Fetcher` is the *policy* a :class:`~MEDS_extract.download.source.Source` uses
when it downloads its files: how many parallel workers, whether per-file failures sink
the run, whether to refetch already-cached copies. One instance is built once (typically
from CLI flags) and shared across every :class:`Source` in a MESSY ``sources:`` block —
that's why it lives off the source rather than on it.

There is no public per-file fetch entry point. The only way to use a :class:`Fetcher`
is to attach it to a :class:`Source` and call :meth:`Source.download_all`. Everything
in this file other than the constructor + :class:`FetchReport` / :class:`FetchResult`
is meant to be called from :class:`Source` and not from user code; the underscore
prefix on :meth:`Fetcher._drive` makes that explicit.

See https://github.com/mmcdermott/MEDS_extract/pull/96 for the design discussion.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ._types import RemoteFile, sha256_of
from .unarchive import ArchiveFormat, resolve_format, safe_extract

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FetchResult:
    """The outcome of fetching one :class:`~MEDS_extract.download.source.RemoteFile`.

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
    """Summary of one :meth:`Source.download_all <MEDS_extract.download.source.Source.download_all>` call.

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


class Fetcher:
    """Bounded-concurrency download policy.

    Construct one (typically from CLI flags), pass it to one or more
    :class:`~MEDS_extract.download.source.Source` instances at their construction, and
    call :meth:`Source.download_all <MEDS_extract.download.source.Source.download_all>`
    on each. The :class:`Fetcher` itself has no public fetch method — it's pure policy.

    Attributes:
        max_concurrency: Maximum number of parallel transport streams per
            :meth:`Source.download_all <MEDS_extract.download.source.Source.download_all>`
            call. 4 is polite against rate-limiting servers (PhysioNet); 8 is typically
            safe; 16+ risks a ban.
        continue_on_error: If ``True``, per-file transport exceptions are captured as
            :class:`FetchResult` with ``status="failed"`` and the run proceeds. If
            ``False`` (default), the first failure is re-raised. Either way, any
            still-queued transfers are cancelled on exit via
            ``pool.shutdown(wait=False, cancel_futures=True)``; already-running worker
            threads are daemon threads and get abandoned at interpreter teardown
            without blocking the caller.
        do_overwrite: If ``True``, skip the "already complete" short-circuit and force
            a re-fetch of every file. Existing ``dest`` / ``.part`` state is cleared.
            If ``False`` (default), existing files that match the manifest's
            ``size`` / ``sha256`` are skipped.

    Examples:
        ``max_concurrency`` is strictly validated — silent coercion of e.g. ``True``
        to ``1`` would mask configuration mistakes:

        >>> Fetcher(max_concurrency=True)
        Traceback (most recent call last):
            ...
        TypeError: max_concurrency must be int, got bool: True
        >>> Fetcher(max_concurrency=1.5)
        Traceback (most recent call last):
            ...
        TypeError: max_concurrency must be int, got float: 1.5
        >>> Fetcher(max_concurrency="4")
        Traceback (most recent call last):
            ...
        TypeError: max_concurrency must be int, got str: '4'
        >>> Fetcher(max_concurrency=0)
        Traceback (most recent call last):
            ...
        ValueError: max_concurrency must be >= 1, got 0
        >>> Fetcher(max_concurrency=-3)
        Traceback (most recent call last):
            ...
        ValueError: max_concurrency must be >= 1, got -3

        Positive ints pass through:

        >>> Fetcher(max_concurrency=1).max_concurrency
        1
        >>> Fetcher(max_concurrency=16).max_concurrency
        16
        >>> default = Fetcher()
        >>> default.max_concurrency, default.continue_on_error, default.do_overwrite
        (4, False, False)
    """

    def __init__(
        self,
        max_concurrency: int = 4,
        continue_on_error: bool = False,
        do_overwrite: bool = False,
    ):
        # Reject non-int / boolean / non-positive ``max_concurrency`` with a clear
        # error rather than silently coercing (``int(True) == 1``, ``int(1.9) == 1``,
        # ``int(False) → 0 → clamp to 1``). The CLI dataclass already type-checks at
        # the Hydra layer, but library callers shouldn't get silent surprises like
        # ``Fetcher(max_concurrency=True)`` becoming a sequential fetcher.
        # ``isinstance(True, int)`` is ``True`` in Python (bool subclasses int), so
        # the bool check must come first.
        if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int):
            raise TypeError(
                f"max_concurrency must be int, got {type(max_concurrency).__name__}: {max_concurrency!r}"
            )
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        self.max_concurrency = max_concurrency
        self.continue_on_error = continue_on_error
        self.do_overwrite = do_overwrite

    def _drive(
        self,
        items: list[RemoteFile],
        dest_dir: Path,
        fetch_one_fn: Callable[[RemoteFile, Path], None],
    ) -> FetchReport:
        """Run ``fetch_one_fn`` over ``items`` under this fetcher's policy.

        Called by :meth:`Source.download_all`. Not part of the public surface — the
        only way users invoke fetch behavior is via a :class:`Source`.

        ``items`` is the result of the source's ``_list_files()``. ``fetch_one_fn`` is
        the source's ``_fetch`` method (transport-specific; writes one file's bytes to
        a path the fetcher chooses).
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Reject collisions up-front — two items with the same resolved dest would race
        # on the same ``.part`` / dest under concurrent workers and silently corrupt
        # the output. We resolve now (cheap) before spinning up the thread pool.
        seen: dict[Path, RemoteFile] = {}
        for item in items:
            dest = self._resolve_dest(dest_dir, item.rel_path)
            if dest in seen:
                raise ValueError(
                    f"Duplicate destination {dest}: rel_path {item.rel_path!r} collides "
                    f"with {seen[dest].rel_path!r}. Each item from a source's "
                    "_list_files() must resolve to a unique dest_dir-relative path."
                )
            seen[dest] = item

        logger.info(f"Fetching {len(items)} files to {dest_dir}")

        results: list[FetchResult] = []
        # Build the pool without a context manager on purpose: ``__exit__`` calls
        # ``shutdown(wait=True)``, which would block a Ctrl+C (or any unwinding
        # exception) until every queued future drains — for a multi-GiB PhysioNet
        # pull, that's literal hours. Instead we shutdown in a ``finally`` with
        # ``wait=False, cancel_futures=True`` so queued futures die immediately and
        # running worker threads (daemon by default in CPython's
        # ``ThreadPoolExecutor``) get abandoned at interpreter teardown. The OS
        # tears down their sockets as part of process shutdown; no explicit
        # main-thread ``httpx.Client.close()`` / ``SSL_shutdown`` is needed (the
        # latter deadlocks on OpenSSL's per-socket lock when called from another
        # thread while a worker is mid-``SSL_read``).
        pool = ThreadPoolExecutor(max_workers=self.max_concurrency)
        try:
            futures = {pool.submit(self._fetch_one, item, dest_dir, fetch_one_fn): item for item in items}
            for fut in as_completed(futures):
                results.append(fut.result())
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        report = FetchReport(results=results)
        logger.info(
            f"Fetch complete: {report.n_downloaded} downloaded, {report.n_skipped} skipped, "
            f"{report.n_failed} failed."
        )
        return report

    def _fetch_one(
        self,
        item: RemoteFile,
        dest_dir: Path,
        fetch_one_fn: Callable[[RemoteFile, Path], None],
    ) -> FetchResult:
        """Apply skip/overwrite/unarchive policy to one item and call its transport."""
        dest = self._resolve_dest(dest_dir, item.rel_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not self.do_overwrite and self._already_complete(dest, item):
            logger.debug(f"Skipping {item.rel_path}: already complete.")
            return FetchResult(item, dest, "skipped")
        # Force overwrite if either the caller requested it OR the on-disk copy
        # failed the completeness check. Both cases are "the file is wrong and
        # needs a clean refetch"; clear stale ``dest`` / ``.part`` before
        # handing off to the transport hook.
        force = self.do_overwrite or dest.exists()
        part = dest.with_name(dest.name + ".part")
        if force:
            if dest.exists():
                dest.unlink()
            if part.exists():
                part.unlink()
        try:
            fetch_one_fn(item, dest)
            self._maybe_unarchive(item, dest)
            return FetchResult(item, dest, "downloaded")
        except Exception as e:
            if not self.continue_on_error:
                raise
            logger.error(f"Failed to fetch {item.rel_path}: {e}")
            return FetchResult(item, dest, "failed", error=e)

    @staticmethod
    def _maybe_unarchive(item: RemoteFile, dest: Path) -> None:
        """Post-fetch unpack hook — runs after the transport finishes successfully.

        Lives on :class:`Fetcher` rather than the transport so per-file
        ``Source._fetch`` calls keep the clean "exact file at exact path" semantics.
        Unarchive is fundamentally an *orchestration* concern (it changes the dest
        from one file to many), so it belongs at the orchestrator layer.

        Tri-state cleanup logic: ``cleanup_archive=None`` defers to the unarchive
        mode — ``AUTO`` removes the archive (the one-arg "fetch + extract + drop"
        flow), explicit formats keep it. Explicit ``True`` / ``False`` always wins.
        """
        if not item.unarchive:
            return
        fmt = resolve_format(item.unarchive, dest)
        if fmt is None:
            return
        safe_extract(dest, dest.parent, fmt)
        if item.cleanup_archive is None:
            cleanup = ArchiveFormat(item.unarchive) is ArchiveFormat.AUTO
        else:
            cleanup = item.cleanup_archive
        if cleanup:
            dest.unlink()

    @staticmethod
    def _resolve_dest(dest_dir: Path, rel_path: str) -> Path:
        """Resolve ``rel_path`` under ``dest_dir``, rejecting any escape attempts.

        A malformed manifest could ship an absolute path or one containing ``..``
        segments that would land the fetched file outside ``dest_dir``. Both are
        rejected eagerly before we touch the filesystem.

        Examples:
            >>> with tempfile.TemporaryDirectory() as d:
            ...     d = Path(d)
            ...     Fetcher._resolve_dest(d, "sub/ok.txt").relative_to(d.resolve()).as_posix()
            'sub/ok.txt'

            >>> with tempfile.TemporaryDirectory() as d:
            ...     Fetcher._resolve_dest(Path(d), "/etc/passwd")
            Traceback (most recent call last):
                ...
            ValueError: rel_path must be relative, got absolute: '/etc/passwd'

            >>> with tempfile.TemporaryDirectory() as d:
            ...     Fetcher._resolve_dest(Path(d), "../../etc/passwd")  # doctest: +ELLIPSIS
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

    @staticmethod
    def _already_complete(dest: Path, item: RemoteFile) -> bool:
        """True if ``dest`` already has the right content per ``item``'s manifest info.

        The cheaper size check gates the expensive SHA-256 check. When neither is
        known, the file's presence is taken as sufficient — callers can force a
        re-fetch by deleting the local copy or constructing the :class:`Fetcher` with
        ``do_overwrite=True``.

        Examples:
            >>> import hashlib
            >>> with tempfile.TemporaryDirectory() as d:
            ...     d = Path(d)
            ...     fp = d / "x.txt"
            ...     _ = fp.write_bytes(b"abc")
            ...     Fetcher._already_complete(fp, RemoteFile("x.txt"))  # no manifest info
            True
            >>> with tempfile.TemporaryDirectory() as d:
            ...     fp = Path(d) / "missing.txt"
            ...     Fetcher._already_complete(fp, RemoteFile("missing.txt"))
            False
            >>> with tempfile.TemporaryDirectory() as d:
            ...     fp = Path(d) / "wrong_size.txt"
            ...     _ = fp.write_bytes(b"abc")
            ...     Fetcher._already_complete(fp, RemoteFile("x.txt", size=100))
            False
            >>> with tempfile.TemporaryDirectory() as d:
            ...     fp = Path(d) / "bad_hash.txt"
            ...     _ = fp.write_bytes(b"abc")
            ...     Fetcher._already_complete(fp, RemoteFile("x.txt", sha256="deadbeef"))
            False
            >>> with tempfile.TemporaryDirectory() as d:
            ...     fp = Path(d) / "good.txt"
            ...     _ = fp.write_bytes(b"abc")
            ...     digest = hashlib.sha256(b"abc").hexdigest()
            ...     Fetcher._already_complete(fp, RemoteFile("x.txt", size=3, sha256=digest))
            True
        """
        if not dest.exists():
            return False
        if item.size is not None and dest.stat().st_size != item.size:
            return False
        return not (item.sha256 is not None and sha256_of(dest) != item.sha256)
