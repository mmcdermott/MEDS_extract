"""The :class:`Fetcher` orchestrator plus :class:`FetchResult` / :class:`FetchReport`.

``Fetcher`` takes a :class:`~MEDS_extract.download.source.Source` and drives it through a
bounded-concurrency download plan to a local ``dest_dir``, producing a
:class:`FetchReport`. Per-file skip logic, error tolerance, and progress logging all live
here; the :class:`Source` just declares what files exist and how to get one.

The :class:`FetchResult` / :class:`FetchReport` dataclasses are Fetcher's output types
and live here (not in ``source.py``) because they describe orchestrator-level outcomes,
not properties of the Source itself.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .source import RemoteFile, sha256_of

if TYPE_CHECKING:
    from .source import Source

logger = logging.getLogger(__name__)


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
    """Summary of one :meth:`Fetcher.fetch_all` call.

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
    """Drives a :class:`Source` through a bounded-concurrency download plan.

    Attributes:
        dest_dir: Root directory that :class:`RemoteFile` ``rel_path`` values land under.
        max_concurrency: Maximum number of parallel transport streams. 4 is polite against
            rate-limiting servers (PhysioNet); 8 is typically safe; 16+ risks a ban.
        continue_on_error: If ``True``, per-file transport exceptions are captured as
            :class:`FetchResult` with ``status="failed"`` and the run proceeds. If
            ``False`` (default), the first failure is re-raised. Either way, any
            still-queued transfers are cancelled on exit via
            ``pool.shutdown(wait=False, cancel_futures=True)``; already-running worker
            threads are daemon threads and get abandoned at interpreter teardown
            without blocking the caller.
        do_overwrite: If ``True``, skip the "already complete" short-circuit and pass
            ``do_overwrite=True`` down to the backend's
            :meth:`~MEDS_extract.download.source.Source.fetch` so cached ``.part`` /
            ``dest`` state is cleared. Every file is re-fetched. If ``False`` (default),
            existing files that match the manifest's size / SHA-256 are skipped.

    Examples:
        ``Fetcher`` can be exercised without any real network via a stub :class:`Source`.
        Subclassing :class:`Source` picks up the overwrite-precondition logic for free;
        stubs only need to implement :meth:`~Source.list_files` and :meth:`~Source._fetch`.

        >>> from MEDS_extract.download.source import RemoteFile, Source
        >>>
        >>> class StubSource(Source):
        ...     def list_files(self):
        ...         return [RemoteFile("a.txt"), RemoteFile("sub/b.txt")]
        ...     def _fetch(self, remote, dest):
        ...         dest.write_text(f"contents of {remote.rel_path}")
        >>>
        >>> with tempfile.TemporaryDirectory() as d:
        ...     report = Fetcher(Path(d), max_concurrency=1).fetch_all(StubSource())
        ...     files = sorted(p.relative_to(d).as_posix() for p in Path(d).rglob("*") if p.is_file())
        >>> report.n_downloaded
        2
        >>> report.n_skipped
        0
        >>> report.ok
        True
        >>> files
        ['a.txt', 'sub/b.txt']

        A file that already exists (and has the right content hash if known) is skipped:

        >>> import hashlib
        >>> class HashSource(Source):
        ...     def list_files(self):
        ...         body = b"abc"
        ...         digest = hashlib.sha256(body).hexdigest()
        ...         return [RemoteFile("x.txt", size=len(body), sha256=digest)]
        ...     def _fetch(self, remote, dest):
        ...         dest.write_bytes(b"abc")
        >>>
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     r1 = Fetcher(d).fetch_all(HashSource())
        ...     r2 = Fetcher(d).fetch_all(HashSource())
        >>> r1.n_downloaded
        1
        >>> r2.n_skipped
        1

        With ``continue_on_error=True``, one failure doesn't sink the rest:

        >>> class PartialSource(Source):
        ...     def list_files(self):
        ...         return [RemoteFile("good.txt"), RemoteFile("bad.txt")]
        ...     def _fetch(self, remote, dest):
        ...         if remote.rel_path == "bad.txt":
        ...             raise RuntimeError("transport failure")
        ...         dest.write_text("ok")
        >>>
        >>> with tempfile.TemporaryDirectory() as d:
        ...     report = Fetcher(Path(d), continue_on_error=True).fetch_all(PartialSource())
        >>> report.n_downloaded
        1
        >>> report.n_failed
        1
        >>> report.ok
        False

        Without ``continue_on_error``, the first failure re-raises:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     Fetcher(Path(d), continue_on_error=False).fetch_all(PartialSource())
        Traceback (most recent call last):
            ...
        RuntimeError: transport failure

        ``do_overwrite=True`` forces re-fetch even when the file already exists on disk
        with the right size + hash (normally that path returns ``skipped``):

        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     _ = Fetcher(d).fetch_all(HashSource())
        ...     re_report = Fetcher(d, do_overwrite=True).fetch_all(HashSource())
        >>> re_report.n_downloaded  # re-fetched despite the on-disk match
        1
        >>> re_report.n_skipped
        0

        ``max_concurrency`` is strictly validated — a silent coercion of e.g. ``True``
        to ``1`` (sequential) was the old behavior when we had ``max(1,
        int(max_concurrency))``. Library callers that construct ``Fetcher`` directly
        (bypassing the CLI's Hydra type-check) now get a clear error instead:

        >>> Fetcher("/tmp", max_concurrency=True)
        Traceback (most recent call last):
            ...
        TypeError: max_concurrency must be int, got bool: True
        >>> Fetcher("/tmp", max_concurrency=1.5)
        Traceback (most recent call last):
            ...
        TypeError: max_concurrency must be int, got float: 1.5
        >>> Fetcher("/tmp", max_concurrency="4")
        Traceback (most recent call last):
            ...
        TypeError: max_concurrency must be int, got str: '4'
        >>> Fetcher("/tmp", max_concurrency=0)
        Traceback (most recent call last):
            ...
        ValueError: max_concurrency must be >= 1, got 0
        >>> Fetcher("/tmp", max_concurrency=-3)
        Traceback (most recent call last):
            ...
        ValueError: max_concurrency must be >= 1, got -3

        Positive ints pass through untouched:

        >>> Fetcher("/tmp", max_concurrency=1).max_concurrency
        1
        >>> Fetcher("/tmp", max_concurrency=16).max_concurrency
        16
    """

    def __init__(
        self,
        dest_dir: Path,
        max_concurrency: int = 4,
        continue_on_error: bool = False,
        do_overwrite: bool = False,
    ):
        # Reject non-int / boolean / non-positive ``max_concurrency`` with a clear
        # error rather than silently coercing (``int(True) == 1``, ``int(1.9) == 1``,
        # ``int(False) → 0 → clamp to 1``). The CLI dataclass already type-checks at
        # the Hydra layer, but library callers that construct ``Fetcher`` directly
        # shouldn't get silent surprises like ``Fetcher(max_concurrency=True)``
        # becoming a sequential fetcher. ``isinstance(True, int)`` is ``True`` in
        # Python (bool subclasses int), so the bool check must come first.
        if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int):
            raise TypeError(
                f"max_concurrency must be int, got {type(max_concurrency).__name__}: {max_concurrency!r}"
            )
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        self.dest_dir = Path(dest_dir)
        self.max_concurrency = max_concurrency
        self.continue_on_error = continue_on_error
        self.do_overwrite = do_overwrite

    def fetch_all(self, source: Source) -> FetchReport:
        """Fetch every file the source lists; return a summary."""
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        remotes = list(source.list_files())

        # Reject collisions up-front — two remotes with the same resolved dest would race
        # on the same ``.part`` / dest under concurrent workers and silently corrupt the
        # output. We resolve now (cheap) so the user sees the duplicate before we spin up
        # the thread pool and start network I/O.
        seen: dict[Path, RemoteFile] = {}
        for remote in remotes:
            dest = self._resolve_dest(remote.rel_path)
            if dest in seen:
                raise ValueError(
                    f"Duplicate destination {dest}: rel_path {remote.rel_path!r} collides "
                    f"with {seen[dest].rel_path!r}. Each RemoteFile in a source's "
                    "list_files() must resolve to a unique dest_dir-relative path."
                )
            seen[dest] = remote

        logger.info(f"Fetching {len(remotes)} files to {self.dest_dir}")

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
            futures = {pool.submit(self._fetch_one, source, r): r for r in remotes}
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

    def _fetch_one(self, source: Source, remote: RemoteFile) -> FetchResult:
        dest = self._resolve_dest(remote.rel_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not self.do_overwrite and self._already_complete(dest, remote):
            logger.debug(f"Skipping {remote.rel_path}: already complete.")
            return FetchResult(remote, dest, "skipped")
        # Force overwrite if either the caller requested it OR the on-disk copy failed the
        # completeness check (wrong size / wrong sha). Both cases are "the file is wrong
        # and needs to be refetched cleanly"; the base ``Source.fetch`` clears the stale
        # ``dest`` / ``.part`` when ``do_overwrite=True``, so we route both through the
        # same precondition path.
        force = self.do_overwrite or dest.exists()
        try:
            source.fetch(remote, dest, do_overwrite=force)
            return FetchResult(remote, dest, "downloaded")
        except Exception as e:
            if not self.continue_on_error:
                raise
            logger.error(f"Failed to fetch {remote.rel_path}: {e}")
            return FetchResult(remote, dest, "failed", error=e)

    def _resolve_dest(self, rel_path: str) -> Path:
        """Resolve ``rel_path`` under ``dest_dir``, rejecting any escape attempts.

        A malformed manifest could ship an absolute path or one containing ``..`` segments
        that would land the fetched file outside ``dest_dir``. Both are rejected eagerly
        before we touch the filesystem — this is a hard user-facing error, not a silent
        clamp, because a rejected path always indicates a broken manifest or malicious
        source.

        Examples:
            >>> with tempfile.TemporaryDirectory() as d:
            ...     f = Fetcher(Path(d))
            ...     f._resolve_dest("sub/ok.txt").relative_to(Path(d).resolve()).as_posix()
            'sub/ok.txt'

            >>> with tempfile.TemporaryDirectory() as d:
            ...     f = Fetcher(Path(d))
            ...     f._resolve_dest("/etc/passwd")
            Traceback (most recent call last):
                ...
            ValueError: rel_path must be relative, got absolute: '/etc/passwd'

            >>> with tempfile.TemporaryDirectory() as d:
            ...     f = Fetcher(Path(d))
            ...     f._resolve_dest("../../etc/passwd")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: rel_path '../../etc/passwd' escapes dest_dir ...
        """
        rp = Path(rel_path)
        if rp.is_absolute():
            raise ValueError(f"rel_path must be relative, got absolute: {rel_path!r}")
        dest_root = self.dest_dir.resolve()
        resolved = (dest_root / rp).resolve()
        try:
            resolved.relative_to(dest_root)
        except ValueError as e:
            raise ValueError(
                f"rel_path {rel_path!r} escapes dest_dir {dest_root} (resolved to {resolved})."
            ) from e
        return resolved

    @staticmethod
    def _already_complete(dest: Path, remote: RemoteFile) -> bool:
        """True if ``dest`` already has the right content per ``remote``'s manifest info.

        The cheaper size check gates the expensive SHA-256 check. When neither is known,
        the file's presence is taken as sufficient — callers can force a re-fetch by
        deleting the local copy.

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
            ...     Fetcher._already_complete(fp, RemoteFile("x.txt", size=100))  # size mismatch
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
        if remote.size is not None and dest.stat().st_size != remote.size:
            return False
        return not (remote.sha256 is not None and sha256_of(dest) != remote.sha256)
