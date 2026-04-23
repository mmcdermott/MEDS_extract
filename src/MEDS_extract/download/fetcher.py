"""The :class:`Fetcher` orchestrator.

Takes a :class:`Source` and drives it through a bounded-concurrency download plan to a
local ``dest_dir``, producing a :class:`FetchReport`. Per-file skip logic, error
tolerance, and progress logging all live here; the :class:`Source` just declares what
files exist and how to get one.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

from .source import FetchReport, FetchResult, RemoteFile, sha256_of

if TYPE_CHECKING:
    from .source import Source

logger = logging.getLogger(__name__)


class Fetcher:
    """Drives a :class:`Source` through a bounded-concurrency download plan.

    Attributes:
        dest_dir: Root directory that :class:`RemoteFile` ``rel_path`` values land under.
        max_concurrency: Maximum number of parallel transport streams. 4 is polite against
            rate-limiting servers (PhysioNet); 8 is typically safe; 16+ risks a ban.
        continue_on_error: If ``True``, per-file transport exceptions are captured as
            :class:`~MEDS_extract.download.source.FetchResult` with ``status="failed"`` and
            the run proceeds. If ``False`` (default), the first failure is re-raised;
            already-submitted concurrent transfers are **not** explicitly canceled, so
            ``fetch_all()`` may still block until submitted worker tasks finish (the
            ``ThreadPoolExecutor`` context manager calls ``shutdown(wait=True)`` on exit)
            and their results are then discarded.

    Examples:
        ``Fetcher`` can be exercised without any real network via a stub :class:`Source`:

        >>> import tempfile
        >>> from MEDS_extract.download.source import RemoteFile, Source
        >>>
        >>> class StubSource:
        ...     def list_files(self):
        ...         return [RemoteFile("a.txt"), RemoteFile("sub/b.txt")]
        ...     def fetch(self, remote, dest):
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

        >>> class HashSource:
        ...     def list_files(self):
        ...         import hashlib
        ...         body = b"abc"
        ...         digest = hashlib.sha256(body).hexdigest()
        ...         return [RemoteFile("x.txt", size=len(body), sha256=digest)]
        ...     def fetch(self, remote, dest):
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

        >>> class PartialSource:
        ...     def list_files(self):
        ...         return [RemoteFile("good.txt"), RemoteFile("bad.txt")]
        ...     def fetch(self, remote, dest):
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
    """

    def __init__(
        self,
        dest_dir: Path,
        max_concurrency: int = 4,
        continue_on_error: bool = False,
    ):
        self.dest_dir = Path(dest_dir)
        self.max_concurrency = max(1, int(max_concurrency))
        self.continue_on_error = continue_on_error

    def fetch_all(self, source: Source) -> FetchReport:
        """Fetch every file the source lists; return a summary."""
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        remotes = list(source.list_files())
        logger.info(f"Fetching {len(remotes)} files to {self.dest_dir}")

        results: list[FetchResult] = []
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futures = {pool.submit(self._fetch_one, source, r): r for r in remotes}
            for fut in as_completed(futures):
                results.append(fut.result())

        report = FetchReport(results=results)
        logger.info(
            f"Fetch complete: {report.n_downloaded} downloaded, {report.n_skipped} skipped, "
            f"{report.n_failed} failed."
        )
        return report

    def _fetch_one(self, source: Source, remote: RemoteFile) -> FetchResult:
        dest = self._resolve_dest(remote.rel_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self._already_complete(dest, remote):
            logger.debug(f"Skipping {remote.rel_path}: already complete.")
            return FetchResult(remote, dest, "skipped")
        # If dest exists but `_already_complete` returned False, something about the local
        # copy didn't match the manifest (wrong size or wrong SHA-256). Delete it now —
        # otherwise a wrong-size-but-no-SHA `_resumable_download` call would see the file
        # on disk, skip the network hop entirely, and silently leave the corrupt file in
        # place (it only re-checks SHA when `expected_sha256` is set).
        if dest.exists():
            logger.debug(f"Removing incomplete existing file for {remote.rel_path} before refetch.")
            dest.unlink()
        try:
            source.fetch(remote, dest)
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
            >>> import tempfile
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
            >>> import tempfile, hashlib
            >>> from pathlib import Path
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
