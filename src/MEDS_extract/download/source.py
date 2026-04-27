"""The :class:`Source` ABC, its manifest type :class:`RemoteFile`, and the orchestration loop.

A :class:`Source` is anywhere raw data comes from — a PhysioNet dataset release, an
explicit list of HTTP URLs, an S3 / GCS / local-filesystem tree. Concrete sources
inherit from this ABC and implement two methods:

- :meth:`Source._list_files` — enumerate what files the source offers (the
  validating wrapper :meth:`Source.list_files` is what callers use).
- :meth:`Source._fetch` — move one file's bytes from the source to a local path.

The single public fetch entry point is :meth:`Source.download_all`. By default it
runs sequentially; pass a :class:`ThreadPoolExecutor` to parallelize. The caller
always owns the pool's lifetime — there's no implicit pool building inside the
download module.
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
    """One manifest row from a :class:`Source` — pure POD.

    Internal: the only legitimate construction sites are inside a backend's
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
        source_path: The source-side address as a plain string. HTTP-backed sources
            put the absolute URL here; fsspec-backed sources put the
            :class:`~upath.UPath` spec (which the backend re-instantiates as a
            ``UPath`` inside :meth:`Source._fetch`).
    """

    rel_path: str
    size: int | None = None
    sha256: str | None = None
    source_path: str | None = None


class Source(ABC):
    """A place raw data comes from.

    Subclasses implement two hooks:

    - :meth:`_list_files` — enumerate what files the source offers, as
      :class:`RemoteFile` rows. The base class wraps this in :meth:`list_files`,
      which materializes the result and validates that every row resolves to a
      unique destination — duplicate rel_paths would race on the same ``.part``
      file under concurrent workers.
    - :meth:`_fetch` — move one file's bytes from the source to a local path.

    The base class supplies :meth:`download_all` as the public fetch entry point.
    By default it runs sequentially; pass ``pool=`` to parallelize.

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
      the skip / overwrite / error logic has already run.

    Examples:
        Simple case — no pool passed, ``download_all`` runs sequentially. A
        successful run returns ``None``; failure raises:

        >>> class StubSource(Source):
        ...     def _list_files(self):
        ...         return [RemoteFile("a.txt"), RemoteFile("sub/b.txt")]
        ...     def _fetch(self, remote, dest):
        ...         dest.write_text(f"contents of {remote.rel_path}")
        >>>
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     StubSource().download_all(d)
        ...     print_directory(d)
        ├── a.txt
        └── sub
            └── b.txt

        Multi-source case — caller owns one :class:`ThreadPoolExecutor` and hands
        it to every source. Two sources writing distinct files into one
        ``dest_dir`` is the typical CLI pattern (e.g. one ``physionet`` source
        plus one ``http`` source for a metadata bundle):

        >>> from concurrent.futures import ThreadPoolExecutor
        >>>
        >>> class SourceA(Source):
        ...     def _list_files(self):
        ...         return [RemoteFile("a.txt")]
        ...     def _fetch(self, remote, dest):
        ...         dest.write_text("from A")
        >>>
        >>> class SourceB(Source):
        ...     def _list_files(self):
        ...         return [RemoteFile("metadata/b.csv")]
        ...     def _fetch(self, remote, dest):
        ...         dest.write_text("from B")
        >>>
        >>> with tempfile.TemporaryDirectory() as d, ThreadPoolExecutor(max_workers=4) as pool:
        ...     d = Path(d)
        ...     for src in [SourceA(), SourceB()]:
        ...         src.download_all(d, pool=pool)
        ...     print_directory(d)
        ├── a.txt
        └── metadata
            └── b.csv

        Already-complete files are skipped — ``_fetch`` is not invoked for any
        :class:`RemoteFile` whose on-disk copy verifies against the manifest's
        ``size`` and (if set) ``sha256``:

        >>> import hashlib
        >>> body = b"abc"
        >>> digest = hashlib.sha256(body).hexdigest()
        >>>
        >>> class SkipSource(Source):
        ...     def _list_files(self):
        ...         return [RemoteFile("x.txt", size=len(body), sha256=digest)]
        ...     def _fetch(self, remote, dest):
        ...         raise RuntimeError("must not be called — file is already complete")
        >>>
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     _ = (d / "x.txt").write_bytes(body)
        ...     SkipSource().download_all(d)  # no exception → already-complete skip worked

        An existing ``dest`` that **doesn't** verify (size mismatch, sha mismatch,
        or no manifest verifier at all) is a hard error rather than a silent
        overwrite. The user has to opt in to overwriting via ``do_overwrite=True``,
        which forces a clean refetch:

        >>> class UnverifiableSource(Source):
        ...     def _list_files(self):
        ...         return [RemoteFile("x.txt")]  # no size, no sha
        ...     def _fetch(self, remote, dest):
        ...         dest.write_text("fresh")
        >>>
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     _ = (d / "x.txt").write_bytes(b"stale")
        ...     UnverifiableSource().download_all(d)
        Traceback (most recent call last):
            ...
        FileExistsError: Refusing to overwrite ...x.txt: ... do_overwrite=True ...

        With ``continue_on_error=True``, per-file failures are collected and a
        single :class:`ExceptionGroup` is raised at the end so the caller still
        sees every error.

        Path-traversal manifests are rejected up-front before any I/O:

        >>> class EscapingSource(Source):
        ...     def _list_files(self):
        ...         return [RemoteFile("../escape.txt")]
        ...     def _fetch(self, remote, dest):
        ...         dest.write_text("never reached")
        >>>
        >>> with tempfile.TemporaryDirectory() as d:
        ...     EscapingSource().download_all(Path(d))
        Traceback (most recent call last):
            ...
        ValueError: rel_path '../escape.txt' escapes dest_dir ...

        Two manifest entries that resolve to the same dest collide at
        :meth:`list_files` time — this is the validation wrapper around the
        subclass-provided ``_list_files`` hook:

        >>> class DupSource(Source):
        ...     def _list_files(self):
        ...         return [RemoteFile("a.txt"), RemoteFile("a.txt")]
        ...     def _fetch(self, remote, dest):
        ...         dest.write_text("never reached")
        >>>
        >>> DupSource().list_files()
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
            pool: Optional :class:`ThreadPoolExecutor` to submit work to. The
                caller owns the pool's lifetime. When ``None`` (default), the
                bundle is fetched sequentially in the calling thread — no thread
                pool is created. Pass a pool when you want parallelism, sized to
                whatever your transport tolerates.
            continue_on_error: If ``False`` (default), the first per-file failure
                propagates. If ``True``, per-file errors are collected and raised
                as a single :class:`ExceptionGroup` at the end so the caller sees
                every failure, not just the first.
            do_overwrite: If ``True``, skip the "verified or error" check and
                clear ``dest`` / ``.part`` before each fetch — re-fetches
                everything from scratch.

        Raises:
            Exception: From the transport layer on any per-file failure when
                ``continue_on_error=False``.
            ExceptionGroup: When ``continue_on_error=True`` and at least one
                file failed.
            FileExistsError: When an existing ``dest`` can't be verified against
                the manifest and ``do_overwrite=False``.
            ValueError: When the manifest contains an unsafe rel_path or
                duplicate destinations (raised by :meth:`list_files`).
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        items = self.list_files()

        if pool is None:
            errors = self._run_sequential(items, dest_dir, continue_on_error, do_overwrite)
        else:
            errors = self._run_parallel(items, dest_dir, pool, continue_on_error, do_overwrite)

        if errors:
            raise ExceptionGroup(f"{len(errors)} of {len(items)} files failed to download", errors)

    def list_files(self) -> list[RemoteFile]:
        """Return the validated manifest — calls :meth:`_list_files`, materializes, and rejects duplicate
        destinations.

        Materialized to a ``list`` here (not in :meth:`download_all`) so the
        validation pass and the orchestration pass don't double-iterate a
        generator. Subclasses MAY implement :meth:`_list_files` as a generator —
        this method is the one stable place we collapse it into a list.
        """
        items = list(self._list_files())
        seen: dict[str, RemoteFile] = {}
        for item in items:
            if item.rel_path in seen:
                raise ValueError(
                    f"Duplicate destination {item.rel_path!r}: collides with "
                    f"{seen[item.rel_path].rel_path!r}. Each item from a source's "
                    "_list_files() must have a unique rel_path."
                )
            seen[item.rel_path] = item
        return items

    @abstractmethod
    def _list_files(self) -> Iterable[RemoteFile]:
        """Subclass hook — enumerate the files this source offers.

        Implementations MAY stream via a generator for large manifests.
        :meth:`list_files` is the validating wrapper that callers (including
        :meth:`download_all`) use — it materializes the result and rejects
        duplicate ``rel_path`` collisions.
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
    def _verifies(dest: Path, item: RemoteFile) -> bool:
        """True iff ``dest`` exists AND positively verifies against the manifest.

        "Positively verifies" requires at least one of ``size`` / ``sha256`` to
        be set on the manifest row, and every set verifier to match. Existence
        alone is not sufficient — without something to compare against, the file
        on disk could be anything.
        """
        if not dest.exists():
            return False
        if item.size is None and item.sha256 is None:
            return False
        if item.size is not None and dest.stat().st_size != item.size:
            return False
        return not (item.sha256 is not None and sha256_of(dest) != item.sha256)

    def _fetch_one(self, item: RemoteFile, dest_dir: Path, do_overwrite: bool) -> None:
        """Apply skip / error / overwrite policy to one item and call its transport hook.

        - dest doesn't exist → fetch
        - dest exists + ``do_overwrite=True`` → clear ``dest``/``.part`` then fetch
          (always wins; no verification check)
        - dest exists + verifies against manifest → skip
        - dest exists + can't verify (mismatch or no manifest verifier) → raise
          :class:`FileExistsError`
        """
        dest = self._resolve_dest(dest_dir, item.rel_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            if do_overwrite:
                dest.unlink()
                part = dest.with_name(dest.name + ".part")
                if part.exists():
                    part.unlink()
            elif self._verifies(dest, item):
                logger.debug(f"Skipping {item.rel_path}: already complete.")
                return
            else:
                raise FileExistsError(
                    f"Refusing to overwrite {dest}: existing file does not verify against "
                    f"the manifest (size or sha mismatch, or no verifier provided). Pass "
                    f"do_overwrite=True to force a refetch, or delete the file first."
                )
        self._fetch(item, dest)

    def _run_sequential(
        self,
        items: list[RemoteFile],
        dest_dir: Path,
        continue_on_error: bool,
        do_overwrite: bool,
    ) -> list[Exception]:
        logger.info(f"Fetching {len(items)} files to {dest_dir} (sequential)")
        errors: list[Exception] = []
        for item in items:
            try:
                self._fetch_one(item, dest_dir, do_overwrite)
            except Exception as e:
                if not continue_on_error:
                    raise
                logger.error(f"Failed to fetch {item.rel_path}: {e}")
                errors.append(e)
        return errors

    def _run_parallel(
        self,
        items: list[RemoteFile],
        dest_dir: Path,
        pool: ThreadPoolExecutor,
        continue_on_error: bool,
        do_overwrite: bool,
    ) -> list[Exception]:
        """Submit every manifest item to ``pool``; collect or re-raise failures.

        Pool ownership: ``pool`` is borrowed, never owned. The caller is
        responsible for ``pool.shutdown``.
        """
        logger.info(f"Fetching {len(items)} files to {dest_dir} (pool={pool!r})")
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
        return errors
