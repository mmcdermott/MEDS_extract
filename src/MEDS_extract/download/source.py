"""The :class:`Source` ABC, its manifest type :class:`RemoteFile`, and the orchestration loop.

A :class:`Source` is anywhere raw data comes from — a PhysioNet dataset release, an
explicit list of HTTP URLs, an S3 / GCS / local-filesystem tree. Concrete sources
inherit from this ABC and implement two methods:

- :meth:`Source._list_files` — enumerate what files the source offers (the
  validating wrapper :attr:`Source.files` is what callers use).
- :meth:`Source._pull` — stream the bytes at one source address into a target
  path. The base class wraps this in :meth:`Source._fetch_one`, which owns
  the full per-file pipeline: the skip / overwrite / error policy on any
  pre-existing dest, ``.part`` staging, SHA-256 verification, and atomic
  rename.

The single public fetch entry point is :meth:`Source.download_all`. By default it
runs sequentially; pass a :class:`ThreadPoolExecutor` to parallelize. The caller
owns the pool's lifetime.
"""

from __future__ import annotations

import hashlib
import logging
import posixpath
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from functools import cached_property, partial
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

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
        source_path: The source-side address as a plain string. HTTP-backed sources
            put the absolute URL here; fsspec-backed sources put the
            :class:`~upath.UPath` spec (which the backend re-instantiates as a
            ``UPath`` inside its :meth:`Source._pull`). Required — every real
            backend has somewhere to fetch from; test stubs that override
            ``_pull`` to write directly should pass a placeholder (the empty
            string is fine).
        sha256: Expected SHA-256 digest (lowercase hex). Backends that can produce
            one (PhysioNet from ``SHA256SUMS.txt``, fsspec by hashing the source
            file, HTTP from explicit per-URL ``sha256:`` config) should set it —
            it's the only verifier the orchestrator trusts to skip a re-fetch.
            ``None`` means "no manifest-side hash"; the orchestrator will refuse
            to silently overwrite an existing dest in that case.
    """

    rel_path: str
    source_path: str
    sha256: str | None = None


class Source(ABC):
    """A place raw data comes from.

    Subclasses implement two private hooks:

    - :meth:`_list_files` — enumerate what files the source offers as
      :class:`RemoteFile` rows.
    - :meth:`_pull` — stream the bytes at one source address into a target path.

    The base class supplies the public surface — :meth:`download_all` for the
    bundle, :attr:`files` for the validated manifest — plus all the cross-cutting
    behavior every backend needs: ``.part`` staging, SHA-256 verification, atomic
    rename, path-traversal validation, duplicate-destination detection, and the
    sequential / parallel orchestration.

    Invariants subclasses must uphold:

    - :meth:`_list_files` is idempotent across calls — re-enumerating must produce
      the same set of :class:`RemoteFile` rows (in the same order when possible).
    - :meth:`_pull` writes the bytes at ``source_path`` into ``target`` and
      raises on any transport error. Backends with resume semantics (e.g. HTTP
      ``Range``) MAY inspect existing content at ``target`` and append;
      backends without resume should overwrite.

    Concrete usage examples live on the methods that implement them:
    :meth:`download_all` (the public entry + orchestration policy),
    :attr:`files` (manifest validation), :meth:`_fetch_one` (the per-file
    pipeline: skip/overwrite/error policy + staging + verify + rename).
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
                duplicate destinations (raised by :attr:`files`).

        Examples:
            Simple case — no pool passed, ``download_all`` runs sequentially:

            >>> class StubSource(Source):
            ...     def _list_files(self):
            ...         return [RemoteFile("a.txt", ""), RemoteFile("sub/b.txt", "")]
            ...     def _pull(self, source_path, target):
            ...         target.write_text(f"contents of {target.name}")
            >>>
            >>> with tempfile.TemporaryDirectory() as d:
            ...     d = Path(d)
            ...     StubSource().download_all(d)
            ...     print_directory(d)
            ├── a.txt
            └── sub
                └── b.txt

            Multi-source case — caller owns one :class:`ThreadPoolExecutor` and
            hands it to every source. Two sources writing distinct files into one
            ``dest_dir`` is the typical CLI pattern (one ``physionet`` source plus
            one ``http`` source for a metadata bundle):

            >>> from concurrent.futures import ThreadPoolExecutor
            >>>
            >>> class SourceA(Source):
            ...     def _list_files(self):
            ...         return [RemoteFile("a.txt", "")]
            ...     def _pull(self, source_path, target):
            ...         target.write_text("from A")
            >>>
            >>> class SourceB(Source):
            ...     def _list_files(self):
            ...         return [RemoteFile("metadata/b.csv", "")]
            ...     def _pull(self, source_path, target):
            ...         target.write_text("from B")
            >>>
            >>> with tempfile.TemporaryDirectory() as d, ThreadPoolExecutor(max_workers=4) as pool:
            ...     d = Path(d)
            ...     for src in [SourceA(), SourceB()]:
            ...         src.download_all(d, pool=pool)
            ...     print_directory(d)
            ├── a.txt
            └── metadata
                └── b.csv

            Already-complete files are skipped — ``_pull`` is not invoked for any
            :class:`RemoteFile` whose on-disk copy verifies against the manifest's
            ``sha256``:

            >>> import hashlib
            >>> body = b"abc"
            >>> digest = hashlib.sha256(body).hexdigest()
            >>>
            >>> class SkipSource(Source):
            ...     def _list_files(self):
            ...         return [RemoteFile("x.txt", "", sha256=digest)]
            ...     def _pull(self, source_path, target):
            ...         raise RuntimeError("must not be called — file is already complete")
            >>>
            >>> with tempfile.TemporaryDirectory() as d:
            ...     d = Path(d)
            ...     _ = (d / "x.txt").write_bytes(body)
            ...     SkipSource().download_all(d)  # no exception → already-complete skip worked

            An existing ``dest`` that **doesn't** verify (sha mismatch, or no
            manifest sha at all) is a hard error rather than a silent overwrite.
            The user has to opt in to overwriting via ``do_overwrite=True``:

            >>> class UnverifiableSource(Source):
            ...     def _list_files(self):
            ...         return [RemoteFile("x.txt", "")]  # no sha
            ...     def _pull(self, source_path, target):
            ...         target.write_text("fresh")
            >>>
            >>> with tempfile.TemporaryDirectory() as d:
            ...     d = Path(d)
            ...     _ = (d / "x.txt").write_bytes(b"stale")
            ...     UnverifiableSource().download_all(d)
            Traceback (most recent call last):
                ...
            FileExistsError: Refusing to overwrite ...x.txt: ... do_overwrite=True ...

            Path-traversal manifests, duplicate destinations, and absolute
            paths are rejected at :attr:`files` (the first thing
            ``download_all`` accesses) — see that property's docstring for
            examples.
        """
        # Validate the manifest before touching the filesystem, so a malformed
        # ``sources:`` entry doesn't leave behind an empty ``dest_dir``.
        items = self.files
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Fetching {len(items)} files to {dest_dir} (pool={pool!r})")

        items_to_fetch = ((item, partial(self._fetch_one, item, dest_dir, do_overwrite)) for item in items)

        errors: list[Exception] = []
        # ``closing`` guarantees the generator's ``finally`` runs even when the loop
        # exits early via ``raise`` (fail-fast) — in pooled mode that ``finally`` is
        # what cancels the still-queued futures so "fail fast" actually stops the run.
        with closing(self._attempts(items_to_fetch, pool)) as attempts:
            for item, run in attempts:
                try:
                    run()
                except Exception as e:
                    # Tag the exception with the item it came from so a caller
                    # inspecting the ExceptionGroup (or a bare re-raise) can tell
                    # which file failed without cross-referencing logs.
                    e.add_note(f"while fetching {item.rel_path!r} from {item.source_path!r}")
                    if not continue_on_error:
                        raise
                    logger.exception(f"Failed to fetch {item.rel_path}")
                    errors.append(e)
        if errors:
            raise ExceptionGroup(f"{len(errors)} of {len(items)} files failed to download", errors)

    @cached_property
    def files(self) -> list[RemoteFile]:
        """The validated manifest — calls :meth:`_list_files` once, materializes, and validates every row.

        Cached on first access. Subsequent ``download_all`` calls reuse the same
        list rather than re-hitting :meth:`_list_files` (which may do network
        I/O — e.g. PhysioNet fetches ``SHA256SUMS.txt``). If a source's contents
        could change between runs and the caller wants a fresh manifest, build
        a new ``Source`` instance.

        Validation runs over the whole manifest before any I/O, so a malformed
        row fails the bundle up-front rather than mid-orchestration:

        - **rel_path must be relative and not escape its dest_dir.** Both
          checks use the *normalized* posix path. ``a/../x.csv`` and ``x.csv``
          normalize to the same key (caught as a collision); ``../etc/passwd``
          normalizes to a path starting with ``..`` (caught as an escape).
          ``_fetch_one`` calls :meth:`_resolve_dest` per-item as a defense-in-
          depth secondary check (e.g. for dest_dirs that contain symlinks),
          but the typical bad-manifest case never reaches there.
        - **rel_paths must be unique** after normalization — two rows
          resolving to the same dest would race on the same ``.part`` file
          under concurrent workers.

        Examples:
            Duplicate destinations are caught even when the strings differ:

            >>> class DupSource(Source):
            ...     def _list_files(self):
            ...         return [RemoteFile("a.txt", ""), RemoteFile("sub/../a.txt", "")]
            ...     def _pull(self, source_path, target):
            ...         target.write_text("never reached")
            >>>
            >>> DupSource().files
            Traceback (most recent call last):
                ...
            ValueError: Duplicate destination ...

            Path-traversal escapes fail here before any fetch is attempted:

            >>> class EscapingSource(Source):
            ...     def _list_files(self):
            ...         return [RemoteFile("../escape.txt", "")]
            ...     def _pull(self, source_path, target):
            ...         target.write_text("never reached")
            >>>
            >>> EscapingSource().files
            Traceback (most recent call last):
                ...
            ValueError: rel_path '../escape.txt' escapes dest_dir ...

            Backslashes are rejected — the documented contract is posix
            forward-slash paths, and a backslash would round-trip differently
            under ``Path`` on Windows vs. POSIX:

            >>> class BackslashSource(Source):
            ...     def _list_files(self):
            ...         return [RemoteFile("sub\\\\file.txt", "")]
            ...     def _pull(self, source_path, target):
            ...         target.write_text("never reached")
            >>>
            >>> BackslashSource().files
            Traceback (most recent call last):
                ...
            ValueError: rel_path 'sub\\\\file.txt' contains backslashes; use forward slashes.
        """
        items = list(self._list_files())
        seen: dict[str, RemoteFile] = {}
        for item in items:
            # rel_paths are documented as forward-slash posix paths. A backslash
            # would round-trip through ``Path(rel_path)`` differently on Windows
            # vs. POSIX, so the validation here (posixpath) would disagree with
            # ``_resolve_dest`` later (``Path``). Reject up-front.
            if "\\" in item.rel_path:
                raise ValueError(f"rel_path {item.rel_path!r} contains backslashes; use forward slashes.")
            if posixpath.isabs(item.rel_path):
                raise ValueError(f"rel_path must be relative, got absolute: {item.rel_path!r}")
            key = posixpath.normpath(item.rel_path)
            if key == ".." or key.startswith("../"):
                raise ValueError(f"rel_path {item.rel_path!r} escapes dest_dir (normalizes to {key!r}).")
            if key in seen:
                raise ValueError(
                    f"Duplicate destination {item.rel_path!r}: collides with "
                    f"{seen[key].rel_path!r}. Each item from a source's "
                    "_list_files() must resolve to a unique rel_path."
                )
            seen[key] = item
        return items

    @abstractmethod
    def _list_files(self) -> Iterable[RemoteFile]:
        """Subclass hook — enumerate the files this source offers.

        :attr:`files` is the validating cached wrapper that callers use; this
        hook just produces the rows.
        """

    @abstractmethod
    def _pull(self, source_path: str, target: Path) -> None:
        """Stream the bytes at ``source_path`` into ``target``.

        ``source_path`` is whatever the backend stored in
        ``RemoteFile.source_path`` when it built the manifest (a URL for HTTP,
        a UPath spec for fsspec). On successful return ``target`` contains
        the complete file; on any transport error, raise.

        Backends with resume semantics (HTTP ``Range``) MAY observe existing
        bytes at ``target`` and append; backends without resume should
        overwrite.
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
        """True iff ``dest`` exists AND the manifest's ``sha256`` matches.

        SHA-256 is the only verifier we trust. Same-size files can have different
        content; existence-with-no-hash means the file on disk could be anything.
        Backends that want skip-on-rerun semantics must populate ``sha256``.
        """
        return item.sha256 is not None and dest.exists() and sha256_of(dest) == item.sha256

    def _fetch_one(self, item: RemoteFile, dest_dir: Path, do_overwrite: bool) -> None:
        """Fetch one manifest entry end-to-end: policy → ``.part`` staging → verify → rename.

        Pipeline:

        1. Resolve ``dest = dest_dir / item.rel_path`` (with traversal validation).
        2. Apply the skip / overwrite / error policy on any pre-existing ``dest``:

           - ``do_overwrite=True``: clear ``dest`` and any stale ``.part``, proceed.
           - ``dest`` verifies against ``item.sha256``: skip and return.
           - otherwise: raise :class:`FileExistsError` — refuse to silently
             overwrite a file we can't prove matches the manifest.

        3. If the manifest has no SHA to verify against, discard any stale
           ``.part`` — resume-without-verification is unsafe.
        4. Call ``self._pull(item.source_path, part)`` — backend streams bytes.
        5. If ``item.sha256`` is set, verify ``part`` via :meth:`_verifies`;
           on mismatch, unlink ``part`` and raise :class:`ChecksumError`.
        6. Atomic-rename ``part`` → ``dest``.

        On any exception, ``dest`` does not exist. ``part`` may exist after a
        partial transport failure (intentional — gives a future run a head
        start via Range-resume on backends that support it).

        Examples:
            Backend's ``_pull`` produces the bytes; this method handles staging,
            sha verification, and atomic rename:

            >>> import hashlib
            >>> class FakeSource(Source):
            ...     def _list_files(self):
            ...         return [RemoteFile("x.txt", "dummy", sha256=hashlib.sha256(b"hi").hexdigest())]
            ...     def _pull(self, source_path, target):
            ...         target.write_bytes(b"hi")
            >>> with tempfile.TemporaryDirectory() as d:
            ...     d = Path(d)
            ...     src = FakeSource()
            ...     [item] = src.files
            ...     src._fetch_one(item, d, do_overwrite=False)
            ...     print((d / "x.txt").read_bytes(), (d / "x.txt.part").exists())
            b'hi' False

            On a SHA mismatch the staged ``.part`` is deleted, ``dest`` is not
            created, and :class:`ChecksumError` propagates:

            >>> class WrongShaSource(Source):
            ...     def _list_files(self):
            ...         return [RemoteFile("x.txt", "dummy", sha256="0" * 64)]
            ...     def _pull(self, source_path, target):
            ...         target.write_bytes(b"hi")
            >>> with tempfile.TemporaryDirectory() as d:
            ...     d = Path(d)
            ...     src = WrongShaSource()
            ...     [item] = src.files
            ...     try:
            ...         src._fetch_one(item, d, do_overwrite=False)
            ...     except ChecksumError:
            ...         print(f"raised; dest={(d / 'x.txt').exists()}, part={(d / 'x.txt.part').exists()}")
            raised; dest=False, part=False
        """
        dest = self._resolve_dest(dest_dir, item.rel_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")

        if do_overwrite:
            # Clear both independently — a ``.part`` from a half-finished prior
            # run can exist even when ``dest`` doesn't, and either one left in
            # place would be picked up as a Range-resume base.
            if dest.exists():
                dest.unlink()
            if part.exists():
                part.unlink()
        elif dest.exists():
            if self._verifies(dest, item):
                logger.debug(f"Skipping {item.rel_path}: already complete.")
                return
            raise FileExistsError(
                f"Refusing to overwrite {dest}: existing file does not verify against "
                f"the manifest (sha mismatch, or no manifest sha provided). Pass "
                f"do_overwrite=True to force a refetch, or delete the file first."
            )

        # Resume-without-verification is unsafe: without a sha to catch silent
        # corruption, a stale ``.part`` could be from a different version of
        # the source file. Clear it so ``_pull`` starts fresh. With sha set,
        # Range-resume is safe because the post-write verify catches mismatches.
        if item.sha256 is None and part.exists():
            part.unlink()

        self._pull(item.source_path, part)

        # Hash once, compare once: ``_verifies`` would re-hash on the success
        # path AND we'd re-hash for the error message on mismatch.
        if item.sha256 is not None:
            actual = sha256_of(part)
            if actual != item.sha256:
                part.unlink()
                raise ChecksumError(item.source_path, item.sha256, actual)
        part.replace(dest)

    @staticmethod
    def _attempts(
        items_to_fetch: Iterable[tuple[RemoteFile, Callable[[], None]]],
        pool: ThreadPoolExecutor | None,
    ) -> Iterator[tuple[RemoteFile, Callable[[], None]]]:
        """Dispatch ``(item, callable)`` pairs sequentially or through a pool.

        Sequential mode yields the input pairs unchanged; the caller invokes
        them in the main thread. Parallel mode submits every callable to
        ``pool`` up front and yields ``(item, future.result)`` pairs in
        completion order.

        Fail-fast in parallel mode: if the caller raises out of its loop on
        the first failure, the ``finally`` cancels every still-queued future
        so the rest of the bundle halts immediately. Already-running and
        already-done futures are unaffected (``cancel`` is a no-op on those).
        The caller must wrap the generator in :func:`contextlib.closing` to
        guarantee the ``finally`` runs.
        """
        if pool is None:
            yield from items_to_fetch
            return
        futures = {pool.submit(run): item for item, run in items_to_fetch}
        try:
            for fut in as_completed(futures):
                yield futures[fut], fut.result
        finally:
            for fut in futures:
                fut.cancel()
