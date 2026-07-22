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
runs sequentially; pass a :class:`~concurrent.futures.Executor` (typically a
:class:`~concurrent.futures.ThreadPoolExecutor`) to parallelize. The caller owns
the pool's lifetime.
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import posixpath
import re
from abc import ABC, abstractmethod
from concurrent.futures import Executor, as_completed
from contextlib import closing
from dataclasses import dataclass
from functools import cached_property, partial
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

logger = logging.getLogger(__name__)

_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")


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
    """One manifest row from a :class:`Source` — a frozen, self-validating POD.

    Constructed inside a backend's ``_list_files`` (in-repo backends and downstream
    :class:`Source` subclasses alike). Validation runs at construction, so a
    malformed row fails the instant it is built — before any filesystem or network
    I/O — rather than mid-orchestration.

    Attributes:
        rel_path: Where the file lands under ``download_all``'s ``dest_dir``. Must
            use forward slashes; path semantics mirror ``pathlib.PurePosixPath``.
            Rejected at construction when absolute, containing backslashes, or
            escaping the destination directory after normalization.
        source_path: The source-side address as a plain string. HTTP-backed sources
            put the absolute URL here; fsspec-backed sources put the
            :class:`~upath.UPath` spec (which the backend re-instantiates as a
            ``UPath`` inside its :meth:`Source._pull`). Required — every real
            backend has somewhere to fetch from; test stubs that override
            ``_pull`` to write directly should pass a placeholder (the empty
            string is fine).
        sha256: Expected SHA-256 digest (hex; normalized to lowercase at
            construction, rejected if not 64 hex chars). Backends that can produce
            one (PhysioNet from ``SHA256SUMS.txt``, fsspec by hashing the source
            file, HTTP from explicit per-URL ``sha256:`` config) should set it —
            it's the only verifier the orchestrator trusts to skip a re-fetch.
            ``None`` means "no manifest-side hash"; the orchestrator will refuse
            to silently overwrite an existing dest in that case.

    Examples:
        Malformed rows fail at construction, not at fetch time:

        >>> RemoteFile("../escape.txt", "")
        Traceback (most recent call last):
            ...
        ValueError: rel_path '../escape.txt' escapes dest_dir (normalizes to '../escape.txt').
        >>> RemoteFile("/abs/path.txt", "")
        Traceback (most recent call last):
            ...
        ValueError: rel_path must be relative, got absolute: '/abs/path.txt'
        >>> RemoteFile("sub\\\\file.txt", "")
        Traceback (most recent call last):
            ...
        ValueError: rel_path 'sub\\\\file.txt' contains backslashes; use forward slashes.
        >>> RemoteFile("x.txt", "", sha256="abc123")
        Traceback (most recent call last):
            ...
        ValueError: sha256 must be 64 hex chars, got 'abc123'

        Uppercase digests are accepted and normalized to lowercase (the compare
        sites hash with :func:`hashlib.sha256`, which emits lowercase):

        >>> RemoteFile("x.txt", "", sha256="A" * 64).sha256 == "a" * 64
        True
    """

    rel_path: str
    source_path: str
    sha256: str | None = None

    def __post_init__(self):
        # rel_paths are documented as forward-slash posix paths. A backslash
        # would round-trip through ``Path(rel_path)`` differently on Windows
        # vs. POSIX, so the validation here (posixpath) would disagree with
        # ``Source._resolve_dest`` later (``Path``). Reject up-front.
        if "\\" in self.rel_path:
            raise ValueError(f"rel_path {self.rel_path!r} contains backslashes; use forward slashes.")
        if posixpath.isabs(self.rel_path):
            raise ValueError(f"rel_path must be relative, got absolute: {self.rel_path!r}")
        norm = posixpath.normpath(self.rel_path)
        if norm in (".", "..") or norm.startswith("../"):
            raise ValueError(f"rel_path {self.rel_path!r} escapes dest_dir (normalizes to {norm!r}).")
        if self.sha256 is not None:
            if not _SHA256_RE.fullmatch(self.sha256):
                raise ValueError(f"sha256 must be 64 hex chars, got {self.sha256!r}")
            object.__setattr__(self, "sha256", self.sha256.lower())

    @property
    def dest_key(self) -> str:
        """The posix-normalized ``rel_path`` — the collision key under a shared dest_dir.

        Two rows whose ``dest_key`` matches would race on the same ``.part`` file
        under concurrent workers, so both :attr:`Source.files` (within one source)
        and :func:`validate_unique_destinations` (across sources) reject them.

        Examples:
            >>> RemoteFile("sub/../a.txt", "").dest_key
            'a.txt'
        """
        return posixpath.normpath(self.rel_path)


class Source(ABC):
    """A place raw data comes from.

    Subclasses implement two private hooks:

    - :meth:`_list_files` — enumerate what files the source offers as
      :class:`RemoteFile` rows.
    - :meth:`_pull` — stream the bytes at one source address into a target path.

    The base class supplies the public surface — :meth:`download_all` for the
    bundle, :attr:`files` for the validated manifest — plus all the cross-cutting
    behavior every backend needs: ``.part`` staging, SHA-256 verification, atomic
    rename, path-traversal validation, duplicate-destination detection,
    include/exclude manifest filtering, and the sequential / parallel
    orchestration.

    Args:
        include: Optional list of :mod:`fnmatch`-style globs. When set, only
            manifest rows whose normalized ``rel_path`` matches at least one
            pattern are downloaded. ``None`` (default) selects everything.
        exclude: Optional list of :mod:`fnmatch`-style globs. Rows matching any
            pattern are dropped (applied after ``include``).

    Invariants subclasses must uphold:

    - :meth:`_list_files` is idempotent across calls — re-enumerating must produce
      the same set of :class:`RemoteFile` rows (in the same order when possible).
    - :meth:`_pull` writes the bytes at ``source_path`` into ``target`` and
      raises on any transport error. Backends with resume semantics (e.g. HTTP
      ``Range``) MAY inspect existing content at ``target`` and append;
      backends without resume should overwrite.
    - Subclasses that define ``__init__`` should call ``super().__init__(...)``
      to wire the ``include`` / ``exclude`` filters through.

    Concrete usage examples live on the methods that implement them:
    :meth:`download_all` (the public entry + orchestration policy),
    :attr:`files` (manifest validation + filtering), :meth:`_fetch_one` (the
    per-file pipeline: skip/overwrite/error policy + staging + verify + rename).
    """

    # Class-level fallbacks so subclasses that define ``__init__`` without calling
    # ``super().__init__`` still get well-defined (unfiltered) behavior.
    _include: list[str] | None = None
    _exclude: list[str] | None = None

    def __init__(self, include: list[str] | None = None, exclude: list[str] | None = None):
        self._include = list(include) if include else None
        self._exclude = list(exclude) if exclude else None

    def download_all(
        self,
        dest_dir: str | Path,
        *,
        pool: Executor | None = None,
        continue_on_error: bool = False,
        do_overwrite: bool = False,
    ) -> None:
        """Download every file this source lists into ``dest_dir``.

        Args:
            dest_dir: Where files land. Created if missing.
            pool: Optional :class:`~concurrent.futures.Executor` (typically a
                :class:`~concurrent.futures.ThreadPoolExecutor`) to submit work
                to. The caller owns the pool's lifetime. When ``None`` (default),
                the bundle is fetched sequentially in the calling thread — no
                thread pool is created. Pass a pool when you want parallelism,
                sized to whatever your transport tolerates.
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
            ValueError: When the manifest contains an unsafe rel_path (raised at
                :class:`RemoteFile` construction) or duplicate destinations
                (raised by :attr:`files`).

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

            Path-traversal manifests and absolute paths are rejected at
            :class:`RemoteFile` construction; duplicate destinations are rejected
            at :attr:`files` (the first thing ``download_all`` accesses) — see
            those docstrings for examples.
        """
        # Materialize + validate the manifest before touching the filesystem, so a
        # malformed ``sources:`` entry doesn't leave behind an empty ``dest_dir``.
        items = self.files
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Fetching {self.n_files} files to {dest_dir} ({'pooled' if pool else 'sequential'})")

        errors: list[Exception] = []
        # ``closing`` guarantees the generator's ``finally`` runs even when the loop
        # exits early via ``raise`` (fail-fast) — in pooled mode that ``finally`` is
        # what cancels the still-queued futures so "fail fast" actually stops the run.
        attempts = self._attempts(self._iter_attempts(items, dest_dir, do_overwrite), pool)
        with closing(attempts):
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
            raise ExceptionGroup(f"{len(errors)} of {self.n_files} files failed to download", errors)

    @cached_property
    def files(self) -> list[RemoteFile]:
        """The validated manifest — calls :meth:`_list_files` once, materializes, filters, and validates.

        Cached on first access. Subsequent ``download_all`` calls reuse the same
        list rather than re-hitting :meth:`_list_files` (which may do network
        I/O — e.g. PhysioNet fetches ``SHA256SUMS.txt``). If a source's contents
        could change between runs and the caller wants a fresh manifest, build
        a new ``Source`` instance.

        Per-row validation (relative, no traversal, no backslashes, well-formed
        sha256) happens at :class:`RemoteFile` construction inside
        :meth:`_list_files`, so it needs no re-checking here. This property adds
        the two whole-manifest steps:

        - **include / exclude filtering** — the constructor's glob patterns are
          matched against each row's normalized ``rel_path``; rows an ``include``
          list doesn't match, or an ``exclude`` list does match, are dropped.
        - **duplicate-destination detection** — two rows whose normalized
          rel_paths collide (``a/../x.csv`` vs ``x.csv``) would race on the same
          ``.part`` file under concurrent workers, so they fail the bundle
          up-front.

        ``_fetch_one`` calls :meth:`_resolve_dest` per-item at fetch time as the
        runtime security boundary (e.g. for dest_dirs that contain symlinks).

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
            ValueError: Duplicate destination 'sub/../a.txt': collides with 'a.txt'. ...

            Unsafe rel_paths fail earlier still — at :class:`RemoteFile`
            construction inside ``_list_files``:

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

            ``include`` / ``exclude`` globs subset the manifest:

            >>> class TreeSource(Source):
            ...     def _list_files(self):
            ...         return [
            ...             RemoteFile("hosp/patients.csv.gz", ""),
            ...             RemoteFile("hosp/labevents.csv.gz", ""),
            ...             RemoteFile("note/discharge.csv.gz", ""),
            ...         ]
            ...     def _pull(self, source_path, target):
            ...         target.write_text("ok")
            >>>
            >>> [f.rel_path for f in TreeSource(include=["hosp/*"]).files]
            ['hosp/patients.csv.gz', 'hosp/labevents.csv.gz']
            >>> [f.rel_path for f in TreeSource(exclude=["*/labevents*"]).files]
            ['hosp/patients.csv.gz', 'note/discharge.csv.gz']
        """
        items = list(self._list_files())
        if self._include is not None or self._exclude is not None:
            kept = [item for item in items if self._selected(item)]
            logger.info(f"Manifest filters selected {len(kept)}/{len(items)} files")
            items = kept
        seen: dict[str, RemoteFile] = {}
        for item in items:
            if item.dest_key in seen:
                raise ValueError(
                    f"Duplicate destination {item.rel_path!r}: collides with "
                    f"{seen[item.dest_key].rel_path!r}. Each item from a source's "
                    "_list_files() must resolve to a unique rel_path."
                )
            seen[item.dest_key] = item
        return items

    @property
    def n_files(self) -> int:
        """Number of files in the validated, filtered manifest."""
        return len(self.files)

    def _selected(self, item: RemoteFile) -> bool:
        """Apply the constructor's ``include`` / ``exclude`` globs to one manifest row."""
        return self._selected_path(item.dest_key)

    def _selected_path(self, dest_key: str) -> bool:
        """String-level filter check, for backends that want to skip expensive per-file work (e.g. hashing) on
        rows the manifest filters would drop anyway."""
        if self._include is not None and not any(fnmatch.fnmatchcase(dest_key, p) for p in self._include):
            return False
        return not (
            self._exclude is not None and any(fnmatch.fnmatchcase(dest_key, p) for p in self._exclude)
        )

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

        :class:`RemoteFile` construction already rejects malformed rel_paths by
        string inspection; this fetch-time check is the runtime security boundary
        against escapes that only materialize on a real filesystem (e.g. symlinks
        inside ``dest_dir``).
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

        3. If a prior run left a ``.part`` that already verifies against
           ``item.sha256`` (interrupted between the last byte and the rename),
           promote it to ``dest`` directly — no re-fetch.
        4. If the manifest has no SHA to verify against, discard any stale
           ``.part`` — resume-without-verification is unsafe.
        5. Call ``self._pull(item.source_path, part)`` — backend streams bytes.
        6. If ``item.sha256`` is set, hash ``part`` once via :func:`sha256_of`
           and compare; on mismatch, unlink ``part`` and raise
           :class:`ChecksumError`.
        7. Atomic-rename ``part`` → ``dest``.

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

        if item.sha256 is not None:
            # A prior run may have died between writing the last byte and the
            # rename below — in that case the ``.part`` is the complete file and
            # re-fetching it (or bouncing off an unsatisfiable Range request)
            # wastes the whole transfer. Verify and promote directly.
            if part.exists() and sha256_of(part) == item.sha256:
                logger.debug(f"Promoting complete .part for {item.rel_path} without re-fetching.")
                part.replace(dest)
                return
        elif part.exists():
            # Resume-without-verification is unsafe: without a sha to catch silent
            # corruption, a stale ``.part`` could be from a different version of
            # the source file. Clear it so ``_pull`` starts fresh. With sha set,
            # Range-resume is safe because the post-write verify catches mismatches.
            part.unlink()

        self._pull(item.source_path, part)

        # Hash once, compare once — the failure message reuses the digest, so
        # ``_verifies`` (which would re-hash) is deliberately not used here.
        if item.sha256 is not None:
            actual = sha256_of(part)
            if actual != item.sha256:
                part.unlink()
                raise ChecksumError(item.source_path, item.sha256, actual)
        part.replace(dest)

    def _iter_attempts(
        self, items: list[RemoteFile], dest_dir: Path, do_overwrite: bool
    ) -> Iterator[tuple[RemoteFile, Callable[[], None]]]:
        """Pair each manifest row with the zero-arg thunk that fetches it.

        The thunks close over everything :meth:`_fetch_one` needs, so the
        dispatch layer (:meth:`_attempts`) can treat sequential and pooled
        execution identically — it invokes (or submits) opaque callables and
        never needs the fetch arguments itself.
        """
        for item in items:
            yield item, partial(self._fetch_one, item, dest_dir, do_overwrite)

    @staticmethod
    def _attempts(
        items_to_fetch: Iterable[tuple[RemoteFile, Callable[[], None]]],
        pool: Executor | None,
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


def validate_unique_destinations(sources: Iterable[Source]) -> None:
    """Reject destination collisions across multiple sources sharing one ``dest_dir``.

    :attr:`Source.files` already rejects collisions *within* one source, but the
    CLI (and any caller composing sources) stages several sources into one shared
    directory, where two sources legally listing the same ``rel_path`` would race
    on the same ``.part`` file under concurrent workers — or serially clobber /
    ``FileExistsError`` on each other. Calling this before any fetch turns that
    late, confusing failure into an immediate, precise config error.

    Accessing each source's :attr:`~Source.files` materializes its manifest
    (cached, so the later ``download_all`` calls reuse it rather than re-listing).

    Examples:
        >>> class A(Source):
        ...     def _list_files(self):
        ...         return [RemoteFile("x.csv", "")]
        ...     def _pull(self, source_path, target):
        ...         target.write_text("A")
        >>> class B(Source):
        ...     def _list_files(self):
        ...         return [RemoteFile("sub/../x.csv", "")]
        ...     def _pull(self, source_path, target):
        ...         target.write_text("B")
        >>> validate_unique_destinations([A(), B()])
        Traceback (most recent call last):
            ...
        ValueError: Duplicate destination across sources: 'sub/../x.csv' from B collides with 'x.csv' from A.

        Distinct destinations pass silently:

        >>> class C(Source):
        ...     def _list_files(self):
        ...         return [RemoteFile("y.csv", "")]
        ...     def _pull(self, source_path, target):
        ...         target.write_text("C")
        >>> validate_unique_destinations([A(), C()])
    """
    seen: dict[str, tuple[str, RemoteFile]] = {}
    for source in sources:
        name = type(source).__name__
        for item in source.files:
            if item.dest_key in seen:
                prior_name, prior_item = seen[item.dest_key]
                raise ValueError(
                    f"Duplicate destination across sources: {item.rel_path!r} from {name} "
                    f"collides with {prior_item.rel_path!r} from {prior_name}."
                )
            seen[item.dest_key] = (name, item)
