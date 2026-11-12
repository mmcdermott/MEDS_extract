"""The :class:`Source` ABC, its manifest type :class:`RemoteFile`, and the orchestration loop.

A :class:`Source` is anywhere raw data comes from — a PhysioNet dataset release, an
explicit list of HTTP URLs, an S3 / GCS / local-filesystem tree. Concrete sources
inherit from this ABC and implement two methods:

- :meth:`Source._list_files` — enumerate what files the source offers (the
  validating wrapper :attr:`Source.files` is what callers use).
- :meth:`Source._pull` — stream the bytes at one source address into a target
  path. The base class wraps this in :meth:`Source._fetch_one`, which owns
  the full per-file pipeline: the skip / re-fetch policy on any pre-existing
  dest, ``.part`` staging, SHA-256 verification, and atomic rename.

The single public fetch entry point is :meth:`Source.download_all`. By default it
runs sequentially; pass a :class:`~concurrent.futures.Executor` (typically a
:class:`~concurrent.futures.ThreadPoolExecutor`) to parallelize. The caller owns
the pool's lifetime.

Extraction completion markers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a :class:`RemoteFile` with ``unarchive`` set, the archive's own SHA-256 says
nothing about whether its *members* are on disk: an extraction can die part-way
(ENOSPC, OOM, Ctrl-C) after the verified archive was renamed into place, and
``cleanup_archive`` deletes the archive — the only thing the plain skip check looks
at — after a successful one. So :meth:`Source._maybe_unarchive` writes a small
hidden JSON *completion marker* next to the archive (see
:func:`extraction_marker_path`) only after every member has been moved into place,
and :meth:`Source._fetch_one` treats a marker matching the manifest row as the skip
witness for that row:

- marker matches and the archive was meant to be cleaned up → skip, even though
  the archive is gone (no re-download on re-run);
- marker matches and the archive is kept → skip if the archive also verifies (it
  is part of the result), else re-fetch as before;
- archive verifies but the marker is absent or describes a different extraction
  (other sha, format, or ``archive_subdir``) → re-extract from the local archive,
  without re-downloading it.

Extraction happens in a scratch directory beside the archive (same filesystem, so
moving each member into place is a cheap atomic rename), which is removed on any
failure; a half-finished extraction therefore never produces a marker.

Marker format (``marker_version`` 1)::

    {
      "marker_version": 1,
      "archive": "bundle.zip",          # the archive's file name
      "archive_sha256": "<hex>|null",   # manifest sha; null => never a skip witness
      "format": "zip",                  # canonical ArchiveFormat value
      "archive_subdir": "data|null",
      "members": ["a.csv", "sub/b.csv"] # extracted non-directory paths, relative
    }                                   # to the archive's directory, all must exist
"""

from __future__ import annotations

import fnmatch
import hashlib
import itertools
import json
import logging
import os
import posixpath
import re
import shutil
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, Executor, wait
from contextlib import closing
from dataclasses import dataclass
from functools import cached_property, partial
from pathlib import Path
from typing import TYPE_CHECKING

from ._paths import resolve_contained
from .unarchive import ArchiveFormat, resolve_format, safe_extract

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

logger = logging.getLogger(__name__)

_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")

# Pooled dispatch keeps at most this many futures in flight (submitted but not yet
# consumed). Each pending Future costs ~2 KB of bookkeeping (its Condition/RLock plus
# dict slots), so submitting a 377k-row manifest (MIMIC-CXR-JPG scale) up-front would
# hold ~780 MB for the whole run; a bounded sliding window keeps that O(window).
_MAX_PENDING_SUBMITS = 1024


class _AbortedError(Exception):
    """Raised by a queued fetch that started after fail-fast tripped.

    Never surfaces: fail-fast means the consumer has already raised, so these results
    are discarded. It exists so an aborted fetch is distinguishable from a real
    transport failure if one is ever inspected.
    """


# ``download_all`` emits an INFO progress line at most this often during the fetch
# loop, so multi-hour transfers are observable at the default log level without
# per-file chatter.
_PROGRESS_INTERVAL_S = 30.0

# Bumped whenever the marker's schema changes; a marker with another version is
# treated as absent, which costs one re-extraction and nothing else.
_MARKER_VERSION = 1


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


def extraction_marker_path(archive: Path) -> Path:
    """The completion marker for ``archive``: a hidden ``.<name>.extracted.json`` sibling.

    A sibling (rather than one shared state file under ``dest_dir``) keeps every
    archive's witness independent, so concurrent workers never contend on it, and
    keeps it next to the members it describes. The leading dot keeps it out of
    ``ls`` and shell globs.

    Examples:
        >>> extraction_marker_path(Path("raw/AUMCdb.zip")).as_posix()
        'raw/.AUMCdb.zip.extracted.json'
    """
    return archive.with_name(f".{archive.name}.extracted.json")


def _extraction_scratch_path(archive: Path) -> Path:
    """Scratch directory an archive is extracted into before its members are moved into place.

    Deterministic (not :func:`tempfile.mkdtemp`) so the next run finds and removes a
    directory orphaned by a SIGKILL, which no ``finally`` can clean up.

    Examples:
        >>> _extraction_scratch_path(Path("raw/AUMCdb.zip")).as_posix()
        'raw/.AUMCdb.zip.extract-tmp'
    """
    return archive.with_name(f".{archive.name}.extract-tmp")


def _move_tree(src_root: Path, dst_root: Path) -> list[str]:
    """Move everything under ``src_root`` into ``dst_root``, merging with existing directories.

    Each non-directory entry (files and symlinks; symlinks are moved, never
    followed) is moved with :func:`os.replace` — atomic per entry on one filesystem,
    and it overwrites an existing file at the destination, which is what a
    re-extraction wants. Directories are created as needed, so empty directories in
    the archive survive. Every destination is checked to resolve inside
    ``dst_root`` first: extraction validated paths against the clean scratch
    directory, so a pre-existing symlink in ``dst_root`` is the one escape left to
    rule out here.

    Returns:
        The sorted POSIX paths, relative to ``dst_root``, of the moved non-directory
        entries.

    Examples:
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     (d / "src" / "sub").mkdir(parents=True)
        ...     (d / "src" / "empty").mkdir()
        ...     _ = (d / "src" / "a.csv").write_text("new")
        ...     _ = (d / "src" / "sub" / "b.csv").write_text("b")
        ...     (d / "dst" / "sub").mkdir(parents=True)
        ...     _ = (d / "dst" / "a.csv").write_text("old")
        ...     _ = (d / "dst" / "sub" / "keep.csv").write_text("k")
        ...     print(_move_tree(d / "src", d / "dst"))
        ...     print((d / "dst" / "a.csv").read_text())
        ...     print_directory(d / "dst")
        ['a.csv', 'sub/b.csv']
        new
        ├── a.csv
        ├── empty
        └── sub
            ├── b.csv
            └── keep.csv
    """
    moved: list[str] = []
    for root_str, dirnames, filenames in os.walk(src_root):
        root = Path(root_str)
        rel_dir = root.relative_to(src_root)
        # A symlink to a directory shows up in ``dirnames``; move it as a link instead
        # of descending into (and so re-creating) whatever it points at.
        links = [name for name in dirnames if (root / name).is_symlink()]
        dirnames[:] = sorted(name for name in dirnames if name not in links)
        for name in [*dirnames, *sorted(filenames + links)]:
            rel = (rel_dir / name).as_posix()
            target, contained = resolve_contained(dst_root, rel, posix_member=True)
            if not contained:
                raise ValueError(
                    f"Refusing to move extracted entry {rel!r} into {dst_root}: it would land at "
                    f"{target}, outside the destination (is there a symlink in the way?)."
                )
            if name in dirnames:
                (dst_root / rel_dir / name).mkdir(parents=True, exist_ok=True)
            else:
                os.replace(root / name, dst_root / rel_dir / name)
                moved.append(rel)
    return sorted(moved)


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
            ``None`` means "no manifest-side hash"; an existing dest can then
            never be proven complete, so the orchestrator re-fetches it on every
            run rather than trusting it.
        unarchive: Optional post-fetch unpack format. ``None`` (default) means no
            unpack. ``"zip"``, ``"tar"``, ``"tar.gz"`` / ``"tgz"`` dispatch to the
            matching :class:`~MEDS_extract.download.unarchive.ArchiveFormat`;
            ``"auto"`` infers the format from ``rel_path``'s extension — useful
            when a single source lists both archive and non-archive files, since
            ``"auto"`` is a no-op on anything that doesn't end in a recognized
            archive extension. Validated at construction against
            :class:`~MEDS_extract.download.unarchive.ArchiveFormat`.
        cleanup_archive: Tri-state controlling whether the source archive file is
            removed after a successful extraction. ``None`` (default) means "use
            the mode-implied default": ``"auto"`` removes the archive (the one-arg
            "fetch + extract + cleanup" flow); explicit formats keep it. Set
            ``True`` to force cleanup, ``False`` to force keep, regardless of
            mode. Has no effect when ``unarchive`` is ``None``.
        archive_subdir: Optional archive-relative directory to extract on its own.
            Only members under it are extracted, written relative to it — so
            ``"1_omop_data_csv"`` turns a PhysioNet ``get-zip`` bundle's
            ``<project>-<version>/1_omop_data_csv/person.csv`` into
            ``person.csv`` next to the archive, and skips the bundle's other
            folders. It matches the shallowest run of whole path components equal
            to it, under any (single, shared) leading prefix; zero matches is an
            error at extraction time. See
            :mod:`~MEDS_extract.download.unarchive` for the full rule. Validated at
            construction: forward slashes, relative, no ``..``; normalized (``./``
            and trailing slashes dropped). Requires ``unarchive``; ignored for rows
            where ``"auto"`` finds no archive extension.

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
        >>> RemoteFile("sub/..", "")
        Traceback (most recent call last):
            ...
        ValueError: rel_path 'sub/..' escapes dest_dir (normalizes to '.').
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

        ``unarchive`` is validated at construction too — a typo'd format fails
        before any I/O rather than surfacing mid-download:

        >>> RemoteFile("x.rar", "", unarchive="rar")
        Traceback (most recent call last):
            ...
        ValueError: 'rar' is not a valid ArchiveFormat
        >>> RemoteFile("bundle.zip", "", unarchive="zip", cleanup_archive=True).unarchive
        'zip'

        ``archive_subdir`` is validated like ``rel_path`` and normalized:

        >>> RemoteFile("b.zip", "", unarchive="zip", archive_subdir="./proj/data/").archive_subdir
        'proj/data'
        >>> RemoteFile("b.zip", "", unarchive="zip", archive_subdir="../data")
        Traceback (most recent call last):
            ...
        ValueError: archive_subdir '../data' must name a directory inside the archive (relative, non-empty,
                    no '..' components).
        >>> RemoteFile("b.zip", "", unarchive="zip", archive_subdir="/data")
        Traceback (most recent call last):
            ...
        ValueError: archive_subdir must be relative, got absolute: '/data'
        >>> RemoteFile("b.zip", "", unarchive="zip", archive_subdir="a\\\\b")
        Traceback (most recent call last):
            ...
        ValueError: archive_subdir 'a\\\\b' contains backslashes; use forward slashes.

        ...and is meaningless without ``unarchive``:

        >>> RemoteFile("b.zip", "", archive_subdir="data")
        Traceback (most recent call last):
            ...
        ValueError: archive_subdir='data' requires unarchive to be set (rel_path 'b.zip').
    """

    rel_path: str
    source_path: str
    sha256: str | None = None
    unarchive: str | None = None
    cleanup_archive: bool | None = None
    archive_subdir: str | None = None

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
        if self.unarchive is not None:
            ArchiveFormat(self.unarchive)  # raises ValueError on unknown tokens
        if self.archive_subdir is not None:
            object.__setattr__(self, "archive_subdir", self._normalize_archive_subdir())

    def _normalize_archive_subdir(self) -> str:
        """Validate ``archive_subdir`` and return its normalized form (see the class docstring)."""
        sub = self.archive_subdir
        if self.unarchive is None:
            raise ValueError(
                f"archive_subdir={sub!r} requires unarchive to be set (rel_path {self.rel_path!r})."
            )
        if "\\" in sub:
            raise ValueError(f"archive_subdir {sub!r} contains backslashes; use forward slashes.")
        if posixpath.isabs(sub):
            raise ValueError(f"archive_subdir must be relative, got absolute: {sub!r}")
        # Stricter than rel_path: ``..`` is rejected anywhere, not just when the path
        # escapes, because the matcher compares literal components.
        parts = [part for part in sub.split("/") if part not in ("", ".")]
        if not parts or ".." in parts:
            raise ValueError(
                f"archive_subdir {sub!r} must name a directory inside the archive (relative, non-empty, "
                "no '..' components)."
            )
        return "/".join(parts)

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
            pattern are downloaded. ``None`` (default) selects everything; an
            empty list matches nothing (standard fnmatch semantics), so
            ``include=[]`` selects zero files.
        exclude: Optional list of :mod:`fnmatch`-style globs. Rows matching any
            pattern are dropped (applied after ``include``). ``None`` (default)
            and ``[]`` both drop nothing.

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
    per-file pipeline: skip/re-fetch policy + staging + verify + rename).
    """

    # Class-level fallbacks so subclasses that define ``__init__`` without calling
    # ``super().__init__`` still get well-defined (unfiltered) behavior.
    _include: list[str] | None = None
    _exclude: list[str] | None = None

    def __init__(self, include: list[str] | None = None, exclude: list[str] | None = None):
        # ``is not None`` (not truthiness): ``include=[]`` must mean "no pattern
        # matches anything" — i.e. select zero files — per fnmatch semantics, not
        # silently collapse to "select the entire release".
        self._include = list(include) if include is not None else None
        self._exclude = list(exclude) if exclude is not None else None

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
            do_overwrite: If ``True``, skip the verified-dest check and clear
                ``dest`` / ``.part`` (and, for archives, the extraction marker)
                before each fetch — re-fetches and re-extracts everything from
                scratch, even files whose local copy verifies.

        Raises:
            Exception: From the transport layer on any per-file failure when
                ``continue_on_error=False``.
            ExceptionGroup: When ``continue_on_error=True`` and at least one
                file failed.
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

            ``do_overwrite=True`` re-fetches even a file whose on-disk copy
            verifies — ``_pull`` runs exactly once across the two calls below
            (skipped without overwrite, forced with it):

            >>> pulls = []
            >>> class CountingSource(Source):
            ...     def _list_files(self):
            ...         return [RemoteFile("x.txt", "", sha256=digest)]
            ...     def _pull(self, source_path, target):
            ...         pulls.append(source_path)
            ...         target.write_bytes(body)
            >>>
            >>> with tempfile.TemporaryDirectory() as d:
            ...     d = Path(d)
            ...     _ = (d / "x.txt").write_bytes(body)
            ...     CountingSource().download_all(d)  # verified on disk → skipped
            ...     CountingSource().download_all(d, do_overwrite=True)  # forced re-fetch
            ...     len(pulls)
            1

            An existing ``dest`` with **no manifest sha** can never be proven
            complete, so it is re-fetched rather than trusted — the stale local
            copy is replaced atomically, and re-runs stay idempotent:

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
            ...     UnverifiableSource().download_all(d)  # re-run: re-fetches again, no error
            ...     print((d / "x.txt").read_text())
            fresh

            An existing ``dest`` whose content **mismatches** the manifest sha is
            likewise re-fetched (with a warning naming the file) — and the fresh
            bytes must still verify before the atomic replace:

            >>> class MismatchRepairSource(Source):
            ...     def _list_files(self):
            ...         return [RemoteFile("x.txt", "", sha256=digest)]
            ...     def _pull(self, source_path, target):
            ...         target.write_bytes(body)
            >>>
            >>> with tempfile.TemporaryDirectory() as d:
            ...     d = Path(d)
            ...     _ = (d / "x.txt").write_bytes(b"corrupt local copy")
            ...     MismatchRepairSource().download_all(d)
            ...     print((d / "x.txt").read_bytes() == body)
            True

            Failure policy: by default the first per-file failure propagates and
            later files are not attempted; ``continue_on_error=True`` attempts
            everything and collects the failures into one :class:`ExceptionGroup`:

            >>> class FlakySource(Source):
            ...     def _list_files(self):
            ...         return [RemoteFile("bad.txt", "bad"), RemoteFile("good.txt", "ok")]
            ...     def _pull(self, source_path, target):
            ...         if source_path == "bad":
            ...             raise RuntimeError("transport boom")
            ...         target.write_text("ok")
            >>>
            >>> with tempfile.TemporaryDirectory() as d:
            ...     d = Path(d)
            ...     try:
            ...         FlakySource().download_all(d)
            ...     except RuntimeError as e:
            ...         print(f"raised: {e}; good.txt fetched: {(d / 'good.txt').exists()}")
            raised: transport boom; good.txt fetched: False

            >>> with tempfile.TemporaryDirectory() as d:
            ...     d = Path(d)
            ...     try:
            ...         FlakySource().download_all(d, continue_on_error=True)
            ...     except ExceptionGroup as eg:
            ...         print(f"{len(eg.exceptions)} failed; good.txt fetched: {(d / 'good.txt').exists()}")
            1 failed; good.txt fetched: True

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
        counts = {"fetched": 0, "skipped": 0, "promoted": 0, "extracted": 0}
        n_failed = 0
        total_bytes = 0
        fetched_bytes = 0
        t0 = last_progress = time.monotonic()
        # ``closing`` guarantees the generator's ``finally`` runs even when the loop
        # exits early via ``raise`` (fail-fast) — in pooled mode that ``finally`` is
        # what cancels the still-queued futures so "fail fast" actually stops the run.
        # Tripped the moment a fetch fails, so queued work stops before the unwind
        # reaches the generator's teardown. See ``_attempts``.
        abort = threading.Event()
        attempts = self._attempts(self._iter_attempts(items, dest_dir, do_overwrite), pool, abort)
        try:
            with closing(attempts):
                for item, run in attempts:
                    try:
                        status, n_bytes = run()
                    except Exception as e:
                        # Tag the exception with the item it came from so a caller
                        # inspecting the ExceptionGroup (or a bare re-raise) can tell
                        # which file failed without cross-referencing logs.
                        e.add_note(f"while fetching {item.rel_path!r} from {item.source_path!r}")
                        n_failed += 1
                        if not continue_on_error:
                            abort.set()
                            raise
                        logger.exception(f"Failed to fetch {item.rel_path}")
                        errors.append(e)
                    else:
                        counts[status] += 1
                        total_bytes += n_bytes
                        if status == "fetched":
                            fetched_bytes += n_bytes
                    now = time.monotonic()
                    if now - last_progress >= _PROGRESS_INTERVAL_S:
                        n_done = sum(counts.values()) + n_failed
                        logger.info(
                            f"Progress: {n_done}/{self.n_files} files "
                            f"({total_bytes / 2**20:.0f} MiB) in {now - t0:.0f}s "
                            f"({total_bytes / 2**20 / max(now - t0, 1e-9):.1f} MiB/s)"
                        )
                        last_progress = now
        finally:
            # Emitted in a ``finally`` so a fail-fast exit still reports the partial
            # totals a multi-hour run accumulated before the failure.
            elapsed = time.monotonic() - t0
            fetched_mib = fetched_bytes / 2**20
            logger.info(
                f"{type(self).__name__}: {counts['fetched']} fetched "
                f"({fetched_mib:.1f} MiB in {elapsed:.1f}s, "
                f"{fetched_mib / max(elapsed, 1e-9):.1f} MiB/s), "
                f"{counts['skipped']} skipped, {counts['promoted']} promoted, "
                f"{counts['extracted']} re-extracted, "
                f"{n_failed} failed of {self.n_files} files -> {dest_dir}"
            )
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

            ``exclude`` is applied after ``include`` — both together intersect:

            >>> [f.rel_path for f in TreeSource(include=["hosp/*"], exclude=["*/labevents*"]).files]
            ['hosp/patients.csv.gz']

            An empty ``include`` list matches nothing (fnmatch semantics) — it
            selects zero files rather than silently selecting everything:

            >>> TreeSource(include=[]).files
            []
        """
        t0 = time.monotonic()
        items = list(self._list_files())
        logger.info(
            f"{type(self).__name__}: manifest listed {len(items)} files in {time.monotonic() - t0:.1f}s"
        )
        if self._include is not None or self._exclude is not None:
            kept = [item for item in items if self._selected(item)]
            # Only log when this pass actually dropped rows — backends that
            # pre-filter inside ``_list_files`` (e.g. FsspecSource, to skip
            # hashing excluded bytes) hand us an already-filtered manifest, and
            # an unconditional "kept N/N" line would misread as "filters were a
            # no-op". Those backends log their own pre-filter counts.
            if len(kept) != len(items):
                logger.info(
                    f"include/exclude filters dropped {len(items) - len(kept)} of "
                    f"{len(items)} manifest rows ({len(kept)} kept)"
                )
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
        if Path(rel_path).is_absolute():
            raise ValueError(f"rel_path must be relative, got absolute: {rel_path!r}")
        resolved, contained = resolve_contained(dest_dir, rel_path)
        if not contained:
            raise ValueError(
                f"rel_path {rel_path!r} escapes dest_dir {Path(dest_dir).resolve()} (resolved to {resolved})."
            )
        return resolved

    @staticmethod
    def _verifies(dest: Path, item: RemoteFile) -> bool:
        """True iff ``dest`` exists AND the manifest's ``sha256`` matches.

        SHA-256 is the only verifier we trust. Same-size files can have different
        content; existence-with-no-hash means the file on disk could be anything.
        Backends that want skip-on-rerun semantics must populate ``sha256``.
        """
        return item.sha256 is not None and dest.exists() and sha256_of(dest) == item.sha256

    @staticmethod
    def _archive_format(item: RemoteFile, dest: Path) -> ArchiveFormat | None:
        """The canonical format ``item`` will be unpacked as, or ``None`` if no unpack applies.

        Resolved against ``dest``'s name — the archive's final name — never against a
        ``.part`` staging name, which ``"auto"`` would not recognize.

        Examples:
            >>> Source._archive_format(RemoteFile("b.zip", "", unarchive="auto"), Path("b.zip"))
            <ArchiveFormat.ZIP: 'zip'>
            >>> Source._archive_format(RemoteFile("p.csv.gz", "", unarchive="auto"), Path("p.csv.gz"))
            >>> Source._archive_format(RemoteFile("b.zip", ""), Path("b.zip"))
        """
        if not item.unarchive:
            return None
        return resolve_format(item.unarchive, dest)

    @staticmethod
    def _cleanup_requested(item: RemoteFile) -> bool:
        """Resolve ``item.cleanup_archive``'s tri-state: ``None`` means "only under ``"auto"``".

        Examples:
            >>> Source._cleanup_requested(RemoteFile("b.zip", "", unarchive="auto"))
            True
            >>> Source._cleanup_requested(RemoteFile("b.zip", "", unarchive="zip"))
            False
            >>> Source._cleanup_requested(RemoteFile("b.zip", "", unarchive="zip", cleanup_archive=True))
            True
        """
        if item.cleanup_archive is None:
            return ArchiveFormat(item.unarchive) is ArchiveFormat.AUTO
        return item.cleanup_archive

    @staticmethod
    def _extraction_complete(item: RemoteFile, dest: Path, fmt: ArchiveFormat) -> bool:
        """True iff ``dest``'s completion marker proves ``item`` was fully extracted.

        The marker must exist, parse, and record this row's manifest ``sha256``,
        ``fmt``, and ``archive_subdir`` — so a new archive version or a changed
        ``archive_subdir`` triggers re-extraction — and every member it lists must
        still exist (a cheap ``lstat`` each; catches members deleted since). Rows
        without a manifest ``sha256`` can never be proven complete, mirroring
        :meth:`_verifies`. The archive itself is not re-hashed here; whether it
        still needs to verify is :meth:`_fetch_one`'s call.

        Examples:
            >>> import zipfile
            >>> digest = "a" * 64
            >>> item = RemoteFile("b.zip", "", sha256=digest, unarchive="zip")
            >>> with tempfile.TemporaryDirectory() as d:
            ...     archive = Path(d) / "b.zip"
            ...     with zipfile.ZipFile(archive, "w") as zf:
            ...         zf.writestr("a.csv", "x")
            ...     before = Source._extraction_complete(item, archive, ArchiveFormat.ZIP)
            ...     Source._maybe_unarchive(item, archive)
            ...     after = Source._extraction_complete(item, archive, ArchiveFormat.ZIP)
            ...     other = RemoteFile("b.zip", "", sha256="b" * 64, unarchive="zip")
            ...     stale_sha = Source._extraction_complete(other, archive, ArchiveFormat.ZIP)
            ...     (Path(d) / "a.csv").unlink()
            ...     member_gone = Source._extraction_complete(item, archive, ArchiveFormat.ZIP)
            ...     (before, after, stale_sha, member_gone)
            (False, True, False, False)
        """
        if item.sha256 is None:
            return False
        try:
            marker = json.loads(extraction_marker_path(dest).read_text())
        except (OSError, ValueError):
            return False
        expected = {
            "marker_version": _MARKER_VERSION,
            "archive_sha256": item.sha256,
            "format": fmt.value,
            "archive_subdir": item.archive_subdir,
        }
        if not isinstance(marker, dict) or any(marker.get(k) != v for k, v in expected.items()):
            return False
        members = marker.get("members")
        if not isinstance(members, list):
            return False
        return all(os.path.lexists(dest.parent / m) for m in members)

    @staticmethod
    def _maybe_unarchive(item: RemoteFile, dest: Path) -> None:
        """Post-fetch unpack hook — runs after bytes newly land at ``dest``.

        A no-op unless ``item.unarchive`` is set. ``"auto"`` resolves the format
        from ``dest``'s extension (and is a no-op on non-archive extensions like
        ``.csv.gz``); explicit formats dispatch directly. Extraction lands in
        ``dest``'s directory via
        :func:`~MEDS_extract.download.unarchive.safe_extract`, which validates
        every member against zip-slip / tar-slip before any bytes are written.
        ``item.archive_subdir`` restricts (and flattens) what is extracted.

        All-or-nothing: the archive is first extracted into a scratch directory
        beside it (:func:`_extraction_scratch_path`; a leftover from a killed run is
        removed first), then each member is moved into place, and only then is the
        completion marker (:func:`extraction_marker_path`) written — atomically,
        via a rename. Any existing marker is removed before work starts, and the
        scratch directory is removed on any exception, so a failed or interrupted
        extraction leaves no marker and no partial tree, and the next run
        re-extracts.

        Tri-state cleanup: ``cleanup_archive=None`` defers to the unarchive mode —
        :attr:`~MEDS_extract.download.unarchive.ArchiveFormat.AUTO` removes the
        archive (the one-arg "fetch + extract + drop" flow), explicit formats keep
        it. Explicit ``True`` / ``False`` always wins. The archive is removed only
        after the marker is written, so a crash in between leaves a verified
        archive and a valid marker rather than neither.

        Invoked from :meth:`_fetch_one` on the ``"fetched"`` and ``"promoted"``
        paths, and on the ``"extracted"`` path when a verified archive has no
        matching marker.

        Examples:
            ``unarchive="auto"`` unpacks a zip next to itself, records the marker,
            and (by AUTO's cleanup default) removes the archive afterwards:

            >>> import zipfile
            >>> with tempfile.TemporaryDirectory() as d:
            ...     d = Path(d)
            ...     archive = d / "bundle.zip"
            ...     with zipfile.ZipFile(archive, "w") as zf:
            ...         zf.writestr("sub/a.csv", "col\\n1")
            ...     Source._maybe_unarchive(RemoteFile("bundle.zip", "", unarchive="auto"), archive)
            ...     print_directory(d)
            ...     marker = json.loads((d / ".bundle.zip.extracted.json").read_text())
            ...     print(marker["format"], marker["members"], marker["archive_sha256"])
            ├── .bundle.zip.extracted.json
            └── sub
                └── a.csv
            zip ['sub/a.csv'] None

            An explicit format keeps the archive by default:

            >>> with tempfile.TemporaryDirectory() as d:
            ...     d = Path(d)
            ...     archive = d / "bundle.zip"
            ...     with zipfile.ZipFile(archive, "w") as zf:
            ...         zf.writestr("a.csv", "col\\n1")
            ...     Source._maybe_unarchive(RemoteFile("bundle.zip", "", unarchive="zip"), archive)
            ...     print_directory(d)
            ├── .bundle.zip.extracted.json
            ├── a.csv
            └── bundle.zip

            ``cleanup_archive`` overrides the mode default in either direction, and
            ``archive_subdir`` extracts one wrapped folder flat:

            >>> with tempfile.TemporaryDirectory() as d:
            ...     d = Path(d)
            ...     archive = d / "bundle.zip"
            ...     with zipfile.ZipFile(archive, "w") as zf:
            ...         zf.writestr("proj-1.0/data/a.csv", "col\\n1")
            ...         zf.writestr("proj-1.0/docs/readme.txt", "hi")
            ...     item = RemoteFile(
            ...         "bundle.zip", "", unarchive="zip", cleanup_archive=True, archive_subdir="data"
            ...     )
            ...     Source._maybe_unarchive(item, archive)
            ...     print_directory(d)
            ├── .bundle.zip.extracted.json
            └── a.csv

            A failed extraction leaves no marker and no scratch directory behind
            (here ``archive_subdir`` matches nothing):

            >>> with tempfile.TemporaryDirectory() as d:
            ...     d = Path(d)
            ...     archive = d / "bundle.zip"
            ...     with zipfile.ZipFile(archive, "w") as zf:
            ...         zf.writestr("a.csv", "col\\n1")
            ...     item = RemoteFile("bundle.zip", "", unarchive="zip", archive_subdir="nope")
            ...     try:
            ...         Source._maybe_unarchive(item, archive)
            ...     except ValueError as e:
            ...         print(f"raised: {e}")
            ...     print_directory(d)
            raised: archive_subdir 'nope' matched no members of ...
            └── bundle.zip

            ``"auto"`` on a non-archive is a no-op — the file is left alone:

            >>> with tempfile.TemporaryDirectory() as d:
            ...     d = Path(d)
            ...     f = d / "patients.csv.gz"
            ...     _ = f.write_bytes(b"not an archive")
            ...     Source._maybe_unarchive(RemoteFile("patients.csv.gz", "", unarchive="auto"), f)
            ...     print_directory(d)
            └── patients.csv.gz
        """
        fmt = Source._archive_format(item, dest)
        if fmt is None:
            return
        marker = extraction_marker_path(dest)
        scratch = _extraction_scratch_path(dest)
        # The tree is about to change, so any previous witness no longer holds.
        marker.unlink(missing_ok=True)
        shutil.rmtree(scratch, ignore_errors=True)
        t0 = time.monotonic()
        try:
            safe_extract(dest, scratch, fmt, archive_subdir=item.archive_subdir)
            members = _move_tree(scratch, dest.parent)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        payload = {
            "marker_version": _MARKER_VERSION,
            "archive": dest.name,
            "archive_sha256": item.sha256,
            "format": fmt.value,
            "archive_subdir": item.archive_subdir,
            "members": members,
        }
        marker_part = marker.with_name(marker.name + ".part")
        marker_part.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(marker_part, marker)
        logger.debug(
            f"Extracted {len(members)} members from {item.rel_path} ({fmt.value}) in "
            f"{time.monotonic() - t0:.1f}s"
        )
        if Source._cleanup_requested(item):
            dest.unlink()

    def _fetch_one(self, item: RemoteFile, dest_dir: Path, do_overwrite: bool) -> tuple[str, int]:
        """Fetch one manifest entry end-to-end: policy → ``.part`` staging → verify → rename.

        Pipeline:

        1. Resolve ``dest = dest_dir / item.rel_path`` (with traversal validation).
        2. If ``do_overwrite=True``: unconditionally clear ``dest``, any stale
           ``.part``, and (for archives) the extraction marker, then proceed to
           step 5. Otherwise, for an archive row (``item.unarchive`` resolves to a
           format) whose ``cleanup_archive`` drops the archive, a matching
           extraction marker (:meth:`_extraction_complete`) means "done": skip and
           return, finishing an interrupted cleanup if the archive is still there.
           Then, on a pre-existing ``dest``:

           - ``dest`` verifies against ``item.sha256``: skip and return — unless
             it is an archive without a matching marker, in which case it is
             re-extracted from the local copy (no re-fetch) and ``"extracted"``
             is returned.
           - ``item.sha256`` is set but ``dest`` mismatches: re-fetch (with a
             warning naming the file) — ``dest`` is replaced only by the atomic
             rename in step 7, after the fresh bytes verify.
           - no ``item.sha256``: re-fetch — a file we can't prove matches the
             manifest is never trusted, and never skipped.

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
        8. If ``item.unarchive`` is set, run the post-fetch unpack hook
           (:meth:`_maybe_unarchive`) — also applied on the promote path in
           step 3.

        Returns:
            A ``(status, n_bytes)`` tuple where ``status`` is ``"skipped"``
            (already-complete dest), ``"promoted"`` (complete ``.part``
            renamed without re-fetching), ``"extracted"`` (verified local
            archive re-extracted without re-fetching), or ``"fetched"`` (bytes
            actually transferred), and ``n_bytes`` is the size of the file the
            status applies to (``0`` for a skipped archive that was already
            cleaned up). ``download_all`` tallies these into its end-of-bundle
            summary.

        On any transport or verification exception, no new ``dest`` is created
        and an existing ``dest`` is never modified — the re-fetch paths replace
        it only via the atomic rename, after the fresh bytes verify. ``part``
        may exist after a partial transport failure (intentional — gives a
        future run a head start via Range-resume on backends that support it).
        An *extraction* failure happens after the rename, so the verified
        archive stays at ``dest`` — but without a completion marker, so the next
        run re-extracts it rather than skipping it.

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
            ('fetched', 2)
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

            When the manifest has no SHA, a stale ``.part`` from a prior failed
            run can't be safely resumed (nothing would catch silent corruption),
            so it is discarded before ``_pull`` runs — the backend starts fresh:

            >>> class NoShaSource(Source):
            ...     def _list_files(self):
            ...         return [RemoteFile("x.txt", "dummy")]  # no sha
            ...     def _pull(self, source_path, target):
            ...         print(f"stale .part visible to _pull: {target.exists()}")
            ...         target.write_text("fresh")
            >>> with tempfile.TemporaryDirectory() as d:
            ...     d = Path(d)
            ...     _ = (d / "x.txt.part").write_bytes(b"stale partial")
            ...     src = NoShaSource()
            ...     [item] = src.files
            ...     src._fetch_one(item, d, do_overwrite=False)
            stale .part visible to _pull: False
            ('fetched', 5)

            A leftover ``.part`` that already verifies against the manifest sha
            (prior run died between the last byte and the rename) is promoted to
            ``dest`` directly — ``_pull`` is never invoked:

            >>> body = b"the whole file, fully written"
            >>> class NoRefetchSource(Source):
            ...     def _list_files(self):
            ...         return [RemoteFile("x.txt", "dummy", sha256=hashlib.sha256(body).hexdigest())]
            ...     def _pull(self, source_path, target):
            ...         raise AssertionError("must not re-fetch a complete .part")
            >>> with tempfile.TemporaryDirectory() as d:
            ...     d = Path(d)
            ...     _ = (d / "x.txt.part").write_bytes(body)
            ...     src = NoRefetchSource()
            ...     [item] = src.files
            ...     src._fetch_one(item, d, do_overwrite=False)
            ...     print((d / "x.txt").read_bytes() == body, (d / "x.txt.part").exists())
            ('promoted', 29)
            True False
        """
        dest = self._resolve_dest(dest_dir, item.rel_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")

        fmt = self._archive_format(item, dest)

        if do_overwrite:
            # Clear both independently — a ``.part`` from a half-finished prior
            # run can exist even when ``dest`` doesn't, and either one left in
            # place would be picked up as a Range-resume base.
            if dest.exists():
                dest.unlink()
            if part.exists():
                part.unlink()
            if fmt is not None:
                extraction_marker_path(dest).unlink(missing_ok=True)
        elif fmt is not None and self._cleanup_requested(item) and self._extraction_complete(item, dest, fmt):
            # The marker is the witness once the archive is meant to be gone — the
            # archive's absence must not force a re-download (#212).
            if dest.exists():
                logger.debug(f"Removing {item.rel_path}: extracted, but its cleanup was interrupted.")
                dest.unlink()
            logger.debug(f"Skipping {item.rel_path}: already extracted.")
            return ("skipped", 0)
        elif dest.exists():
            if self._verifies(dest, item):
                if fmt is None or self._extraction_complete(item, dest, fmt):
                    logger.debug(f"Skipping {item.rel_path}: already complete.")
                    return ("skipped", dest.stat().st_size)
                # Verified archive, but no marker vouching for its members: a prior
                # extraction died part-way (#211), or the extraction settings changed.
                # Re-extract from the local copy; there is nothing to re-download.
                logger.info(f"Re-extracting {item.rel_path}: no matching extraction marker.")
                n_bytes = dest.stat().st_size
                self._maybe_unarchive(item, dest)
                return ("extracted", n_bytes)
            # An existing dest that can't be proven complete is never trusted and
            # never skipped: re-fetch it. The stale copy stays in place until the
            # fresh bytes verify — the atomic rename below is what replaces it.
            if item.sha256 is None:
                logger.debug(f"Re-fetching {item.rel_path}: existing file has no manifest sha to verify.")
            else:
                logger.warning(
                    f"Re-fetching {item.rel_path}: existing file at {dest} failed SHA-256 "
                    "verification against the manifest."
                )

        if item.sha256 is not None:
            # A prior run may have died between writing the last byte and the
            # rename below — in that case the ``.part`` is the complete file and
            # re-fetching it (or bouncing off an unsatisfiable Range request)
            # wastes the whole transfer. Verify and promote directly.
            if part.exists() and sha256_of(part) == item.sha256:
                logger.debug(f"Promoting complete .part for {item.rel_path} without re-fetching.")
                n_bytes = part.stat().st_size
                part.replace(dest)
                self._maybe_unarchive(item, dest)
                return ("promoted", n_bytes)
        elif part.exists():
            # Resume-without-verification is unsafe: without a sha to catch silent
            # corruption, a stale ``.part`` could be from a different version of
            # the source file. Clear it so ``_pull`` starts fresh. With sha set,
            # Range-resume is safe because the post-write verify catches mismatches.
            part.unlink()

        t_pull = time.monotonic()
        self._pull(item.source_path, part)
        pull_s = time.monotonic() - t_pull
        n_bytes = part.stat().st_size

        # Hash once, compare once — the failure message reuses the digest, so
        # ``_verifies`` (which would re-hash) is deliberately not used here.
        verify_note = ""
        if item.sha256 is not None:
            t_hash = time.monotonic()
            actual = sha256_of(part)
            verify_note = f" + {time.monotonic() - t_hash:.2f}s verify"
            if actual != item.sha256:
                part.unlink()
                raise ChecksumError(item.source_path, item.sha256, actual)
        part.replace(dest)
        self._maybe_unarchive(item, dest)
        logger.debug(f"Fetched {item.rel_path}: {n_bytes} bytes in {pull_s:.2f}s transfer{verify_note}")
        return ("fetched", n_bytes)

    def _iter_attempts(
        self, items: list[RemoteFile], dest_dir: Path, do_overwrite: bool
    ) -> Iterator[tuple[RemoteFile, Callable[[], tuple[str, int]]]]:
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
        items_to_fetch: Iterable[tuple[RemoteFile, Callable[[], tuple[str, int]]]],
        pool: Executor | None,
        abort: threading.Event,
    ) -> Iterator[tuple[RemoteFile, Callable[[], tuple[str, int]]]]:
        """Dispatch ``(item, callable)`` pairs sequentially or through a pool.

        Sequential mode yields the input pairs unchanged; the caller invokes
        them in the main thread. Parallel mode submits callables to ``pool``
        through a bounded sliding window — an initial batch of at most
        :data:`_MAX_PENDING_SUBMITS`, then one fresh submission per completion
        — and yields ``(item, future.result)`` pairs in completion order.
        Bounding the window keeps dispatch bookkeeping O(window) rather than
        O(manifest): each pending Future costs ~2 KB, which adds up to
        hundreds of MB on 100k+-row manifests if submitted all at once.

        Fail-fast in parallel mode: if the caller raises out of its loop on the first
        failure, the ``finally`` halts the rest of the bundle. That takes TWO
        mechanisms, because either alone leaks work:

        - ``Future.cancel()`` stops futures that have not started. It is a no-op on a
          future already running — there is no way to interrupt a thread mid-fetch.
        - An **abort flag**, checked at the top of every queued fetch, stops the ones
          that start between the failure surfacing and the cancel landing. Without it,
          a worker freed by the failure immediately picks up the next queued item and
          runs it to completion, and with a window of :data:`_MAX_PENDING_SUBMITS`
          there can be many such items — each one a real transfer in production.

        WHEN the flag is set is what makes it effective. ``download_all`` trips it the
        instant a fetch raises, before re-raising — not here in the ``finally``. Waiting
        for teardown leaves the worker free to drain queued items during the unwind:
        measured on a loaded box with microsecond-fast fetches, that leaked up to 15 of
        19 queued items. Tripping it at detection cuts the leak to the fetches already
        in flight, which nothing short of killing threads could stop.

        This ``finally`` still sets it, covering early exits that are not failures (a
        caller ``break``), and cancels the queue either way.

        The caller must wrap the generator in :func:`contextlib.closing` to guarantee
        the ``finally`` runs.
        """
        if pool is None:
            yield from items_to_fetch
            return
        item_iter = iter(items_to_fetch)

        def guarded(run: Callable[[], tuple[str, int]]) -> Callable[[], tuple[str, int]]:
            """Wrap a fetch so it becomes a no-op once fail-fast has tripped."""

            def _run() -> tuple[str, int]:
                if abort.is_set():
                    raise _AbortedError
                return run()

            return _run

        pending = {
            pool.submit(guarded(run)): item for item, run in itertools.islice(item_iter, _MAX_PENDING_SUBMITS)
        }
        try:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    item = pending.pop(fut)
                    # Replenish before yielding: if the caller raises out of this
                    # yield, the fresh submission is still queued and the
                    # ``finally`` below cancels it — same fail-fast semantics as
                    # the queued remainder of an up-front submission.
                    nxt = next(item_iter, None)
                    if nxt is not None:
                        nxt_item, nxt_run = nxt
                        pending[pool.submit(guarded(nxt_run))] = nxt_item
                    yield item, fut.result
        finally:
            # Flag first, then cancel: a future that slips past ``cancel`` still sees
            # the flag and returns without fetching.
            abort.set()
            for fut in pending:
                fut.cancel()


def validate_unique_destinations(sources: Iterable[Source]) -> None:
    """Reject destination collisions across multiple sources sharing one ``dest_dir``.

    :attr:`Source.files` already rejects collisions *within* one source, but the
    CLI (and any caller composing sources) stages several sources into one shared
    directory, where two sources legally listing the same ``rel_path`` would race
    on the same ``.part`` file under concurrent workers — or serially clobber
    each other. Calling this before any fetch turns that late, confusing failure
    into an immediate, precise config error.

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

        Colliding sources are named by their position in the resolved list (plus
        class name), so several same-type entries — the common case, e.g. a
        ``common:`` bucket of multiple ``HTTPSource`` entries — stay
        distinguishable in the error:

        >>> validate_unique_destinations([A(), B()])
        Traceback (most recent call last):
            ...
        ValueError: Duplicate destination ... 'sub/../x.csv' from B#1 collides with 'x.csv' from A#0.

        Distinct destinations pass silently:

        >>> class C(Source):
        ...     def _list_files(self):
        ...         return [RemoteFile("y.csv", "")]
        ...     def _pull(self, source_path, target):
        ...         target.write_text("C")
        >>> validate_unique_destinations([A(), C()])
    """
    seen: dict[str, tuple[str, RemoteFile]] = {}
    for idx, source in enumerate(sources):
        # ``ClassName#index`` (enumeration order in the resolved list): specs
        # routinely declare several entries of the same type, so the class name
        # alone would leave "HTTPSource collides with HTTPSource" ambiguous.
        name = f"{type(source).__name__}#{idx}"
        for item in source.files:
            if item.dest_key in seen:
                prior_name, prior_item = seen[item.dest_key]
                raise ValueError(
                    f"Duplicate destination across sources: {item.rel_path!r} from {name} "
                    f"collides with {prior_item.rel_path!r} from {prior_name}."
                )
            seen[item.dest_key] = (name, item)
