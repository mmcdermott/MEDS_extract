"""Post-fetch archive unpack for :class:`~MEDS_extract.download.source.Source`.

Two of the public ETLs (AUMCdb and HIRID) ship their raw data as a single archive that
the rest of the pipeline can't read directly. Rather than pushing archive handling into
every ETL's ``pre_MEDS.py``, we expose an optional ``unarchive`` field on each
:class:`~MEDS_extract.download.source.RemoteFile` and unpack in the base
:meth:`~MEDS_extract.download.source.Source.fetch` — so every transport picks it up for
free without bespoke plumbing.

Why stdlib (``zipfile`` / ``tarfile``) and not a third-party library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our format set is narrow: ``zip`` (AUMCdb), ``tar.gz`` (HIRID), plus ``tar`` and ``tgz``
for completeness. Both are first-class in stdlib. Alternatives we considered:

- ``libarchive`` (via ``python-libarchive`` / ``libarchive-c``): native C performance and
  ~30 supported formats. **Rejected**: needs the system ``libarchive`` shared lib
  installed (extra OS-level install friction), and our entire format requirement is two
  formats — neither win matters for one-shot extraction of CSVs that came over the wire.
- ``patool`` / ``pyunpack``: shell out to ``7z``, ``unrar``, etc. **Rejected**: needs
  external binaries on ``PATH``, runs subprocesses, and again only buys format breadth
  we don't need.
- ``extractcode``: kitchen-sink wrapper over libarchive + 7z + stdlib. **Rejected**:
  same dep-weight issues amplified.

So: import stdlib ``zipfile`` and ``tarfile`` directly. The code in this module is not
"we re-implemented zip extraction" — it is the *path-validation wrapper* around the
stdlib calls, surfacing a clear :class:`ValueError` that names the offending archive
member when a zip-slip / tar-slip / symlink-escape attempt is found.

Module shape: :class:`ArchiveFormat` (a :class:`enum.StrEnum`) is the single source of
truth for what we support. :func:`safe_extract` does format dispatch via a
``{format → extractor function}`` table; per-format extractors share a common safety
pre-pass via :func:`_validate_member`. Adding a new format means: add an enum value,
write a one-screen extractor function, register it in ``_EXTRACTORS``.

Safety: every member path is validated before any bytes are written — a traversal
attempt raises :class:`ValueError` while the target directory is still clean. Both
zip-slip and tar-slip variants are covered: absolute paths, ``..`` components, and
symlinks / hardlinks that point outside the target.

The tarfile branch additionally uses Python 3.12+'s ``data_filter`` (PEP 706, the
official fix for CVE-2007-4559), which rejects special files (devices, fifos), strips
the setuid / setgid bits, and re-enforces the same path constraints. Our explicit
pre-pass is defense-in-depth: it surfaces a clear :class:`ValueError` naming the
offending member rather than tarfile's generic :class:`tarfile.FilterError`.
"""

from __future__ import annotations

import logging
import tarfile
import zipfile
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class ArchiveFormat(StrEnum):
    """Recognized ``unarchive`` values.

    ``AUTO`` defers the decision to :func:`infer_format` at extraction time and is the
    one-arg way to opt into the full "fetch → extract → cleanup-archive" flow (see
    :class:`~MEDS_extract.download.source.RemoteFile` ``cleanup_archive`` semantics).
    Explicit values (``ZIP``, ``TAR``, ``TAR_GZ``) are used when the URL extension is
    misleading — e.g. a content-negotiated download URL like
    ``…/api/access/datafile/123`` that returns a zip body without a ``.zip`` suffix.

    ``TGZ`` is an alias that resolves to ``TAR_GZ``; preserved as a distinct enum
    value so users can write ``unarchive: tgz`` in YAML and it round-trips through
    :func:`resolve_format` to the canonical ``"tar.gz"`` extractor.

    Examples:
        >>> ArchiveFormat.AUTO
        <ArchiveFormat.AUTO: 'auto'>
        >>> ArchiveFormat("zip")
        <ArchiveFormat.ZIP: 'zip'>
        >>> ArchiveFormat("tar.gz")
        <ArchiveFormat.TAR_GZ: 'tar.gz'>
        >>> ArchiveFormat("tgz")
        <ArchiveFormat.TGZ: 'tgz'>
        >>> ArchiveFormat("rar")
        Traceback (most recent call last):
            ...
        ValueError: 'rar' is not a valid ArchiveFormat
    """

    AUTO = "auto"
    ZIP = "zip"
    TAR = "tar"
    TAR_GZ = "tar.gz"
    TGZ = "tgz"


# Maps the on-disk file extension to the canonical ArchiveFormat. Order matters
# only for ``.tar.gz`` (must be checked before ``.gz`` would, were ``.gz`` a key).
_EXT_TO_FORMAT: tuple[tuple[str, ArchiveFormat], ...] = (
    (".tar.gz", ArchiveFormat.TAR_GZ),
    (".tgz", ArchiveFormat.TAR_GZ),
    (".tar", ArchiveFormat.TAR),
    (".zip", ArchiveFormat.ZIP),
)


def infer_format(path: Path) -> ArchiveFormat | None:
    """Return the archive format implied by a filename, or ``None`` if unknown.

    Uses lowercase suffix matching — ``AUMCdb.ZIP`` and ``AUMCdb.zip`` both resolve
    to :attr:`ArchiveFormat.ZIP`. Returns ``None`` for any other extension; callers
    treat that as "no unpack needed" under :attr:`ArchiveFormat.AUTO` semantics.

    Examples:
        >>> infer_format(Path("AUMCdb.zip"))
        <ArchiveFormat.ZIP: 'zip'>
        >>> infer_format(Path("hirid.tar.gz"))
        <ArchiveFormat.TAR_GZ: 'tar.gz'>
        >>> infer_format(Path("raw_stage.tgz"))
        <ArchiveFormat.TAR_GZ: 'tar.gz'>
        >>> infer_format(Path("release.tar"))
        <ArchiveFormat.TAR: 'tar'>

        Case-insensitive on the extension (not the whole path):

        >>> infer_format(Path("RELEASE.TAR.GZ"))
        <ArchiveFormat.TAR_GZ: 'tar.gz'>

        Non-archive extensions return ``None`` so ``AUTO`` is a no-op on regular
        files — e.g. PhysioNet's ``LICENSE.txt`` or ``patients.csv.gz``:

        >>> infer_format(Path("patients.csv.gz"))  # gz compression, not a tar archive
        >>> infer_format(Path("README.md"))
        >>> infer_format(Path("plain"))
    """
    name = path.name.lower()
    for ext, fmt in _EXT_TO_FORMAT:
        if name.endswith(ext):
            return fmt
    return None


def resolve_format(unarchive: str | ArchiveFormat, path: Path) -> ArchiveFormat | None:
    """Normalize a user-facing ``unarchive`` token to a canonical :class:`ArchiveFormat`.

    :attr:`ArchiveFormat.AUTO` delegates to :func:`infer_format` based on ``path``'s
    extension; :attr:`ArchiveFormat.TGZ` folds to :attr:`ArchiveFormat.TAR_GZ`
    (same extractor). Returns ``None`` when ``AUTO`` is requested but no known
    extension matches — the caller treats that as "leave the file alone".

    Examples:
        >>> resolve_format("zip", Path("x.zip"))
        <ArchiveFormat.ZIP: 'zip'>
        >>> resolve_format("tgz", Path("x.tgz"))  # folded to canonical TAR_GZ
        <ArchiveFormat.TAR_GZ: 'tar.gz'>
        >>> resolve_format("auto", Path("AUMCdb.zip"))
        <ArchiveFormat.ZIP: 'zip'>
        >>> resolve_format("auto", Path("hirid.tar.gz"))
        <ArchiveFormat.TAR_GZ: 'tar.gz'>

        ``AUTO`` on a non-archive returns ``None`` (caller leaves the file alone):

        >>> resolve_format("auto", Path("patients.csv.gz")) is None
        True

        Unknown explicit formats raise — catching typos early beats a silent skip:

        >>> resolve_format("rar", Path("x.rar"))
        Traceback (most recent call last):
            ...
        ValueError: 'rar' is not a valid ArchiveFormat
    """
    fmt = ArchiveFormat(unarchive)
    if fmt is ArchiveFormat.AUTO:
        return infer_format(path)
    if fmt is ArchiveFormat.TGZ:
        return ArchiveFormat.TAR_GZ
    return fmt


def _validate_member(
    member_name: str,
    target_root: Path,
    *,
    archive_path: Path,
    fmt_label: str,
    link_target: str | None = None,
) -> None:
    """Raise :class:`ValueError` if extracting ``member_name`` would escape ``target_root``.

    Checks the member's name against zip-slip / tar-slip (absolute paths, ``..``
    components, anything that resolves outside ``target_root``). When ``link_target``
    is provided (tar symlinks / hardlinks), it is checked the same way — a safe member
    name pointing at ``../../etc/passwd`` would otherwise bypass the name check.

    Archive entries always use forward-slash separators per format spec, so
    :class:`~pathlib.PurePosixPath` is the right parser regardless of host OS.

    Examples:
        >>> with tempfile.TemporaryDirectory() as d:
        ...     root = Path(d).resolve()
        ...     archive = root / "x.zip"
        ...     _validate_member("sub/file.csv", root, archive_path=archive, fmt_label="zip")
        ...     _validate_member("./ok.csv", root, archive_path=archive, fmt_label="zip")

        Absolute paths, ``..`` escapes, and unsafe link targets all raise:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     root = Path(d).resolve()
        ...     archive = root / "x.tar"
        ...     _validate_member("/etc/passwd", root, archive_path=archive, fmt_label="tar")
        Traceback (most recent call last):
            ...
        ValueError: Refusing to extract unsafe tar member '/etc/passwd' from ...

        >>> with tempfile.TemporaryDirectory() as d:
        ...     root = Path(d).resolve()
        ...     archive = root / "x.zip"
        ...     _validate_member("../escaped.csv", root, archive_path=archive, fmt_label="zip")
        Traceback (most recent call last):
            ...
        ValueError: Refusing to extract unsafe zip member '../escaped.csv' from ...

        >>> with tempfile.TemporaryDirectory() as d:
        ...     root = Path(d).resolve()
        ...     archive = root / "x.tar"
        ...     _validate_member(
        ...         "good.txt", root, archive_path=archive, fmt_label="tar",
        ...         link_target="../../etc/passwd",
        ...     )
        Traceback (most recent call last):
            ...
        ValueError: Refusing to extract unsafe tar member 'good.txt' ...: link target '../../etc/passwd' ...
    """
    if not _is_safe_path(member_name, target_root):
        raise ValueError(
            f"Refusing to extract unsafe {fmt_label} member {member_name!r} from {archive_path}: "
            "member path would land outside the extraction directory."
        )
    if link_target is not None and not _is_safe_path(link_target, target_root):
        raise ValueError(
            f"Refusing to extract unsafe {fmt_label} member {member_name!r} from {archive_path}: "
            f"link target {link_target!r} would land outside the extraction directory."
        )


def _is_safe_path(member_name: str, target_root: Path) -> bool:
    """Internal helper: ``True`` iff ``member_name`` resolves strictly inside ``target_root``."""
    pure = PurePosixPath(member_name)
    if pure.is_absolute():
        return False
    resolved = (target_root / Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(target_root)
    except ValueError:
        return False
    return True


def _extract_zip(archive_path: Path, target_root: Path) -> None:
    """Extract a ``.zip`` archive into ``target_root`` after validating every member."""
    with zipfile.ZipFile(archive_path) as zf:
        for name in zf.namelist():
            _validate_member(name, target_root, archive_path=archive_path, fmt_label="zip")
        zf.extractall(target_root)


def _extract_tar_gz(archive_path: Path, target_root: Path) -> None:
    """Extract a gzip-compressed tar archive."""
    _extract_tar_with_mode(archive_path, target_root, mode="r:gz")


def _extract_tar(archive_path: Path, target_root: Path) -> None:
    """Extract an uncompressed tar archive."""
    _extract_tar_with_mode(archive_path, target_root, mode="r:")


def _extract_tar_with_mode(archive_path: Path, target_root: Path, *, mode: str) -> None:
    """Tar extraction body shared between :func:`_extract_tar` and :func:`_extract_tar_gz`.

    Validates member names AND link targets (a safe-named tar entry can still escape
    via a symlink whose ``linkname`` walks outside the target). PEP 706's
    ``data_filter`` is applied at ``extractall`` as belt-and-suspenders defense-in-depth.
    """
    with tarfile.open(archive_path, mode=mode) as tf:
        members = list(tf)
        for m in members:
            if m.isdev() or m.isfifo():
                raise ValueError(
                    f"Refusing to extract unsafe tar member {m.name!r} (type={m.type!r}): "
                    "device and fifo entries are not supported."
                )
            link_target = m.linkname if (m.issym() or m.islnk()) else None
            _validate_member(
                m.name,
                target_root,
                archive_path=archive_path,
                fmt_label="tar",
                link_target=link_target,
            )
        # PEP 706's ``data_filter`` strips setuid / setgid / sticky bits, rejects
        # special files, and re-applies the same path-traversal check we just did.
        # Defense-in-depth: if our pre-pass ever regresses, ``data_filter`` catches it.
        tf.extractall(target_root, members=members, filter="data")


# Format → extractor dispatch. Adding a new format = add an enum value above + a
# ``_extract_<fmt>`` function + an entry here. ``AUTO`` and ``TGZ`` are not in this
# table on purpose — they're aliases that ``resolve_format`` collapses to a canonical
# format before dispatch.
_EXTRACTORS: dict[ArchiveFormat, Callable[[Path, Path], None]] = {
    ArchiveFormat.ZIP: _extract_zip,
    ArchiveFormat.TAR: _extract_tar,
    ArchiveFormat.TAR_GZ: _extract_tar_gz,
}


def safe_extract(archive_path: Path, target_dir: Path, fmt: str | ArchiveFormat) -> None:
    """Extract ``archive_path`` into ``target_dir`` with path-traversal guards.

    All member paths are validated before any bytes are written — a traversal attempt
    raises :class:`ValueError` while the target directory is still clean. ``fmt``
    must be one of :attr:`ArchiveFormat.ZIP`, :attr:`~ArchiveFormat.TAR`, or
    :attr:`~ArchiveFormat.TAR_GZ`; ``AUTO`` / ``TGZ`` should be normalized via
    :func:`resolve_format` first.

    Args:
        archive_path: The archive to unpack. Must exist.
        target_dir: Destination directory. Created if absent.
        fmt: Canonical archive format (string or :class:`ArchiveFormat`).

    Raises:
        ValueError: On unsupported / alias ``fmt`` (``AUTO``, ``TGZ`` not handled
            here), on a member path that escapes ``target_dir`` (zip-slip /
            tar-slip), or on an unsupported tar entry type (device, fifo).
        FileNotFoundError: If ``archive_path`` doesn't exist.

    Examples:
        Zip round-trip — extraction preserves the internal directory layout
        under the target:

        >>> import zipfile as _zipfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     archive = d / "bundle.zip"
        ...     with _zipfile.ZipFile(archive, "w") as zf:
        ...         zf.writestr("a.csv", "col\\n1\\n2")
        ...         zf.writestr("sub/b.csv", "col\\n3\\n4")
        ...     target = d / "out"
        ...     safe_extract(archive, target, "zip")
        ...     sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file())
        ['a.csv', 'sub/b.csv']

        Tar.gz round-trip:

        >>> import io, tarfile as _tarfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     archive = d / "bundle.tar.gz"
        ...     with _tarfile.open(archive, "w:gz") as tf:
        ...         data = b"col,val\\n1,x\\n2,y\\n"
        ...         info = _tarfile.TarInfo(name="records.csv")
        ...         info.size = len(data)
        ...         tf.addfile(info, io.BytesIO(data))
        ...     target = d / "out"
        ...     safe_extract(archive, target, ArchiveFormat.TAR_GZ)
        ...     (target / "records.csv").read_bytes()
        b'col,val\\n1,x\\n2,y\\n'

        Zip-slip: a member whose relative path walks out of the target is
        rejected BEFORE any bytes land on disk, so the target directory is
        still empty after the rejection:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     archive = d / "evil.zip"
        ...     with _zipfile.ZipFile(archive, "w") as zf:
        ...         zf.writestr("../escaped.txt", "pwned")
        ...     target = d / "out"
        ...     target.mkdir()
        ...     try:
        ...         safe_extract(archive, target, "zip")
        ...     except ValueError as e:
        ...         print(f"rejected: {e}")
        ...     print(f"target empty: {sorted(p.name for p in target.iterdir()) == []}")
        rejected: Refusing to extract unsafe zip member '../escaped.txt' from ...
        target empty: True

        Absolute-path member is also rejected:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     archive = d / "evil.tar"
        ...     with _tarfile.open(archive, "w") as tf:
        ...         info = _tarfile.TarInfo(name="/etc/passwd-hijack")
        ...         info.size = 3
        ...         tf.addfile(info, io.BytesIO(b"bad"))
        ...     safe_extract(archive, d / "out", "tar")
        Traceback (most recent call last):
            ...
        ValueError: Refusing to extract unsafe tar member '/etc/passwd-hijack' ...

        Tar-slip via ``..`` is rejected the same way:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     archive = d / "evil.tar.gz"
        ...     with _tarfile.open(archive, "w:gz") as tf:
        ...         info = _tarfile.TarInfo(name="../escaped.csv")
        ...         info.size = 3
        ...         tf.addfile(info, io.BytesIO(b"bad"))
        ...     safe_extract(archive, d / "out", "tar.gz")
        Traceback (most recent call last):
            ...
        ValueError: Refusing to extract unsafe tar member '../escaped.csv' ...

        Aliases (``AUTO`` / ``TGZ``) are not directly extractable — :func:`resolve_format`
        must collapse them first:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     safe_extract(Path(d) / "any.bin", Path(d) / "out", "auto")
        Traceback (most recent call last):
            ...
        ValueError: safe_extract: alias format 'auto' must be resolved via resolve_format(...) first.
    """
    fmt = ArchiveFormat(fmt)
    if fmt in (ArchiveFormat.AUTO, ArchiveFormat.TGZ):
        raise ValueError(
            f"safe_extract: alias format {fmt.value!r} must be resolved via resolve_format(...) first."
        )
    if not archive_path.exists():
        raise FileNotFoundError(f"safe_extract: archive does not exist: {archive_path}")
    target_dir.mkdir(parents=True, exist_ok=True)
    target_root = target_dir.resolve()
    _EXTRACTORS[fmt](archive_path, target_root)
