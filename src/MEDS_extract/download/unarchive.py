"""Post-fetch archive unpack for :class:`~MEDS_extract.download.source.Source`.

Two of the public ETLs (AUMCdb and HIRID) ship their raw data as a single archive that
the rest of the pipeline can't read directly. Rather than pushing archive handling into
every ETL's ``pre_MEDS.py``, we expose an optional ``unarchive`` field on each
:class:`~MEDS_extract.download.source.RemoteFile` and unpack in the post-fetch hook
:meth:`~MEDS_extract.download.source.Source._maybe_unarchive` (invoked from
:meth:`~MEDS_extract.download.source.Source._fetch_one` whenever bytes newly land at a
dest) — so every transport picks it up for free without bespoke plumbing.

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

Extracting one sub-directory (``archive_subdir``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Project-level bundles rarely have the flat layout an ETL wants. PhysioNet's
``get-zip`` archives, for example, always wrap their content in a
``<project>-<version>/`` top-level directory, and the MIMIC-IV OMOP demo bundle puts
the tables one level further down (``<project>/1_omop_data_csv/*.csv``) next to
folders no ETL reads (``2_achilles_json/``, ``3_data_quality_dashboard_files/``).
:func:`safe_extract`'s ``archive_subdir`` selects one directory of the archive and
writes its members *relative to it*, so ``<project>/1_omop_data_csv/person.csv``
lands as ``person.csv``.

Matching rule (see :func:`_match_subdir`): ``archive_subdir`` is a forward-slash
relative path whose components must appear as a contiguous run of a member's path
components; the shallowest such run wins, and everything before it is the member's
*anchor*. This lets ``1_omop_data_csv`` match under any wrapper directory without the
caller having to know (or pin) the wrapper's versioned name. Two guards keep the
rule from being too clever:

- every selected member must share **one** anchor — if the subdir name occurs under
  two different parents (``a/data/…`` and ``b/data/…``) the archive is ambiguous and
  extraction raises rather than silently merging two trees;
- zero selected members raises — a typo'd ``archive_subdir`` must not "succeed" by
  extracting nothing.

Alternatives considered: matching ``archive_subdir`` only at the archive root
(forces callers to spell out versioned wrapper names like
``mimic-iv-demo-omop-0.9/1_omop_data_csv``, which then break on every release), and
a generic ``strip_components=N`` (tar's ``--strip-components``), which cannot
*select* a subtree and so would still extract the unneeded sibling folders.

Path safety is unchanged by the rewrite: every member's *original* name is
validated exactly as before (so a hostile member anywhere in the archive still
rejects the whole archive), and every selected member's *rewritten* name — plus its
rewritten link target, for tar links — is validated again against the target.
"""

from __future__ import annotations

import copy
import logging
import posixpath
import tarfile
import zipfile
from enum import StrEnum
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from ._paths import resolve_contained

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

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
    """Internal helper: ``True`` iff ``member_name`` resolves strictly inside ``target_root``.

    Archive members are POSIX-shaped by format, hence ``posix_member=True``. The
    containment check itself is shared with the fetch path's ``rel_path`` validation —
    see :mod:`MEDS_extract.download._paths` for why there is exactly one of it.
    """
    _, contained = resolve_contained(target_root, member_name, posix_member=True)
    return contained


def _match_subdir(member_name: str, subdir_parts: tuple[str, ...]) -> tuple[str, str] | None:
    """Locate ``subdir_parts`` inside ``member_name``; return ``(anchor, remainder)`` or ``None``.

    ``subdir_parts`` matches the shallowest contiguous run of ``member_name``'s path
    components. ``anchor`` is the (possibly empty) path before the match and
    ``remainder`` the path after it — the member's location relative to the selected
    sub-directory. A member that *is* the sub-directory (its directory entry, or a
    file sharing the name) has no remainder and is not selected. Components are
    compared whole, never as string prefixes, and ``.`` / trailing-slash noise is
    dropped by :class:`~pathlib.PurePosixPath` first.

    Examples:
        >>> _match_subdir("proj-0.9/1_omop_data_csv/person.csv", ("1_omop_data_csv",))
        ('proj-0.9', 'person.csv')
        >>> _match_subdir("1_omop_data_csv/sub/x.csv", ("1_omop_data_csv",))  # at the root
        ('', 'sub/x.csv')
        >>> _match_subdir("./w/a/b/c.csv", ("a", "b"))  # multi-component subdir
        ('w', 'c.csv')

        The shallowest occurrence wins, so a nested same-named folder stays nested:

        >>> _match_subdir("w/data/x/data/y.csv", ("data",))
        ('w', 'x/data/y.csv')

        Whole-component matching only, and the directory entry itself is not selected:

        >>> _match_subdir("w/1_omop_data_csv_old/x.csv", ("1_omop_data_csv",)) is None
        True
        >>> _match_subdir("w/1_omop_data_csv/", ("1_omop_data_csv",)) is None
        True
    """
    parts = PurePosixPath(member_name).parts
    n = len(subdir_parts)
    for i in range(len(parts) - n + 1):
        if parts[i : i + n] == subdir_parts:
            rest = parts[i + n :]
            if not rest:
                return None
            return "/".join(parts[:i]), "/".join(rest)
    return None


def _select_members(
    names: list[str], archive_subdir: str | None, *, archive_path: Path
) -> tuple[dict[str, str], str | None]:
    """Map each selected member name to its archive-relative output name.

    Without ``archive_subdir`` every member is selected under its own name. With it,
    only members under the sub-directory are selected and each maps to its path
    relative to it (see :func:`_match_subdir`). Returns ``(mapping, anchor)`` where
    ``anchor`` is the shared prefix the sub-directory was found under (``None``
    without ``archive_subdir``) — tar hard links need it to rewrite their targets.

    Raises:
        ValueError: If ``archive_subdir`` selects no members, or occurs under more
            than one anchor (ambiguous — two different trees would be merged).

    Examples:
        >>> names = ["p/", "p/csv/", "p/csv/a.csv", "p/csv/s/b.csv", "p/json/c.json"]
        >>> _select_members(names, "csv", archive_path=Path("x.zip"))
        ({'p/csv/a.csv': 'a.csv', 'p/csv/s/b.csv': 's/b.csv'}, 'p')
        >>> _select_members(["a.csv"], None, archive_path=Path("x.zip"))
        ({'a.csv': 'a.csv'}, None)
        >>> _select_members(names, "nope", archive_path=Path("x.zip"))
        Traceback (most recent call last):
            ...
        ValueError: archive_subdir 'nope' matched no members of x.zip; refusing to extract nothing. ...
        >>> _select_members(["a/data/x.csv", "b/data/y.csv"], "data", archive_path=Path("x.zip"))
        Traceback (most recent call last):
            ...
        ValueError: archive_subdir 'data' is ambiguous in x.zip: found under ['a', 'b']. Qualify ...
    """
    if archive_subdir is None:
        return {name: name for name in names}, None
    subdir_parts = tuple(archive_subdir.split("/"))
    mapping: dict[str, str] = {}
    anchors: set[str] = set()
    for name in names:
        match = _match_subdir(name, subdir_parts)
        if match is None:
            continue
        anchor, remainder = match
        anchors.add(anchor)
        mapping[name] = remainder
    if not mapping:
        raise ValueError(
            f"archive_subdir {archive_subdir!r} matched no members of {archive_path}; refusing to "
            "extract nothing. Check the spelling against the archive's internal layout."
        )
    if len(anchors) > 1:
        raise ValueError(
            f"archive_subdir {archive_subdir!r} is ambiguous in {archive_path}: found under "
            f"{sorted(anchors)!r}. Qualify it with more leading path components."
        )
    return mapping, anchors.pop()


def _extract_zip(archive_path: Path, target_root: Path, archive_subdir: str | None = None) -> None:
    """Extract a ``.zip`` archive into ``target_root`` after validating every member.

    Members are extracted one at a time (equivalent to ``extractall``) so that, under
    ``archive_subdir``, each selected member can be written under its rewritten name:
    a shallow copy of its :class:`zipfile.ZipInfo` gets the new ``filename`` while
    ``orig_filename`` — which :meth:`zipfile.ZipFile.open` checks against the local
    header — is left intact.
    """
    with zipfile.ZipFile(archive_path) as zf:
        infos = zf.infolist()
        for info in infos:
            _validate_member(info.filename, target_root, archive_path=archive_path, fmt_label="zip")
        mapping, _ = _select_members([i.filename for i in infos], archive_subdir, archive_path=archive_path)
        for info in infos:
            new_name = mapping.get(info.filename)
            if new_name is None:
                continue
            if new_name != info.filename:
                _validate_member(new_name, target_root, archive_path=archive_path, fmt_label="zip")
                info = copy.copy(info)
                # ``ZipInfo.is_dir`` keys on the trailing slash, which the rewrite drops.
                info.filename = new_name + "/" if info.is_dir() else new_name
            zf.extract(info, target_root)


def _extract_tar_gz(archive_path: Path, target_root: Path, archive_subdir: str | None = None) -> None:
    """Extract a gzip-compressed tar archive."""
    _extract_tar_with_mode(archive_path, target_root, mode="r:gz", archive_subdir=archive_subdir)


def _extract_tar(archive_path: Path, target_root: Path, archive_subdir: str | None = None) -> None:
    """Extract an uncompressed tar archive."""
    _extract_tar_with_mode(archive_path, target_root, mode="r:", archive_subdir=archive_subdir)


def _extract_tar_with_mode(
    archive_path: Path, target_root: Path, *, mode: str, archive_subdir: str | None = None
) -> None:
    """Tar extraction body shared between :func:`_extract_tar` and :func:`_extract_tar_gz`.

    Validates member names AND link targets (a safe-named tar entry can still escape
    via a symlink whose ``linkname`` walks outside the target). PEP 706's
    ``data_filter`` is applied at ``extractall`` as belt-and-suspenders defense-in-depth.

    Under ``archive_subdir`` each selected member is re-named via
    :meth:`tarfile.TarInfo.replace` and validated again under its new name. Link
    targets need format-specific care: a symlink's ``linkname`` is relative to the
    link's own directory, so it moves with the subtree unchanged and is validated
    from the link's rewritten location; a hard link's ``linkname`` is relative to the
    archive root, so it is rewritten through the same anchor — and a hard link whose
    target lies outside ``archive_subdir`` is rejected, since that target is never
    extracted.
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
        if archive_subdir is not None:
            members = _rewrite_tar_members(members, archive_subdir, target_root, archive_path=archive_path)
        # PEP 706's ``data_filter`` strips setuid / setgid / sticky bits, rejects
        # special files, and re-applies the same path-traversal check we just did.
        # Defense-in-depth: if our pre-pass ever regresses, ``data_filter`` catches it.
        tf.extractall(target_root, members=members, filter="data")


def _rewrite_tar_members(
    members: list[tarfile.TarInfo], archive_subdir: str, target_root: Path, *, archive_path: Path
) -> list[tarfile.TarInfo]:
    """Select and re-name the tar members under ``archive_subdir`` (see :func:`_extract_tar_with_mode`)."""
    mapping, anchor = _select_members([m.name for m in members], archive_subdir, archive_path=archive_path)
    subdir_parts = tuple(archive_subdir.split("/"))
    out: list[tarfile.TarInfo] = []
    for m in members:
        new_name = mapping.get(m.name)
        if new_name is None:
            continue
        linkname = m.linkname
        check_target: str | None = None
        if m.islnk():
            match = _match_subdir(m.linkname, subdir_parts)
            if match is None or match[0] != anchor:
                raise ValueError(
                    f"Refusing to extract tar member {m.name!r} from {archive_path}: hard link target "
                    f"{m.linkname!r} lies outside archive_subdir {archive_subdir!r}."
                )
            linkname = check_target = match[1]
        elif m.issym():
            check_target = posixpath.join(posixpath.dirname(new_name), m.linkname)
        _validate_member(
            new_name, target_root, archive_path=archive_path, fmt_label="tar", link_target=check_target
        )
        out.append(m.replace(name=new_name, linkname=linkname, deep=False))
    return out


# Format → extractor dispatch. Adding a new format = add an enum value above + a
# ``_extract_<fmt>`` function + an entry here. ``AUTO`` and ``TGZ`` are not in this
# table on purpose — they're aliases that ``resolve_format`` collapses to a canonical
# format before dispatch. Every extractor takes ``(archive_path, target_root,
# archive_subdir)``.
_EXTRACTORS: dict[ArchiveFormat, Callable[[Path, Path, str | None], None]] = {
    ArchiveFormat.ZIP: _extract_zip,
    ArchiveFormat.TAR: _extract_tar,
    ArchiveFormat.TAR_GZ: _extract_tar_gz,
}


def safe_extract(
    archive_path: Path,
    target_dir: Path,
    fmt: str | ArchiveFormat,
    *,
    archive_subdir: str | None = None,
) -> None:
    """Extract ``archive_path`` into ``target_dir`` with path-traversal guards.

    All member paths are validated before any bytes are written — a traversal attempt
    raises :class:`ValueError` while the target directory is still clean. ``fmt``
    must be one of :attr:`ArchiveFormat.ZIP`, :attr:`~ArchiveFormat.TAR`, or
    :attr:`~ArchiveFormat.TAR_GZ`; ``AUTO`` / ``TGZ`` should be normalized via
    :func:`resolve_format` first.

    This function writes members straight into ``target_dir`` and has no notion of
    completion: an exception part-way through leaves a partial tree. Callers that
    need all-or-nothing semantics extract into a scratch directory and move the
    result into place, as :meth:`~MEDS_extract.download.source.Source._maybe_unarchive`
    does.

    Args:
        archive_path: The archive to unpack. Must exist.
        target_dir: Destination directory. Created if absent.
        fmt: Canonical archive format (string or :class:`ArchiveFormat`).
        archive_subdir: Optional archive-relative directory (forward slashes) to
            extract on its own; its members land relative to it, directly under
            ``target_dir``. See the module docstring for the matching rule.

    Raises:
        ValueError: On unsupported / alias ``fmt`` (``AUTO``, ``TGZ`` not handled
            here), on a member path that escapes ``target_dir`` (zip-slip /
            tar-slip), on an unsupported tar entry type (device, fifo), or when
            ``archive_subdir`` selects no members or is ambiguous.
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

        ``archive_subdir`` extracts one directory, flattened, from a wrapped bundle —
        the PhysioNet ``get-zip`` shape. Sibling folders are not extracted:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     archive = d / "bundle.zip"
        ...     with _zipfile.ZipFile(archive, "w") as zf:
        ...         zf.writestr("proj-0.9/1_omop_data_csv/person.csv", "person_id\\n1")
        ...         zf.writestr("proj-0.9/1_omop_data_csv/sub/visit.csv", "visit_id\\n2")
        ...         zf.writestr("proj-0.9/2_achilles_json/report.json", "{}")
        ...     target = d / "out"
        ...     safe_extract(archive, target, "zip", archive_subdir="1_omop_data_csv")
        ...     sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file())
        ['person.csv', 'sub/visit.csv']

        A sub-directory that matches nothing is an error, not a silent no-op:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     archive = d / "bundle.zip"
        ...     with _zipfile.ZipFile(archive, "w") as zf:
        ...         zf.writestr("proj/a.csv", "x")
        ...     safe_extract(archive, d / "out", "zip", archive_subdir="missing")
        Traceback (most recent call last):
            ...
        ValueError: archive_subdir 'missing' matched no members of ...

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
    _EXTRACTORS[fmt](archive_path, target_root, archive_subdir)
