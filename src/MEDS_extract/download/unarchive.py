"""Post-fetch archive unpack for :class:`~MEDS_extract.download.source.Source`.

Two of the public ETLs (AUMCdb and HIRID) ship their raw data as a single archive that
the rest of the pipeline can't read directly. Rather than pushing archive handling into
every ETL's ``pre_MEDS.py``, we expose an optional ``unarchive`` field on each
:class:`~MEDS_extract.download.source.RemoteFile` and unpack in the base
:meth:`~MEDS_extract.download.source.Source.fetch` — so every transport picks it up for
free without bespoke plumbing.

Safety: :func:`safe_extract` rejects every member whose final path would escape the
extraction directory. Both zip-slip and tar-slip variants are covered: absolute paths,
``..`` components, and symlinks / hardlinks that point outside the target. The check
happens BEFORE any bytes are written — a malicious archive raises before any filesystem
state changes.

The tarfile branch uses Python 3.12+'s ``data_filter`` (PEP 706), which rejects special
files (devices, fifos), strips the setuid / setgid bits, and enforces the same path
constraints. Our explicit pre-pass is defense-in-depth: it surfaces a clear
:class:`ValueError` naming the offending member rather than tarfile's generic
:class:`tarfile.FilterError`.
"""

from __future__ import annotations

import logging
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

# Recognized ``unarchive`` tokens. ``"auto"`` defers the decision to
# :func:`infer_format` at extraction time.
_SUPPORTED_FORMATS = frozenset({"zip", "tar", "tar.gz", "tgz"})


def infer_format(path: Path) -> str | None:
    """Return the archive format implied by a filename, or ``None`` if unknown.

    Uses lowercase suffix matching — ``AUMCdb.ZIP`` and ``AUMCdb.zip`` both resolve
    to ``"zip"``. Returns ``None`` for any other extension; callers treat that as
    "no unpack needed" under ``unarchive="auto"`` semantics.

    Examples:
        >>> infer_format(Path("AUMCdb.zip"))
        'zip'
        >>> infer_format(Path("hirid.tar.gz"))
        'tar.gz'
        >>> infer_format(Path("raw_stage.tgz"))
        'tar.gz'
        >>> infer_format(Path("release.tar"))
        'tar'

        Case-insensitive on the extension (not the whole path):

        >>> infer_format(Path("RELEASE.TAR.GZ"))
        'tar.gz'

        Non-archive extensions return ``None`` so ``unarchive="auto"`` is a
        no-op on regular files — e.g. PhysioNet's ``LICENSE.txt`` or
        ``patients.csv.gz``:

        >>> infer_format(Path("patients.csv.gz"))  # gz compression, not a tar archive
        >>> infer_format(Path("README.md"))
        >>> infer_format(Path("plain"))
    """
    name = path.name.lower()
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return "tar.gz"
    if name.endswith(".tar"):
        return "tar"
    if name.endswith(".zip"):
        return "zip"
    return None


def resolve_format(unarchive: str, path: Path) -> str | None:
    """Normalize a user-facing ``unarchive`` token to a canonical format.

    ``"auto"`` delegates to :func:`infer_format` based on ``path``'s extension. Any
    other value must be one of the supported formats (``"zip"``, ``"tar"``,
    ``"tar.gz"``, ``"tgz"``). ``"tgz"`` is folded to ``"tar.gz"``. Returns ``None``
    when ``"auto"`` is requested but no known extension matches — the caller treats
    that as "leave the file alone".

    Examples:
        >>> resolve_format("zip", Path("x.zip"))
        'zip'
        >>> resolve_format("tgz", Path("x.tgz"))  # folded to canonical 'tar.gz'
        'tar.gz'
        >>> resolve_format("auto", Path("AUMCdb.zip"))
        'zip'
        >>> resolve_format("auto", Path("hirid.tar.gz"))
        'tar.gz'

        ``"auto"`` on a non-archive returns ``None`` (caller leaves the file alone):

        >>> resolve_format("auto", Path("patients.csv.gz")) is None
        True

        Unknown explicit formats raise — catching typos early beats a silent skip:

        >>> resolve_format("rar", Path("x.rar"))
        Traceback (most recent call last):
            ...
        ValueError: Unsupported unarchive format 'rar'. Supported: 'auto', 'tar', 'tar.gz', 'tgz', 'zip'.
    """
    if unarchive == "auto":
        return infer_format(path)
    if unarchive == "tgz":
        return "tar.gz"
    if unarchive not in _SUPPORTED_FORMATS:
        supported = ", ".join(f"'{s}'" for s in sorted(_SUPPORTED_FORMATS | {"auto"}))
        raise ValueError(f"Unsupported unarchive format {unarchive!r}. Supported: {supported}.")
    return unarchive


def _is_safe_member(member_name: str, target_root: Path) -> bool:
    """Return ``True`` iff extracting ``member_name`` lands strictly inside ``target_root``.

    Rejects absolute paths, ``..`` escapes, and any resolved path that isn't a
    descendant of the target. Archive entries always use forward-slash separators per
    format spec, so ``PurePosixPath`` is the right parser regardless of host OS.

    Examples:
        >>> with tempfile.TemporaryDirectory() as d:
        ...     root = Path(d).resolve()
        ...     _is_safe_member("sub/file.csv", root)
        ...     _is_safe_member("./ok.csv", root)
        True
        True
        >>> with tempfile.TemporaryDirectory() as d:
        ...     root = Path(d).resolve()
        ...     _is_safe_member("/etc/passwd", root)
        ...     _is_safe_member("../evil.csv", root)
        ...     _is_safe_member("sub/../../evil.csv", root)
        False
        False
        False
    """
    pure = PurePosixPath(member_name)
    if pure.is_absolute():
        return False
    resolved = (target_root / Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(target_root)
    except ValueError:
        return False
    return True


def _iter_tar_members(tf: tarfile.TarFile) -> Iterable[tarfile.TarInfo]:
    """Yield tar members, fast-failing on unsupported kinds (devices, fifos, etc.).

    tarfile's ``data_filter`` would do this at extraction time, but surfacing the
    rejection during the pre-pass gives a clearer error ("archive X contains unsafe
    member Y") than a mid-extraction :class:`tarfile.FilterError`.
    """
    for m in tf:
        if m.isdev() or m.isfifo():
            raise ValueError(
                f"Refusing to extract unsafe tar member {m.name!r} (type={m.type!r}): "
                "device and fifo entries are not supported."
            )
        yield m


def safe_extract(archive_path: Path, target_dir: Path, fmt: str) -> None:
    """Extract ``archive_path`` into ``target_dir`` with path-traversal guards.

    All member paths are validated before any bytes are written — a traversal attempt
    raises :class:`ValueError` while the target directory is still clean. ``fmt``
    must be one of ``"zip"``, ``"tar"``, or ``"tar.gz"``; ``"auto"`` / ``"tgz"``
    should be normalized via :func:`resolve_format` first.

    Args:
        archive_path: The archive to unpack. Must exist.
        target_dir: Destination directory. Created if absent.
        fmt: Canonical format string. ``"tar.gz"`` covers both ``.tar.gz`` and
            ``.tgz``.

    Raises:
        ValueError: On unsupported ``fmt``, on a member path that escapes
            ``target_dir`` (zip-slip / tar-slip), or on an unsupported tar entry
            type (device, fifo).
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
        ...     safe_extract(archive, target, "tar.gz")
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

        Unknown format raises directly:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     safe_extract(Path(d) / "any.bin", Path(d) / "out", "rar")
        Traceback (most recent call last):
            ...
        ValueError: safe_extract: unsupported fmt 'rar'; expected one of 'tar', 'tar.gz', 'zip'.
    """
    if fmt not in {"zip", "tar", "tar.gz"}:
        raise ValueError(f"safe_extract: unsupported fmt {fmt!r}; expected one of 'tar', 'tar.gz', 'zip'.")
    if not archive_path.exists():
        raise FileNotFoundError(f"safe_extract: archive does not exist: {archive_path}")
    target_dir.mkdir(parents=True, exist_ok=True)
    target_root = target_dir.resolve()

    if fmt == "zip":
        with zipfile.ZipFile(archive_path) as zf:
            for name in zf.namelist():
                if not _is_safe_member(name, target_root):
                    raise ValueError(
                        f"Refusing to extract unsafe zip member {name!r} from {archive_path}: "
                        "member path would land outside the extraction directory."
                    )
            zf.extractall(target_root)
        return

    # ``"r:*"`` autodetects both plain tar and tar.gz; we pass the user-facing ``fmt``
    # through the validation above anyway, so the mode arg is really just for readability.
    mode = "r:gz" if fmt == "tar.gz" else "r:"
    with tarfile.open(archive_path, mode=mode) as tf:
        members = list(_iter_tar_members(tf))
        for m in members:
            if not _is_safe_member(m.name, target_root):
                raise ValueError(
                    f"Refusing to extract unsafe tar member {m.name!r} from {archive_path}: "
                    "member path would land outside the extraction directory."
                )
            # A symlink's link target is a second escape vector: a safe member name
            # ``good.txt`` pointing at ``../../etc/passwd`` would bypass the name check.
            if (m.issym() or m.islnk()) and not _is_safe_member(m.linkname, target_root):
                raise ValueError(
                    f"Refusing to extract unsafe tar member {m.name!r} from {archive_path}: "
                    f"link target {m.linkname!r} would land outside the extraction directory."
                )
        # PEP 706's ``data_filter`` strips setuid / setgid / sticky bits, rejects
        # special files, and re-applies the same path-traversal check we just did.
        # Defense-in-depth: if our pre-pass ever regresses, ``data_filter`` catches it.
        tf.extractall(target_root, members=members, filter="data")
