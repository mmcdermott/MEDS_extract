"""The one filesystem containment check the download layer trusts.

Two places need to answer "does this relative path, resolved on a real filesystem,
stay inside this root?" — :meth:`Source._resolve_dest` for a manifest's
``rel_path``, and :func:`~MEDS_extract.download.unarchive.safe_extract` for an
archive member. Both are security boundaries against the same class of escape
(``..`` segments, absolute paths, symlinks inside the root), so they resolve to one
implementation here rather than two that can drift apart.

They differ only in path *dialect* and in how they report a failure, which is why
this returns a verdict instead of raising: archive members are always POSIX-shaped
(a zip's ``a/b.txt`` means that on Windows too), and each caller words its own error.

Deliberately NOT deduped: ``RemoteFile.__post_init__``'s check, which inspects the
manifest string before any filesystem exists (rejecting backslashes, ``..``, and
absolute paths at construction). That one is a validation-time string check; this is
the fetch-time filesystem check that catches escapes only a real directory can
produce.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath


def resolve_contained(root: Path, rel_path: str, *, posix_member: bool = False) -> tuple[Path, bool]:
    """Resolve ``rel_path`` beneath ``root``; report where it landed and whether that is inside.

    Args:
        root: The directory the result must stay within. Resolved before comparison,
            so a symlinked root compares by its real location.
        rel_path: The relative path to place under ``root``.
        posix_member: Treat ``rel_path`` as an archive member — always POSIX-separated,
            regardless of the host platform — rather than a native path.

    Returns:
        ``(resolved, is_contained)``. ``resolved`` is where the path lands (useful in
        the caller's error message); ``is_contained`` is ``False`` for an absolute
        input or one that escapes ``root``.

    Examples:
        >>> with tempfile.TemporaryDirectory() as d:
        ...     root = Path(d)
        ...     _, ok_plain = resolve_contained(root, "a/b.txt")
        ...     _, ok_dotdot = resolve_contained(root, "../escape.txt")
        ...     _, ok_abs = resolve_contained(root, "/etc/passwd")
        ...     _, ok_inner = resolve_contained(root, "sub/../still_inside.txt")
        ...     (ok_plain, ok_dotdot, ok_abs, ok_inner)
        (True, False, False, True)

        The resolved path comes back even when the verdict is negative, so callers can
        name the offending location:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     resolved, ok = resolve_contained(Path(d), "../sneaky.txt")
        ...     ok, resolved.name
        (False, 'sneaky.txt')

        ``posix_member`` interprets separators the archive's way rather than the
        platform's, which is what makes a zip member portable:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     resolved, ok = resolve_contained(Path(d), "nested/dir/f.txt", posix_member=True)
        ...     ok, resolved.relative_to(Path(d).resolve()).as_posix()
        (True, 'nested/dir/f.txt')
    """
    root = Path(root).resolve()
    pure = PurePosixPath(rel_path) if posix_member else Path(rel_path)
    if pure.is_absolute():
        return Path(rel_path), False
    # ``Path(*parts)`` re-joins POSIX member segments with the host separator.
    relative = Path(*pure.parts) if posix_member else pure
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return resolved, False
    return resolved, True
