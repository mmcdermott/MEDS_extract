"""File-IO helpers shared across stages.

This module is the *one* place in the pipeline where file-format dispatch
(``parquet`` / ``par`` / ``csv`` / ``csv.gz``) and layout resolution
(bare file vs. sub-sharded directory) live. Stages call :func:`scan_source`
(read one or more files, concatenated) and :func:`resolve_source_files`
(find the files for a given prefix) rather than rolling their own
``scan_parquet`` / ``glob`` calls.
"""

from __future__ import annotations

import gzip
import logging
import warnings
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable

    from upath import UPath

logger = logging.getLogger(__name__)

# Supported external-table formats. No priority order — if a prefix resolves to
# more than one layout simultaneously (e.g. both a ``foo.parquet`` file and a
# ``foo/`` directory), :func:`resolve_source_files` raises.
SOURCE_FILE_EXTS = (".parquet", ".par", ".csv.gz", ".csv")


def scan_source(
    fps: Path | UPath | Iterable[Path | UPath],
    **scan_kwargs: Any,
) -> pl.LazyFrame:
    """Scan one or more source files, dispatching on extension.

    Accepts either a single path or an iterable of paths. In the multi-path case
    the resulting LazyFrames are concatenated with ``vertical_relaxed``.
    Extension-specific adjustments (parquet ``glob=False``, csv.gz via
    ``gzip.open`` + ``read_csv``, parquet ignoring ``infer_schema_length``) live
    here so callers never need to care about format.

    Examples:
        >>> from tempfile import TemporaryDirectory
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> with TemporaryDirectory() as d:
        ...     fp = Path(d) / "t.parquet"
        ...     df.write_parquet(fp)
        ...     scan_source(fp).collect()["a"].to_list()
        [1, 2, 3]

        Unsupported formats raise ``ValueError``:

        >>> scan_source(Path("t.json"))
        Traceback (most recent call last):
            ...
        ValueError: Unsupported source file type: t.json
    """
    if isinstance(fps, PurePath):
        return _scan_one(fps, **scan_kwargs)
    fps = list(fps)
    if len(fps) == 1:
        return _scan_one(fps[0], **scan_kwargs)
    return pl.concat([_scan_one(fp, **scan_kwargs) for fp in fps], how="vertical_relaxed")


def _scan_one(fp: Path | UPath, **scan_kwargs: Any) -> pl.LazyFrame:
    suffixes = "".join(fp.suffixes).lower()
    if suffixes.endswith(".csv.gz"):
        with gzip.open(fp, mode="rb") as f, warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return pl.read_csv(f, **scan_kwargs).lazy()
    if suffixes.endswith(".csv"):
        return pl.scan_csv(fp, **scan_kwargs)
    if suffixes.endswith((".parquet", ".par")):
        # glob=False: we've already resolved the exact file path, so polars must
        # treat it literally. Critical for shard_events' "[0-10).parquet" output,
        # where the filename itself contains glob metacharacters.
        scan_kwargs.pop("infer_schema_length", None)
        return pl.scan_parquet(fp, glob=False, **scan_kwargs)
    raise ValueError(f"Unsupported source file type: {fp}")


def resolve_source_files(dir: Path | UPath, prefix: str) -> list[Path | UPath]:
    """Find all source files for ``prefix`` under ``dir``.

    Two supported layouts:

    - **Sub-sharded directory**: ``{dir}/{prefix}/*.parquet`` — the output of
      ``shard_events`` and the format of user-supplied pre-subsharded data.
    - **Single bare file**: ``{dir}/{prefix}.{parquet,par,csv.gz,csv}`` — raw
      user data or subject-sharded stage output.

    Both layouts are checked. If **both** match simultaneously (e.g. a user has
    both ``labs.parquet`` and ``labs/*.parquet`` under the same directory),
    this is an ambiguity and raises ``ValueError``. If neither matches, raises
    ``FileNotFoundError``.

    ``prefix`` may contain slashes (e.g. ``hosp/patients``); that's just a
    path component, not a glob — no recursive walking happens.
    """
    matches: list[tuple[str, list[Path | UPath]]] = []

    sub_dir = dir / prefix
    try:
        is_dir = sub_dir.is_dir()
    except OSError:
        is_dir = False
    if is_dir:
        dir_fps: list[Path | UPath] = []
        for ext in SOURCE_FILE_EXTS:
            dir_fps.extend(sub_dir.glob(f"*{ext}"))
        if dir_fps:
            matches.append((f"sub-sharded directory '{prefix}/'", sorted(dir_fps)))

    for ext in SOURCE_FILE_EXTS:
        fp = dir / f"{prefix}{ext}"
        try:
            if fp.is_file():
                matches.append((f"bare file '{prefix}{ext}'", [fp]))
        except OSError:
            continue

    if len(matches) > 1:
        layouts = ", ".join(desc for desc, _ in matches)
        raise ValueError(
            f"Ambiguous source layout for prefix '{prefix}' under {dir}: "
            f"matched {layouts}. Only one layout may exist per prefix."
        )
    if not matches:
        raise FileNotFoundError(
            f"No source files found for prefix '{prefix}' under {dir}. Tried "
            f"'{prefix}/*.{{parquet,par,csv.gz,csv}}' (sub-sharded directory) "
            f"and '{prefix}.{{parquet,par,csv.gz,csv}}' (bare file)."
        )
    return matches[0][1]
