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
from upath import UPath

if TYPE_CHECKING:
    from collections.abc import Iterable

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
    ``gzip.open`` + ``read_csv``, parquet ignoring the csv-only ``infer_schema``
    / ``infer_schema_length`` kwargs) live here — and are applied **per file**,
    not per batch — so callers never need to care about format, even when a
    single prefix mixes formats across its chunk files.

    Examples:
        Scanning a single parquet file returns a LazyFrame for that file alone.

        >>> with yaml_disk('''
        ... patients.parquet:
        ...   subject_id: [1, 2, 3]
        ...   dob: ['1980-01-01', '1985-06-15', '1990-12-31']
        ... ''') as d:
        ...     scan_source(Path(d) / 'patients.parquet').collect()
        shape: (3, 2)
        ┌────────────┬────────────┐
        │ subject_id ┆ dob        │
        │ ---        ┆ ---        │
        │ i64        ┆ str        │
        ╞════════════╪════════════╡
        │ 1          ┆ 1980-01-01 │
        │ 2          ┆ 1985-06-15 │
        │ 3          ┆ 1990-12-31 │
        └────────────┴────────────┘

        Passing a list of paths concatenates them vertically. This is the
        common case downstream of ``shard_events``, where a prefix resolves
        to many row-chunk files.

        >>> with yaml_disk('''
        ... vitals/[0-2).parquet:
        ...   subject_id: [1, 1]
        ...   hr: [80, 85]
        ... vitals/[2-4).parquet:
        ...   subject_id: [2, 2]
        ...   hr: [72, 78]
        ... ''') as d:
        ...     fps = sorted((Path(d) / 'vitals').glob('*.parquet'))
        ...     scan_source(fps).collect().sort('subject_id', 'hr')
        shape: (4, 2)
        ┌────────────┬─────┐
        │ subject_id ┆ hr  │
        │ ---        ┆ --- │
        │ i64        ┆ i64 │
        ╞════════════╪═════╡
        │ 1          ┆ 80  │
        │ 1          ┆ 85  │
        │ 2          ┆ 72  │
        │ 2          ┆ 78  │
        └────────────┴─────┘

        A prefix may mix formats across its chunk files (e.g. one ``.csv`` and
        one ``.parquet`` chunk). Format dispatch — including which kwargs each
        format accepts — happens per file, so csv-only kwargs like
        ``infer_schema`` are silently dropped for the parquet chunks instead of
        crashing ``scan_parquet`` (issue #137). Mismatched dtypes unify through
        ``vertical_relaxed`` (``Int64`` + ``String`` → ``String``):

        >>> with yaml_disk('''
        ... items/a.csv: |
        ...   itemid,label
        ...   1,Heart Rate
        ... items/b.parquet:
        ...   itemid: [2]
        ...   label: [NBP systolic]
        ... ''') as d:
        ...     fps = sorted((Path(d) / 'items').glob('*'))
        ...     scan_source(fps, infer_schema=False).collect().sort('itemid')
        shape: (2, 2)
        ┌────────┬──────────────┐
        │ itemid ┆ label        │
        │ ---    ┆ ---          │
        │ str    ┆ str          │
        ╞════════╪══════════════╡
        │ 1      ┆ Heart Rate   │
        │ 2      ┆ NBP systolic │
        └────────┴──────────────┘

        Unsupported formats raise ``ValueError``:

        >>> scan_source(Path("t.json"))
        Traceback (most recent call last):
            ...
        ValueError: Unsupported source file type: t.json
    """
    # PurePath catches local pathlib.Path and local UPaths (which inherit from PurePath).
    # Cloud UPath subclasses (S3Path, GCSPath, etc.) don't inherit from PurePath — they
    # inherit from a separate CloudPath → UPath hierarchy — so we check UPath explicitly.
    if isinstance(fps, PurePath | UPath):
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
        # csv-only kwargs are dropped here, per file, so a prefix mixing csv and
        # parquet chunks scans cleanly (#137) — scan_parquet would TypeError on them.
        scan_kwargs.pop("infer_schema_length", None)
        scan_kwargs.pop("infer_schema", None)
        return pl.scan_parquet(fp, glob=False, **scan_kwargs)
    raise ValueError(f"Unsupported source file type: {fp}")


def resolve_source_files(dir: Path | UPath, prefix: str) -> list[Path | UPath]:
    """Find all source files for ``prefix`` under ``dir``.

    Two supported layouts:

    - **Sub-sharded directory**: ``{dir}/{prefix}/*.{parquet,par,csv.gz,csv}``
      — the output of ``shard_events`` and the format of user-supplied
      pre-subsharded data.
    - **Single bare file**: ``{dir}/{prefix}.{parquet,par,csv.gz,csv}`` — raw
      user data or subject-sharded stage output.

    Both layouts are checked. If **both** match simultaneously (e.g. a user
    has both ``labs.parquet`` and ``labs/*.parquet`` under the same
    directory), this is an ambiguity and raises ``ValueError``. If neither
    matches, raises ``FileNotFoundError``.

    ``prefix`` may contain slashes (e.g. ``hosp/patients``); that's just a
    path component, not a glob — no recursive walking happens.

    Examples:
        **Bare file layout** — a single file per prefix, typical of raw input
        to ``shard_events`` and subject-sharded output elsewhere:

        >>> with yaml_disk('''
        ... patients.parquet:
        ...   subject_id: [1, 2]
        ... labs.csv: |
        ...   subject_id,value
        ...   1,5.0
        ...   2,7.0
        ... ''') as d:
        ...     resolved = resolve_source_files(Path(d), "patients")
        ...     [fp.name for fp in resolved]
        ['patients.parquet']
        >>> with yaml_disk('''
        ... labs.csv: |
        ...   subject_id,value
        ...   1,5.0
        ... ''') as d:
        ...     [fp.name for fp in resolve_source_files(Path(d), "labs")]
        ['labs.csv']

        **Sub-sharded directory layout** — many chunks per prefix, typical
        output of ``shard_events``. All files under ``{prefix}/`` are
        returned sorted by name, and may mix formats:

        >>> with yaml_disk('''
        ... vitals/[0-2).parquet:
        ...   hr: [80, 85]
        ... vitals/[2-4).parquet:
        ...   hr: [72, 78]
        ... ''') as d:
        ...     resolved = resolve_source_files(Path(d), "vitals")
        ...     [fp.name for fp in resolved]
        ['[0-2).parquet', '[2-4).parquet']

        **Nested prefixes** like ``hosp/patients`` are handled naturally as
        path components — no recursive walking happens, so a nested prefix
        maps directly to its nested filesystem path:

        >>> with yaml_disk('''
        ... hosp:
        ...   patients.parquet:
        ...     subject_id: [1]
        ... ''') as d:
        ...     [str(fp.relative_to(Path(d))) for fp in resolve_source_files(Path(d), "hosp/patients")]
        ['hosp/patients.parquet']

        **Ambiguous layouts** raise — having both a bare file and a
        sub-sharded directory for the same prefix is never intentional:

        >>> with yaml_disk('''
        ... labs.parquet:
        ...   subject_id: [1]
        ... labs/extra.parquet:
        ...   subject_id: [2]
        ... ''') as d:
        ...     resolve_source_files(Path(d), "labs")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: Ambiguous source layout for prefix 'labs' ...
        matched sub-sharded directory 'labs/', bare file 'labs.parquet'...

        Similarly, having multiple bare files in different formats (e.g.
        ``labs.parquet`` AND ``labs.csv``) is ambiguous:

        >>> with yaml_disk('''
        ... labs.parquet:
        ...   subject_id: [1]
        ... labs.csv: |
        ...   subject_id
        ...   2
        ... ''') as d:
        ...     resolve_source_files(Path(d), "labs")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: Ambiguous source layout for prefix 'labs' ...
        matched bare file 'labs.parquet', bare file 'labs.csv'...

        **No match** raises ``FileNotFoundError`` listing what was tried:

        >>> with yaml_disk('other.parquet:\\n  x: [1]') as d:
        ...     resolve_source_files(Path(d), "missing")
        Traceback (most recent call last):
            ...
        FileNotFoundError: No source files found for prefix 'missing' under ...
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
