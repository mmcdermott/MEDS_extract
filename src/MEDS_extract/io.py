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
import warnings
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any

import polars as pl
from upath import UPath

if TYPE_CHECKING:
    from collections.abc import Iterable

# Supported external-table formats. No priority order — if a prefix resolves to
# more than one layout simultaneously (e.g. both a ``foo.parquet`` file and a
# ``foo/`` directory), :func:`resolve_source_files` raises.
SOURCE_FILE_EXTS = (".parquet", ".par", ".csv.gz", ".csv")

# Provenance anchor columns stamped onto every ingested row at raw-data ingestion:
# the 0-based row index within the original source file and the input-dir-relative
# path of that file. They live here because both the ingest stage and ``config``
# need them.
ROW_IDX_NAME = "__row_idx__"
SOURCE_FILE_COL = "__source_file__"


def scan_source(
    fps: Path | UPath | Iterable[Path | UPath],
    **scan_kwargs: Any,
) -> pl.LazyFrame:
    """Scan one or more source files, dispatching on extension.

    Accepts either a single path or an iterable of paths. In the multi-path case
    the resulting LazyFrames are concatenated with ``vertical_relaxed``.
    Extension-specific adjustments (parquet ``glob=False``, csv.gz via
    ``gzip.open`` + ``read_csv``, parquet ignoring the csv-only
    ``infer_schema_length`` kwarg) live here — and are applied **per file**,
    not per batch — so callers never need to care about format. Multi-file
    sources must be format-homogeneous (csv-family or parquet-family, not a
    mix); heterogeneity across *separate* single-file scans is fine.

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
        common case for a pre-sharded source table, where a prefix resolves
        to many files.

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

        A multi-file scan must be format-homogeneous — mixing csv-family and
        parquet-family chunks in one source is a config error rather than a silent
        dtype coercion (typed parquet + all-String csv would otherwise unify through
        ``vertical_relaxed``):

        >>> with yaml_disk('''
        ... items/a.csv: |
        ...   itemid,label
        ...   1,Heart Rate
        ... items/b.parquet:
        ...   itemid: [2]
        ...   label: [NBP systolic]
        ... ''') as d:
        ...     fps = sorted((Path(d) / 'items').glob('*'))
        ...     scan_source(fps, infer_schema=False)
        Traceback (most recent call last):
            ...
        ValueError: Cannot scan a mix of csv- and parquet-family files as one source: ...

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
    # A multi-file scan must be format-homogeneous (csv-family or parquet-family, not a
    # mix): the families take different reader options, and silently concatenating typed
    # parquet with all-String csv would coerce dtypes through ``vertical_relaxed``.
    families = {_format_family(fp) for fp in fps}
    if len(families) > 1:
        raise ValueError(
            f"Cannot scan a mix of csv- and parquet-family files as one source: {sorted(map(str, fps))}. "
            "Convert the chunks to a single format."
        )
    return pl.concat([_scan_one(fp, **scan_kwargs) for fp in fps], how="vertical_relaxed")


def _format_family(fp: Path | UPath) -> str:
    suffixes = "".join(fp.suffixes).lower()
    if suffixes.endswith((".csv.gz", ".csv")):
        return "csv"
    if suffixes.endswith((".parquet", ".par")):
        return "parquet"
    raise ValueError(f"Unsupported source file type: {fp}")


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
        # treat it literally — a source filename may contain glob metacharacters.
        # ``infer_schema_length`` is ignored for parquet, so a caller can pass one kwargs
        # set while scanning files individually, whatever each file's format. (Safe
        # because multi-file scans are format-homogeneous — enforced in scan_source — so
        # this never silently mixes typed and inferred chunks of one source. csv-only
        # ``infer_schema`` needs no such tolerance: its sole caller chooses kwargs per
        # resolved prefix and passes it to csv-family prefixes only.)
        scan_kwargs.pop("infer_schema_length", None)
        return pl.scan_parquet(fp, glob=False, **scan_kwargs)
    raise ValueError(f"Unsupported source file type: {fp}")


def resolve_source_files(dir: Path | UPath, prefix: str) -> list[Path | UPath]:
    """Find all source files for ``prefix`` under ``dir``.

    Two supported layouts:

    - **Sub-sharded directory**: ``{dir}/{prefix}/*.{parquet,par,csv.gz,csv}``
      — the format of user-supplied pre-sharded data, preserved through
      ``convert_to_parquet``.
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
        to ``convert_to_parquet`` and subject-sharded output elsewhere:

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
        layout of a pre-sharded source table. All non-hidden files under
        ``{prefix}/`` are returned sorted by name; the chunks must share one
        format family (csv-family or parquet-family — ``scan_source`` rejects
        a mix):

        >>> with yaml_disk('''
        ... vitals/[0-2).parquet:
        ...   hr: [80, 85]
        ... vitals/[2-4).parquet:
        ...   hr: [72, 78]
        ... ''') as d:
        ...     resolved = resolve_source_files(Path(d), "vitals")
        ...     [fp.name for fp in resolved]
        ['[0-2).parquet', '[2-4).parquet']

        Hidden files are never source data, so dotfile debris in a sub-sharded
        directory (e.g. a stranded writer intermediate) is ignored:

        >>> with yaml_disk('''
        ... vitals/chunk_0.parquet:
        ...   hr: [80, 85]
        ... ''') as d:
        ...     (Path(d) / "vitals" / ".x.parquet.strings.tmp.parquet").touch()
        ...     [fp.name for fp in resolve_source_files(Path(d), "vitals")]
        ['chunk_0.parquet']

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
            # Hidden files are never source data (e.g. in-progress writer intermediates),
            # and pathlib globs would otherwise match them.
            dir_fps.extend(fp for fp in sub_dir.glob(f"*{ext}") if not fp.name.startswith("."))
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


# ── CSV → parquet conversion ─────────────────────────────────────────

# The field-level acceptor regexes of polars' own CSV type inference, copied verbatim
# from polars 1.39.3 (``crates/polars-io/src/utils/other.rs``). Full-file inference
# (``infer_schema_length=None``) classifies every non-null field with these — booleans
# case-insensitively, integers with an optional ``-`` but never ``+``, floats requiring
# a dot or exponent (plus signed ``inf``/``NaN`` in exactly those casings) — and
# unifies per column: all-int → Int64, int/float mix → Float64, all-bool → Boolean,
# anything else → String. ``str.contains`` runs the same Rust regex engine, so the
# semantics (anchoring, no whitespace trimming, Unicode ``\d``) carry over unchanged.
_CSV_BOOL_RE = r"^(?i:true|false)$"
_CSV_TRUE_RE = r"^(?i:true)$"
_CSV_INT_RE = r"^-?(\d+)$"
_CSV_FLOAT_RE = r"^[-+]?((\d*\.\d+)([eE][-+]?\d+)?|inf|NaN|(\d+)[eE][-+]?\d+|\d+\.)$"

# Polars' CSV type inference, reproduced over an already-materialized String column.
# Ordered widest-last: the first entry every non-null value satisfies wins, matching
# what ``infer_schema_length=None`` decides — a round-trip guarantee (same schema, same
# values) enforced by the property test in ``tests/test_convert_to_parquet.py`` for
# every CSV polars itself can read. Where polars' full-file read *hard-errors* on a
# column its regexes classify as typed but whose values its parsers then reject — an
# integer-shaped column overflowing Int64 is the practical case (exotic ones: Unicode
# near-misses like non-ASCII ``\d`` digits or a long-s "false") — this ladder
# instead degrades gracefully: lossy Float64 for the overflow case. That lenience is
# the sole deliberate deviation, exercised deterministically in the same test file.
_INFERENCE_LADDER: tuple[tuple[pl.DataType, str], ...] = (
    (pl.Int64, "n_int"),
    (pl.Float64, "n_float"),
    (pl.Boolean, "n_bool"),
)


def infer_column_dtypes(lf: pl.LazyFrame) -> dict[str, pl.DataType]:
    """Infer each String column's dtype the way full-file CSV inference would.

    This is the middle pass of :func:`convert_csv_to_parquet`. It runs over a
    *columnar* frame (the all-String parquet written by pass 1), so deciding a
    column's type is an aggregate over one column rather than a scan of the whole
    table — which is what lets full-file-accurate inference happen without the
    file in memory. ``polars``' own ``infer_schema_length=None`` cannot do this:
    it materializes the CSV to decide.

    A column is given the narrowest type that EVERY non-null value satisfies, judged
    by polars' own field-acceptor regexes (see ``_CSV_*_RE`` above), and an all-null
    column stays ``String`` — matching what polars infers for a CSV column that is
    empty in every row (a real case: an always-empty ``dod``).

    Non-String columns pass through unchanged, so this is a no-op on frames that
    already carry types.

    Examples:
        >>> lf = pl.LazyFrame({
        ...     "ints": ["1", "2", None],
        ...     "floats": ["1.5", "2", None],
        ...     "bools": ["true", "false", None],
        ...     "strs": ["1", "abc", None],
        ...     "empty": [None, None, None],
        ... }, schema={c: pl.String for c in ("ints", "floats", "bools", "strs", "empty")})
        >>> infer_column_dtypes(lf)
        {'ints': Int64, 'floats': Float64, 'bools': Boolean, 'strs': String, 'empty': String}

        The acceptors are polars', not Python's: booleans match in any casing, while a
        ``+``-prefixed integer — which ``int()`` (and a String→Int64 cast) would take —
        keeps its column String, exactly as CSV inference leaves it:

        >>> lf = pl.LazyFrame({
        ...     "bools": ["True", "FALSE", "tRuE"],
        ...     "plus": ["+1", "2"] + [None],
        ... }, schema={"bools": pl.String, "plus": pl.String})
        >>> infer_column_dtypes(lf)
        {'bools': Boolean, 'plus': String}

        A single unparsable value keeps the whole column String — inference is over
        every row, never a sample, so there is no mid-file "type flip" to fear:

        >>> lf = pl.LazyFrame({"mostly_int": ["1"] * 999 + ["NOT_A_NUMBER"]})
        >>> infer_column_dtypes(lf)
        {'mostly_int': String}

        The one deliberate divergence from ``infer_schema_length=None``: an
        integer-shaped column that overflows Int64 makes polars' full-file read
        hard-error, while this ladder degrades it to (lossy) Float64:

        >>> infer_column_dtypes(pl.LazyFrame({"big": ["9223372036854775808"]}))
        {'big': Float64}
    """
    schema = lf.collect_schema()
    str_cols = [c for c, t in schema.items() if t == pl.String]
    if not str_cols:
        return dict(schema)

    # One query PER COLUMN, not one query over all of them. Parquet is columnar, so a
    # single-column probe reads only that column — whereas probing every column at once
    # holds them all at peak. Measured on a 1 GB table: 1049 MB per-column vs 3197 MB
    # all-at-once, for ~1s more wall time.
    counts: dict[str, dict[str, int]] = {}
    for c in str_cols:
        col = pl.col(c)
        int_shaped = col.str.contains(_CSV_INT_RE)
        counts[c] = (
            lf.select(
                n=col.is_not_null().sum(),
                # Int64: integer-shaped AND in-range. A failed range check here is what
                # lets an Int64-overflowing column fall through to Float64 — the
                # documented lenient stand-in for polars' full-read hard error.
                n_int=(int_shaped & col.cast(pl.Int64, strict=False).is_not_null()).sum(),
                # Float64 additionally admits integer-shaped values: a mixed int/float
                # column unifies to Float64 under polars, and the overflow case above
                # lands here. The castability guard keeps regex-accepted but
                # unparsable text (e.g. non-ASCII ``\d`` digits) from qualifying.
                n_float=(
                    (col.str.contains(_CSV_FLOAT_RE) | int_shaped)
                    & col.cast(pl.Float64, strict=False).is_not_null()
                ).sum(),
                n_bool=col.str.contains(_CSV_BOOL_RE).sum(),
            )
            .collect()
            .row(0, named=True)
        )

    out: dict[str, pl.DataType] = {}
    for c, t in schema.items():
        if t != pl.String:
            out[c] = t
            continue
        n = counts[c]["n"]
        # An all-null column carries no evidence for any type; polars leaves it String.
        out[c] = next(
            (dt for dt, key in _INFERENCE_LADDER if n and counts[c][key] == n),
            pl.String,
        )
    return out


def _decode_expr(c: str, t: pl.DataType) -> pl.Expr:
    """Decode String column ``c`` to its inferred dtype ``t``, as the CSV reader would.

    Polars does not support String→Boolean casts at all (#199), so Boolean columns are
    decoded by matching the reader's case-insensitive ``true`` token instead —
    inference already guaranteed every non-null value is some casing of true/false,
    and ``str.contains`` preserves nulls. Numeric columns cast directly;
    ``strict=False`` never actually nulls a value here, because the inference probes
    only award a dtype when every non-null value is castable to it.
    """
    if t == pl.Boolean:
        return pl.col(c).str.contains(_CSV_TRUE_RE)
    return pl.col(c).cast(t, strict=False)


def convert_csv_to_parquet(
    src: Path | UPath,
    dest: Path,
    columns: list[str] | None = None,
    *,
    tmp_dir: Path | None = None,
) -> dict[str, pl.DataType]:
    """Convert a csv-family file to parquet in three bounded passes; return the schema.

    The obvious one-liner — ``scan_csv(infer_schema_length=None).sink_parquet()`` —
    is not an option: full-file inference materializes the file to decide types, so
    it peaks at more memory than reading the CSV outright (measured 3.1 GB on a 1 GB
    input, against 2.6 GB for a plain eager read). Splitting inference from
    conversion is what makes both halves cheap:

    1. **CSV → all-String parquet.** ``infer_schema_length=0`` types nothing, so
       polars streams straight through. Empty fields become nulls here, exactly as
       they would under inference — the distinction CSV itself cannot express, and
       the one downstream ``??`` coalescing and null-drops depend on.
    2. **Infer** each column's dtype off that parquet (:func:`infer_column_dtypes`) —
       a per-column aggregate over a columnar file, not a table scan.
    3. **Cast and write** the final typed parquet, projecting to ``columns``.

    The intermediate is written under ``tmp_dir`` (default: beside ``dest``) and
    removed afterwards, so the cost is transient disk rather than memory.

    The written parquet matches what ``pl.read_csv(src, infer_schema_length=None)``
    would produce — same schema, same values — for every CSV polars itself can read;
    ``tests/test_convert_to_parquet.py`` enforces that round-trip with a property
    test. The sole deliberate deviation: where polars' full read hard-errors on a
    column its inference classifies as typed but whose values it then cannot parse
    (an integer-shaped column overflowing Int64, in practice), this conversion
    degrades gracefully instead — lossy Float64 for the overflow case.

    Args:
        src: The csv / csv.gz file to convert.
        dest: Where to write the typed parquet. Parent directories are created.
        columns: Project to these columns. ``None`` keeps every column. Columns
            absent from the source raise, naming what the file actually has.
        tmp_dir: Where the intermediate String parquet goes.

    Returns:
        The inferred schema of the *written* columns.

    Raises:
        ValueError: If ``columns`` names a column the source lacks.

    Examples:
        >>> with yaml_disk('''
        ... labs.csv: |
        ...   subject_id,value,unit,note
        ...   1,1.5,mg,ok
        ...   2,3,mg,
        ... ''') as d:
        ...     out = Path(d) / "labs.parquet"
        ...     schema = convert_csv_to_parquet(Path(d) / "labs.csv", out)
        ...     print(schema)
        ...     pl.read_parquet(out)
        {'subject_id': Int64, 'value': Float64, 'unit': String, 'note': String}
        shape: (2, 4)
        ┌────────────┬───────┬──────┬──────┐
        │ subject_id ┆ value ┆ unit ┆ note │
        │ ---        ┆ ---   ┆ ---  ┆ ---  │
        │ i64        ┆ f64   ┆ str  ┆ str  │
        ╞════════════╪═══════╪══════╪══════╡
        │ 1          ┆ 1.5   ┆ mg   ┆ ok   │
        │ 2          ┆ 3.0   ┆ mg   ┆ null │
        └────────────┴───────┴──────┴──────┘

        The empty ``note`` on row 2 arrives as a real null, not ``""`` — carrying
        CSV's missing-value semantics across the conversion.

        Projection keeps only what the MESSY config needs, so a wide source table
        never costs downstream stages anything for columns nobody reads:

        >>> with yaml_disk('''
        ... labs.csv: |
        ...   subject_id,value,unit,note
        ...   1,1.5,mg,ok
        ... ''') as d:
        ...     out = Path(d) / "labs.parquet"
        ...     _ = convert_csv_to_parquet(Path(d) / "labs.csv", out, ["subject_id", "value"])
        ...     pl.read_parquet(out).columns
        ['subject_id', 'value']

        A requested column the file lacks fails naming both sides:

        >>> with yaml_disk('''
        ... labs.csv: |
        ...   subject_id,value
        ...   1,2
        ... ''') as d:
        ...     convert_csv_to_parquet(Path(d) / "labs.csv", Path(d) / "o.parquet", ["nope"])
        Traceback (most recent call last):
            ...
        ValueError: ...labs.csv is missing requested column(s) ['nope']. It has:
        ['subject_id', 'value'].
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tmp_dir) if tmp_dir is not None else dest.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # Name the intermediate after the destination so concurrent conversions of
    # different files never collide on it.
    tmp_fp = tmp_dir / f".{dest.name}.strings.tmp.parquet"

    try:
        # Pass 1: no inference, so nothing is materialized to decide types.
        pl.scan_csv(src, infer_schema_length=0).sink_parquet(tmp_fp)

        as_strings = pl.scan_parquet(tmp_fp, glob=False)
        available = as_strings.collect_schema().names()
        if columns is not None:
            missing = [c for c in columns if c not in available]
            if missing:
                raise ValueError(
                    f"{src} is missing requested column(s) {sorted(missing)}. It has: {sorted(available)}."
                )
            as_strings = as_strings.select(columns)

        # Pass 2: infer over the columnar intermediate.
        schema = infer_column_dtypes(as_strings)

        # Pass 3: decode and write. String columns need no expression at all.
        casts = [_decode_expr(c, t) for c, t in schema.items() if t != pl.String]
        (as_strings.with_columns(casts) if casts else as_strings).sink_parquet(dest)
        return schema
    finally:
        tmp_fp.unlink(missing_ok=True)
