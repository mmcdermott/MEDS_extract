"""File-IO helpers shared across stages.

This module is the *one* place in the pipeline where file-format dispatch
(``parquet`` / ``par`` / ``csv`` / ``csv.gz``) and layout resolution
(bare file vs. sub-sharded directory) live. Stages call :func:`scan_source`
(read one or more files, concatenated) and :func:`resolve_source_files`
(find the files for a given prefix) rather than rolling their own
``scan_parquet`` / ``glob`` calls.
"""

from __future__ import annotations

import codecs
import gzip
import logging
import shutil
import warnings
from collections.abc import Mapping
from contextlib import ExitStack
from io import TextIOWrapper
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


def is_utf8(encoding: str) -> bool:
    """Whether ``encoding`` names UTF-8, under any of Python's aliases for it.

    Raises ``LookupError`` for a name Python does not know.

    Examples:
        >>> is_utf8("utf8"), is_utf8("UTF-8"), is_utf8("latin-1"), is_utf8("cp1252")
        (True, True, False, False)
        >>> is_utf8("not-a-codec")
        Traceback (most recent call last):
            ...
        LookupError: unknown encoding: not-a-codec
    """
    return codecs.lookup(encoding).name == "utf-8"


# The ``etl.encoding`` value (and default) under which each csv source's codec is
# detected rather than declared.
AUTO_ENCODING = "auto"


def source_encoding(encoding: str | Mapping[str, str], prefix: str) -> str:
    """The codec to decode source ``prefix`` with, from the ``etl.encoding`` option.

    The option is either one codec for every source or a mapping of source prefix to
    codec; a prefix the mapping does not list is :data:`AUTO_ENCODING`, detected per file
    by :func:`resolve_encoding`.

    Examples:
        >>> source_encoding("latin-1", "labs")
        'latin-1'
        >>> source_encoding({"labs": "latin-1"}, "labs"), source_encoding({"labs": "latin-1"}, "vitals")
        ('latin-1', 'auto')
    """
    if isinstance(encoding, Mapping):
        return encoding.get(prefix, AUTO_ENCODING)
    return encoding


# Byte-order marks that name their codec outright. UTF-32 precedes UTF-16 because a
# UTF-32-LE mark begins with the UTF-16-LE one. A UTF-8 mark needs no entry: it is
# valid UTF-8, and polars strips it.
_BOMS = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)

# The five bytes cp1252 leaves undefined; a non-UTF-8 source containing any of them
# cannot be cp1252, so it falls back to latin-1, which defines every byte.
_CP1252_UNDEFINED = (b"\x81", b"\x8d", b"\x8f", b"\x90", b"\x9d")


def _open_bytes(src: Path | UPath, stack: ExitStack):
    """Open ``src`` for binary reading on ``stack``, transparently decompressing ``.gz``."""
    raw = stack.enter_context(src.open("rb"))
    if "".join(src.suffixes).lower().endswith(".gz"):
        raw = stack.enter_context(gzip.GzipFile(fileobj=raw))
    return raw


def detect_encoding(src: Path | UPath, *, chunk_bytes: int = 1 << 22) -> str:
    """Detect the text encoding of a csv / csv.gz file, deterministically.

    In order: a UTF-16/32 byte-order mark names its codec; a file that is valid UTF-8
    is UTF-8; any other file is cp1252, or latin-1 if it holds a byte cp1252 leaves
    undefined. Real non-ASCII text in another encoding almost never forms valid UTF-8,
    and cp1252 is what Western European exports use. General-purpose statistical
    detectors are deliberately avoided: on short code and unit strings they mistake
    cp1252 for cp1250, which garbles ``°`` and ``µ`` without raising. Other codecs
    (cp1250, Shift-JIS, ...) and BOM-less UTF-16 are not detected and must be
    declared. Memory use is bounded by ``chunk_bytes``.

    Examples:
        >>> import tempfile
        >>> text = "item,unit\\nTemperatuur,\\u00b0C\\nCaf\\u00e9\\u00efne,\\u00b5g\\n"
        >>> with tempfile.TemporaryDirectory() as d:
        ...     fp = Path(d) / "items.csv"
        ...     for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-32", "cp1252", "latin-1"):
        ...         _ = fp.write_bytes(text.encode(enc))
        ...         print(enc, "->", detect_encoding(fp))
        utf-8 -> utf8
        utf-8-sig -> utf8
        utf-16 -> utf-16
        utf-32 -> utf-32
        cp1252 -> cp1252
        latin-1 -> cp1252

        Latin-1 text that uses a byte cp1252 leaves undefined (here a C1 control
        character) is still decoded strictly, as latin-1:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     fp = Path(d) / "items.csv"
        ...     _ = fp.write_bytes("item\\nCaf\\u00e9\\x81\\n".encode("latin-1"))
        ...     detect_encoding(fp)
        'latin-1'

        A UTF-8 sequence split across a chunk boundary is still valid UTF-8:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     fp = Path(d) / "items.csv"
        ...     _ = fp.write_bytes("ab\\u00b5g".encode("utf-8"))
        ...     detect_encoding(fp, chunk_bytes=3)
        'utf8'
    """
    with ExitStack() as stack:
        raw = _open_bytes(src, stack)
        chunk = raw.read(chunk_bytes)
        for bom, codec in _BOMS:
            if chunk.startswith(bom):
                return codec
        decoder = codecs.getincrementaldecoder("utf-8")()
        valid_utf8, cp1252_undefined = True, False
        while True:
            final = not chunk
            if valid_utf8:
                try:
                    decoder.decode(chunk, final=final)
                except UnicodeDecodeError:
                    valid_utf8 = False
            cp1252_undefined = cp1252_undefined or any(b in chunk for b in _CP1252_UNDEFINED)
            if final or (not valid_utf8 and cp1252_undefined):
                break
            chunk = raw.read(chunk_bytes)
    if valid_utf8:
        return "utf8"
    return "latin-1" if cp1252_undefined else "cp1252"


def resolve_encoding(src: Path | UPath, encoding: str) -> str:
    """The codec to decode ``src`` with: ``encoding`` itself, unless it is ``"auto"``.

    Under :data:`AUTO_ENCODING` the codec is detected (:func:`detect_encoding`), and a
    detected non-UTF-8 codec is logged as a warning naming the file, so a wrong guess
    is visible and can be pinned in the MESSY config.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     fp = Path(d) / "items.csv"
        ...     _ = fp.write_bytes("item\\nCaf\\u00e9\\n".encode("cp1252"))
        ...     resolve_encoding(fp, "auto"), resolve_encoding(fp, "latin-1")
        ('cp1252', 'latin-1')
    """
    if encoding != AUTO_ENCODING:
        return encoding
    codec = detect_encoding(src)
    if not is_utf8(codec):
        logger.warning(
            f"{src} is not valid UTF-8; decoding it as detected encoding {codec}. If that is "
            f"wrong, declare the source's codec in the MESSY config, e.g. "
            f"`etl: {{encoding: {{<prefix>: <codec>}}}}`."
        )
    return codec


def transcode_to_utf8(src: Path | UPath, dest: Path, encoding: str, *, chunk_chars: int = 1 << 22) -> None:
    """Stream a csv / csv.gz file decoded as ``encoding`` into an uncompressed UTF-8 copy.

    Polars' CSV reader only understands UTF-8 (strictly, or with lossy replacement of
    invalid bytes), so a source in any other encoding is re-encoded on disk before it is
    scanned. Decoding is strict: a byte sequence that is invalid in ``encoding`` raises
    rather than being replaced, so nothing is silently corrupted. Memory use is bounded
    by ``chunk_chars``, and ``newline=""`` keeps line endings (including any inside quoted
    fields) byte-for-byte.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     src, dest = Path(d) / "items.csv", Path(d) / "items.utf8.csv"
        ...     text = "item,unit\\nTemperatuur,\\u00b0C\\nCaf\\u00e9\\u00efne,mg\\n"
        ...     _ = src.write_bytes(text.encode("latin-1"))
        ...     transcode_to_utf8(src, dest, "latin-1")
        ...     dest.read_bytes().decode("utf-8").splitlines()
        ['item,unit', 'Temperatuur,°C', 'Caféïne,mg']

        Bytes that are invalid in the declared encoding fail loudly:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     src = Path(d) / "items.csv"
        ...     _ = src.write_bytes(b"item\\n\\xff\\n")
        ...     transcode_to_utf8(src, Path(d) / "out.csv", "ascii")
        Traceback (most recent call last):
            ...
        UnicodeDecodeError: 'ascii' codec can't decode byte 0xff in position 5: ordinal not in range(128)
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        raw = _open_bytes(src, stack)
        fin = stack.enter_context(TextIOWrapper(raw, encoding=encoding, newline=""))
        fout = stack.enter_context(dest.open("w", encoding="utf-8", newline=""))
        shutil.copyfileobj(fin, fout, chunk_chars)


def convert_csv_to_parquet(
    src: Path | UPath,
    dest: Path,
    columns: list[str] | None = None,
    *,
    tmp_dir: Path | None = None,
    encoding: str = AUTO_ENCODING,
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

    A source in a non-UTF-8 ``encoding`` is first streamed into a transient UTF-8 copy
    (:func:`transcode_to_utf8`), also under ``tmp_dir``, because polars' reader cannot
    decode anything else. Under the default ``"auto"`` the source is read as UTF-8, and
    only when polars rejects it is its codec detected (:func:`resolve_encoding`) and
    the copy made, so a UTF-8 source costs nothing extra.

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
        tmp_dir: Where the intermediate String parquet (and UTF-8 copy) goes.
        encoding: The source's text encoding, as a Python codec name (e.g.
            ``"latin-1"``, ``"cp1252"``), or ``"auto"`` to detect it. Decoding is strict.

    Returns:
        The inferred schema of the *written* columns.

    Raises:
        ValueError: If ``columns`` names a column the source lacks, or if a source
            declared UTF-8 is not valid UTF-8.
        UnicodeDecodeError: If a source is not valid in a non-UTF-8 ``encoding``.

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

        A source in another encoding is detected by default, or decoded with the codec
        it declares. Declared UTF-8, it fails naming the file and the fix, instead of a
        bare polars error:

        >>> import tempfile
        >>> d = tempfile.TemporaryDirectory()
        >>> src = Path(d.name) / "items.csv"
        >>> _ = src.write_bytes("item,value\\nCaf\\u00e9\\u00efne,1\\n".encode("latin-1"))
        >>> _ = convert_csv_to_parquet(src, Path(d.name) / "items.parquet")
        >>> pl.read_parquet(Path(d.name) / "items.parquet")["item"].to_list()
        ['Caféïne']
        >>> _ = convert_csv_to_parquet(src, Path(d.name) / "items.parquet", encoding="latin-1")
        >>> pl.read_parquet(Path(d.name) / "items.parquet")["item"].to_list()
        ['Caféïne']
        >>> convert_csv_to_parquet(src, Path(d.name) / "items.parquet", encoding="utf8")
        Traceback (most recent call last):
            ...
        ValueError: ...items.csv is not valid UTF-8. If the source uses another encoding,
        declare it in the MESSY config, e.g. `etl: {encoding: latin-1}`. ...
        >>> d.cleanup()
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tmp_dir) if tmp_dir is not None else dest.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # Name the intermediate after the destination so concurrent conversions of
    # different files never collide on it.
    tmp_fp = tmp_dir / f".{dest.name}.strings.tmp.parquet"
    utf8_fp = tmp_dir / f".{dest.name}.utf8.tmp.csv"

    try:
        csv_fp = src
        if encoding != AUTO_ENCODING and not is_utf8(encoding):
            transcode_to_utf8(src, utf8_fp, encoding)
            csv_fp = utf8_fp

        # Pass 1: no inference, so nothing is materialized to decide types.
        try:
            pl.scan_csv(csv_fp, infer_schema_length=0).sink_parquet(tmp_fp)
        except pl.exceptions.ComputeError as e:
            if "invalid utf-8" not in str(e):
                raise
            if encoding != AUTO_ENCODING:
                raise ValueError(
                    f"{src} is not valid UTF-8. If the source uses another encoding, declare it "
                    f"in the MESSY config, e.g. `etl: {{encoding: latin-1}}`. Polars reported: {e}"
                ) from e
            # Detection is deferred to here so a UTF-8 source is read only once.
            transcode_to_utf8(src, utf8_fp, resolve_encoding(src, encoding))
            pl.scan_csv(utf8_fp, infer_schema_length=0).sink_parquet(tmp_fp)

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
        utf8_fp.unlink(missing_ok=True)
