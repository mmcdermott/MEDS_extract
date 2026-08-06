"""Round-trip parity tests for ``MEDS_extract.io.convert_csv_to_parquet``.

The conversion promises to type and decode a CSV exactly as polars' own full-file
inference would: for any CSV file that ``pl.read_csv(fp, infer_schema_length=None)``
can read, converting it and reading the parquet back must yield the identical frame —
same schema, same values. The property test here enforces that promise end-to-end
over generated CSVs. The oracle is differential — polars reads the very same file —
so no generated token needs a known answer, and cell generation can be genuinely
open-ended. Cells come from three tiers of leaf strategy:

1. PRIMARY (coverage): draw a real value (unbounded int, arbitrary float, arbitrary
   date/datetime), then render it through channels that know nothing about polars —
   Python format specs, ``repr``, ``Decimal``/numpy renderings, ``strftime`` over
   real-world formats. This searches how values are expressed in the wild.
2. SECONDARY (structured exploration): compositional string grammars over the shape
   of numeric/boolean notation in general — sign x digit-runs x separators x
   exponents, per-character casing flips — plus one-character boundary mutants of
   tier-1/tier-2 tokens.
3. CANARY (drift detection only): tokens sampled from the acceptor regexes extracted
   from polars' source. These are deliberately NOT coverage — they exist so that if a
   future polars changes its CSV grammar, old-acceptor-shaped tokens start diverging
   in CI.

Columns mix classes as well as drawing them pure, so cross-class unification is
exercised as hard as the leaves.

The sole deliberate deviation, excluded from the property and pinned deterministically
instead: an integer-shaped column overflowing Int64 makes polars' full-file read
hard-error, while the conversion degrades it to (lossy) Float64.
"""

from __future__ import annotations

import tempfile
from decimal import Decimal
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

from MEDS_extract.io import _CSV_BOOL_RE, _CSV_FLOAT_RE, _CSV_INT_RE, convert_csv_to_parquet

# ── Tier 1: value renderings through polars-agnostic channels ────────
# Draw a VALUE from the class's full domain, then print it the ways real-world CSV
# writers do. None of these carry any expectation of being accepted as typed; the
# differential oracle decides what polars makes of each file.

_INT_FORMAT_SPECS = ["d", "+d", "08d", "016d", ",", "_", "x"]


@st.composite
def _rendered_int_tokens(draw: st.DrawFn) -> str:
    value = draw(st.integers())
    spec = draw(st.sampled_from(_INT_FORMAT_SPECS))
    return format(value, spec)


_FLOAT_FORMAT_SPECS = ["e", "E", "f", "g", "G", ".0f", ".3f", ".17g", ".2%"]


@st.composite
def _rendered_float_tokens(draw: st.DrawFn) -> str:
    value = draw(st.floats(allow_nan=True, allow_infinity=True))
    channel = draw(st.sampled_from(["str", "repr", "format", "decimal", "np32", "np64"]))
    match channel:
        case "str":
            return str(value)
        case "repr":
            return repr(value)
        case "decimal":
            # Exact binary expansions: Decimal(0.1) prints 55 digits; tiny magnitudes
            # print scientific 'E-300' forms; nan/inf print 'NaN'/'Infinity'.
            return str(Decimal(value))
        case "np32":
            with np.errstate(over="ignore"):  # f64 → f32 overflow to inf is the point
                return str(np.float32(value))
        case "np64":
            return str(np.float64(value))
        case _:
            return format(value, draw(st.sampled_from(_FLOAT_FORMAT_SPECS)))


# Real-world temporal renderings. Polars' CSV inference types no temporals (default
# ``try_parse_dates=False``), so every one of these must stay String and round-trip
# verbatim — generating them broadly guards exactly that.
_STRFTIME_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%m/%d/%Y",
    "%d/%m/%Y %H:%M",
    "%Y-%m-%d %I:%M %p",
    "%b %d, %Y",
    "%Y%m%d",
]


@st.composite
def _rendered_temporal_tokens(draw: st.DrawFn) -> str:
    value = draw(st.dates() | st.datetimes())
    return value.strftime(draw(st.sampled_from(_STRFTIME_FORMATS)))


# ── Tier 2: compositional grammars over the shape of numeric notation ─
# Built from what numeric/boolean text looks like in general — signs, digit runs,
# separators, exponent parts, word casings — NOT from polars' acceptors. Overlap with
# the acceptor space is incidental, never the design target.

_signs = st.sampled_from(["", "-", "+"])
_digit_runs = st.text(alphabet="0123456789", min_size=1, max_size=45)


def _cased(word: str) -> st.SearchStrategy[str]:
    """Per-character upper/lower flips of ``word`` ('nan' → 'NaN', 'nAN', ...)."""
    flips = st.lists(st.booleans(), min_size=len(word), max_size=len(word))
    return flips.map(lambda fs: "".join(c.upper() if f else c for c, f in zip(word, fs, strict=True)))


@st.composite
def _grammar_int_tokens(draw: st.DrawFn) -> str:
    """Sign x leading zeros x magnitude, with magnitudes well past i64::MAX."""
    return draw(_signs) + "0" * draw(st.integers(0, 4)) + str(draw(st.integers(0, 2**70)))


@st.composite
def _grammar_float_tokens(draw: st.DrawFn) -> str:
    """Sign x (integer/fraction parts) x optional exponent, or a cased special word.

    Reaches forms no float ``repr`` emits: '5.', '.5', '00.5e+007', 45-digit
    mantissas, exponents far beyond f64 range, 'iNfInItY', ...
    """
    sign = draw(_signs)
    shape = draw(st.sampled_from(["int.frac", ".frac", "int.", "int_exp", "special"]))
    if shape == "special":
        return sign + draw(_cased(draw(st.sampled_from(["inf", "infinity", "nan"]))))
    exponent = ""
    if shape == "int_exp" or draw(st.booleans()):
        exponent = draw(st.sampled_from("eE")) + draw(_signs) + draw(_digit_runs)
    match shape:
        case "int.frac":
            body = draw(_digit_runs) + "." + draw(_digit_runs)
        case ".frac":
            body = "." + draw(_digit_runs)
        case "int.":
            body = draw(_digit_runs) + "."
        case _:
            body = draw(_digit_runs)
    return sign + body + exponent


# Near-words a CSV might plausibly carry, including deliberate truncations/doublings.
_BOOL_NEAR_WORDS = [
    "t",
    "f",
    "T",
    "F",
    "yes",
    "no",
    "y",
    "n",
    "1",
    "0",
    "tru",  # codespell:ignore tru
    "truee",
]


@st.composite
def _grammar_bool_tokens(draw: st.DrawFn) -> str:
    """Any casing of true/false, plus the near-words a CSV might plausibly carry."""
    if draw(st.integers(0, 3)):  # 3:1 real boolean words to near-words
        return draw(_cased(draw(st.sampled_from(["true", "false"]))))
    return draw(st.sampled_from(_BOOL_NEAR_WORDS))


@st.composite
def _grammar_temporal_tokens(draw: st.DrawFn) -> str:
    """ISO-ish datetime assembly: separator, fractional width 0-9, tz suffix."""
    if draw(st.booleans()):
        return draw(st.dates()).isoformat()
    token = draw(st.datetimes()).replace(microsecond=0).isoformat(sep=draw(st.sampled_from(" T")))
    frac_width = draw(st.integers(0, 9))
    if frac_width:
        token += "." + draw(st.text(alphabet="0123456789", min_size=frac_width, max_size=frac_width))
    return token + draw(st.sampled_from(["", "Z", "+00:00", "-05:00", "+0530"]))


_int_tokens = _rendered_int_tokens() | _grammar_int_tokens()
_float_tokens = _rendered_float_tokens() | _grammar_float_tokens()
_bool_tokens = _grammar_bool_tokens()
_temporal_tokens = _rendered_temporal_tokens() | _grammar_temporal_tokens()
_well_formed_tokens = st.one_of(_int_tokens, _float_tokens, _bool_tokens, _temporal_tokens)

# One-character edits of well-formed tokens — probing just outside (and further
# inside) every token grammar, from both sides. The alphabet leans on characters that
# sit near numeric notation: digits, signs, exponent letters, separators, whitespace,
# unicode minus U+2212, and unicode digits (arabic-indic).
# The escapes are U+2212 MINUS SIGN, U+0660 ARABIC-INDIC ZERO, U+06F1 EXTENDED
# ARABIC-INDIC ONE — confusability with ASCII '-', '0', '1' is exactly their job.
_EDIT_CHARS = '0123456789+-eE._ ,"x' + "\u2212\u0660\u06f1"


@st.composite
def _mutant_tokens(draw: st.DrawFn) -> str:
    base = draw(_well_formed_tokens)
    op = draw(st.sampled_from(["insert", "delete", "replace"]))
    if op == "insert" or not base:
        i = draw(st.integers(0, len(base)))
        return base[:i] + draw(st.sampled_from(_EDIT_CHARS)) + base[i:]
    i = draw(st.integers(0, len(base) - 1))
    if op == "delete":
        return base[:i] + base[i + 1 :]
    return base[:i] + draw(st.sampled_from(_EDIT_CHARS)) + base[i + 1 :]


# ── Tier 3: acceptor-regex canary (drift detection ONLY) ─────────────
# Tokens sampled from the acceptor regexes extracted verbatim from polars' source.
# This leaf is intentionally NOT coverage — sampling the same grammar io.py
# implements is circular. It exists so that if a future polars changes its CSV
# acceptors, tokens shaped like the OLD grammar start diverging in CI immediately.
_canary_regex_tokens = st.one_of(
    st.from_regex(_CSV_INT_RE, fullmatch=True),
    st.from_regex(_CSV_FLOAT_RE, fullmatch=True),
    st.from_regex(_CSV_BOOL_RE, fullmatch=True),
)

# NUL bytes are excluded: CSV cannot carry them faithfully, so they only test the
# writer, not the conversion. Everything else — newlines, quotes, commas, unicode —
# stays in play and rides through ``write_csv``'s quoting.
_arbitrary_text = st.text(st.characters(codec="utf-8", exclude_characters="\x00"), max_size=8)

# A column draws one flavor, so columns are often homogeneous enough to exercise the
# Int64/Float64/Boolean rungs; the final everything-mix flavor exercises cross-class
# unification.
_COLUMN_FLAVORS: list[st.SearchStrategy[str]] = [
    _int_tokens,
    _float_tokens,
    _bool_tokens,
    _temporal_tokens,
    _arbitrary_text,
    _mutant_tokens(),
    _canary_regex_tokens,
]
_COLUMN_FLAVORS.append(st.one_of(*_COLUMN_FLAVORS))


@st.composite
def string_frames(draw: st.DrawFn) -> pl.DataFrame:
    """An all-String DataFrame whose ``write_csv`` output is the file under test."""
    n_rows = draw(st.integers(min_value=1, max_value=30))
    n_cols = draw(st.integers(min_value=1, max_value=4))
    cols: dict[str, list[str | None]] = {}
    for i in range(n_cols):
        elem = st.none() | draw(st.sampled_from(_COLUMN_FLAVORS))
        cols[f"c{i}"] = draw(st.lists(elem, min_size=n_rows, max_size=n_rows))
    return pl.DataFrame(cols, schema=dict.fromkeys(cols, pl.String))


# ── Pinned boundary pools ────────────────────────────────────────────
# The original fixed token pools, demoted from generator vocabulary to deterministic
# boundary probes: every token still runs on every invocation via the ``@example``
# pins below, but hypothesis is free to search well beyond them.

INT_TOKENS = ["0", "1", "-2", "30", "007", "-007", "+1", "9223372036854775807"]
FLOAT_TOKENS = [
    "1",
    "2",
    "1.0",
    "2.5",
    "-0.0",
    ".5",
    "5.",
    "00.5",
    "1e5",
    "1E5",
    "1e+5",
    "1e-5",
    "1.5E-3",
    "5.e3",
    "1e999",
    "+1.5",
    "-.5",
    "inf",
    "-inf",
    "+inf",
    "Inf",
    "INF",
    "infinity",
    "nan",
    "NaN",
    "NAN",
    "-NaN",
]
BOOL_TOKENS = ["true", "false", "True", "False", "TRUE", "FALSE", "tRuE", "FaLsE", "t", "F", "yes", "no"]
DATE_TOKENS = ["2024-01-01", "2023-12-31", "2024-01-01 12:00:00", "2024-01-01T12:00:00"]
TEXT_TOKENS = [
    "",
    " ",
    "a,b",
    'he said "hi"',
    "naïve",
    "日本語",
    " padded ",
    " true",
    "true ",
    " 1",
    "1 ",
    " 1.5",
    "line\nbreak",
    "tab\there",
    "NULL",
    "null",
    "NA",
    "None",
    "1_000",
    "0x1F",
    "1.2.3",
    "e5",
    ".e5",
    ".",
    "+",
    "-",
]
# Tokens that make polars' full-file inference itself hard-error when a column is
# otherwise integer-shaped; pinned mixed with a float so the file stays readable, and
# generated organically by the int leaves so the ``assume`` exclusion path is walked.
OVERFLOW_TOKENS = ["9223372036854775808", "99999999999999999999", "-9223372036854775809"]
ALL_TOKENS = INT_TOKENS + FLOAT_TOKENS + BOOL_TOKENS + DATE_TOKENS + TEXT_TOKENS + OVERFLOW_TOKENS


def _sdf(**cols: list[str | None]) -> pl.DataFrame:
    """Shorthand for the explicit ``@example`` frames below."""
    return pl.DataFrame(cols, schema=dict.fromkeys(cols, pl.String))


def _round_trip(csv_fp: Path) -> tuple[dict[str, pl.DataType], pl.DataFrame]:
    """Convert ``csv_fp`` and read the parquet back, returning (schema, frame)."""
    dest = csv_fp.parent / f"{csv_fp.name}.out.parquet"
    schema = convert_csv_to_parquet(csv_fp, dest)
    return schema, pl.read_parquet(dest)


# ── The property ─────────────────────────────────────────────────────


# Known-divergence regressions (#199/#200), pinned so they run on every invocation:
@example(df=_sdf(flag=["true", "false", None]), quote_style="necessary")  # 199: the crash
@example(df=_sdf(flag=["True", "False"]), quote_style="necessary")  # pandas-style casing
@example(df=_sdf(flag=["TRUE", "FALSE"]), quote_style="necessary")
@example(df=_sdf(flag=["tRuE", "FaLsE"]), quote_style="always")
@example(df=_sdf(n=["+1", "2"]), quote_style="necessary")  # '+1' is String to polars
@example(df=_sdf(x=["1", "2.5", None]), quote_style="necessary")  # int/float unify
@example(df=_sdf(x=["inf", "-inf", "NaN", "-NaN"]), quote_style="necessary")
@example(df=_sdf(x=["Inf", "nan", "NAN", "infinity"]), quote_style="necessary")  # String casings
@example(df=_sdf(x=[None, None]), quote_style="necessary")  # all-null column
@example(df=_sdf(x=["", ""]), quote_style="always")  # quoted "" is a value, not null
@example(df=_sdf(x=[" true", "false "], y=[" 1", "1 "]), quote_style="necessary")  # no trimming
@example(df=_sdf(x=["007", "-007"]), quote_style="necessary")  # leading zeros
@example(df=_sdf(x=["9223372036854775807"]), quote_style="necessary")  # i64::MAX, single row
@example(df=_sdf(x=["9223372036854775808", "1.5"]), quote_style="necessary")  # readable overflow
@example(df=_sdf(x=["5.", "5.e3", ".5e3"]), quote_style="necessary")  # trailing-dot forms
@example(
    df=_sdf(d=["2024-01-01", "2023-12-31"], t=["2024-01-01 12:00:00", "2024-01-01T12:00:00"]),
    quote_style="necessary",
)
# The demoted boundary pools, one column each (plus one everything-column):
@example(df=_sdf(ints=[*INT_TOKENS, None]), quote_style="necessary")
@example(df=_sdf(floats=[*FLOAT_TOKENS, None]), quote_style="necessary")
@example(df=_sdf(bools=[*BOOL_TOKENS, None]), quote_style="necessary")
@example(df=_sdf(dates=[*DATE_TOKENS, None]), quote_style="necessary")
@example(df=_sdf(text=[*TEXT_TOKENS, None]), quote_style="non_numeric")
@example(df=_sdf(overflow=[*OVERFLOW_TOKENS, "1.5"]), quote_style="necessary")
@example(df=_sdf(everything=ALL_TOKENS), quote_style="necessary")
@settings(max_examples=75, deadline=None)
@given(df=string_frames(), quote_style=st.sampled_from(["necessary", "always", "non_numeric"]))
def test_round_trip_matches_full_file_inference(df: pl.DataFrame, quote_style: str) -> None:
    """``convert_csv_to_parquet`` output must equal a full-inference read of the CSV."""
    with tempfile.TemporaryDirectory() as tmp:
        fp = Path(tmp) / "t.csv"
        df.write_csv(fp, quote_style=quote_style)

        try:
            expected = pl.read_csv(fp, infer_schema_length=None)
        except pl.exceptions.PolarsError:
            # Full-file inference itself refuses this file — in practice an
            # integer-shaped column overflowing Int64. The parity promise covers only
            # files polars can read; the conversion's documented lenient fallback for
            # this case is pinned in test_int64_overflow_degrades_to_float64.
            assume(False)

        schema, actual = _round_trip(fp)
        assert dict(actual.schema) == dict(expected.schema)
        assert dict(schema) == dict(expected.schema)
        assert_frame_equal(actual, expected, check_exact=True)


# ── Deterministic regressions ────────────────────────────────────────


def _write(tmp_path: Path, text: str) -> Path:
    fp = tmp_path / "t.csv"
    fp.write_text(text)
    return fp


def test_lowercase_booleans_end_to_end(tmp_path: Path) -> None:
    """The #199 crash case: an all-true/false column must convert, not raise."""
    fp = _write(tmp_path, "flag,pad\ntrue,1\nfalse,2\n,3\n")
    schema, out = _round_trip(fp)
    assert schema["flag"] == pl.Boolean
    assert out["flag"].to_list() == [True, False, None]
    assert_frame_equal(out, pl.read_csv(fp, infer_schema_length=None), check_exact=True)


@pytest.mark.parametrize(
    ("truthy", "falsy"),
    [("True", "False"), ("TRUE", "FALSE"), ("tRuE", "FaLsE")],
    ids=["pandas-style", "upper", "mixed-case"],
)
def test_boolean_casings_infer_boolean(tmp_path: Path, truthy: str, falsy: str) -> None:
    """Polars accepts any casing of true/false — so must the conversion (#200a)."""
    fp = _write(tmp_path, f"flag,pad\n{truthy},1\n{falsy},2\n,3\n")
    schema, out = _round_trip(fp)
    assert schema["flag"] == pl.Boolean
    assert out["flag"].to_list() == [True, False, None]
    assert_frame_equal(out, pl.read_csv(fp, infer_schema_length=None), check_exact=True)


@pytest.mark.parametrize(
    ("truthy", "falsy"),
    [("t", "f"), ("T", "F"), ("yes", "no"), (" true", "false"), ("true ", "false"), ("1", "true")],
    ids=["t-f", "T-F", "yes-no", "leading-space", "trailing-space", "bool-int-mix"],
)
def test_non_boolean_tokens_stay_string(tmp_path: Path, truthy: str, falsy: str) -> None:
    """Exactly the tokens polars rejects for Boolean must stay String, verbatim."""
    fp = _write(tmp_path, f"flag,pad\n{truthy},1\n{falsy},2\n")
    schema, out = _round_trip(fp)
    assert schema["flag"] == pl.String
    assert out["flag"].to_list() == [truthy, falsy]
    assert_frame_equal(out, pl.read_csv(fp, infer_schema_length=None), check_exact=True)


def test_plus_prefixed_integers_stay_string(tmp_path: Path) -> None:
    """Polars infers String for '+1' (#200b); the text must survive unrewritten."""
    fp = _write(tmp_path, "n\n+1\n2\n")
    schema, out = _round_trip(fp)
    assert schema["n"] == pl.String
    assert out["n"].to_list() == ["+1", "2"]
    assert_frame_equal(out, pl.read_csv(fp, infer_schema_length=None), check_exact=True)


def test_mixed_int_float_becomes_float(tmp_path: Path) -> None:
    fp = _write(tmp_path, "x\n1\n2.5\n")
    schema, out = _round_trip(fp)
    assert schema["x"] == pl.Float64
    assert out["x"].to_list() == [1.0, 2.5]
    assert_frame_equal(out, pl.read_csv(fp, infer_schema_length=None), check_exact=True)


def test_quoted_empty_string_is_a_value_not_null(tmp_path: Path) -> None:
    fp = _write(tmp_path, 'c,pad\n"",1\n,2\n')
    schema, out = _round_trip(fp)
    assert schema["c"] == pl.String
    assert out["c"].to_list() == ["", None]
    assert_frame_equal(out, pl.read_csv(fp, infer_schema_length=None), check_exact=True)


def test_all_empty_column_stays_null_string(tmp_path: Path) -> None:
    fp = _write(tmp_path, "c,pad\n,1\n,2\n")
    schema, out = _round_trip(fp)
    assert schema["c"] == pl.String
    assert out["c"].to_list() == [None, None]
    assert_frame_equal(out, pl.read_csv(fp, infer_schema_length=None), check_exact=True)


def test_int64_overflow_degrades_to_float64(tmp_path: Path) -> None:
    """The documented deviation (#200c): polars hard-errors, the conversion is lenient.

    A column of integer-shaped values with one past i64::MAX cannot be read by polars'
    own full-file inference at all. Rather than replicate the crash, the conversion
    deliberately degrades the column to (lossy) Float64 — the file converts, at the
    cost of float precision on the overflowing values.
    """
    fp = _write(tmp_path, "x\n1\n9223372036854775808\n")
    with pytest.raises(pl.exceptions.PolarsError):
        pl.read_csv(fp, infer_schema_length=None)
    schema, out = _round_trip(fp)
    assert schema["x"] == pl.Float64
    assert out["x"].to_list() == [1.0, 9.223372036854776e18]


# ── Projection (columns=) ────────────────────────────────────────────


def test_columns_projection_selects_subset(tmp_path: Path) -> None:
    fp = _write(tmp_path, "a,b,c\n1,true,x\n2,false,y\n")
    dest = tmp_path / "out.parquet"
    schema = convert_csv_to_parquet(fp, dest, columns=["a", "b"])
    out = pl.read_parquet(dest)
    assert out.columns == ["a", "b"]
    assert dict(schema) == {"a": pl.Int64, "b": pl.Boolean}
    assert out["b"].to_list() == [True, False]


def test_columns_projection_missing_column_raises(tmp_path: Path) -> None:
    fp = _write(tmp_path, "a,b\n1,2\n")
    with pytest.raises(ValueError, match=r"missing requested column\(s\) \['nope'\]"):
        convert_csv_to_parquet(fp, tmp_path / "out.parquet", columns=["nope"])
