"""Round-trip parity tests for ``MEDS_extract.io.convert_csv_to_parquet``.

The conversion promises to type and decode a CSV exactly as polars' own full-file
inference would: for any CSV file that ``pl.read_csv(fp, infer_schema_length=None)``
can read, converting it and reading the parquet back must yield the identical frame —
same schema, same values. The hypothesis property test here enforces that promise
end-to-end over generated CSVs drawn from a token pool built to stress every acceptor
edge polars has: boolean casings, ``+``-signed numbers, leading zeros, trailing-dot
and exponent float forms, ``inf``/``NaN`` casings, whitespace padding, quoted fields,
unicode, and empty-vs-missing cells.

The sole deliberate deviation, excluded from the property and pinned deterministically
instead: an integer-shaped column overflowing Int64 makes polars' full-file read
hard-error, while the conversion degrades it to (lossy) Float64.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

from MEDS_extract.io import convert_csv_to_parquet

# ── Token pools ──────────────────────────────────────────────────────
# Grouped by the column dtype they *tend* to produce, so generated columns are often
# homogeneous enough to actually exercise the Int64/Float64/Boolean rungs rather than
# collapsing to String. Every pool deliberately includes near-miss tokens too.

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
# otherwise integer-shaped; kept in the mixed pool so the ``assume`` exclusion path
# is exercised too.
OVERFLOW_TOKENS = ["9223372036854775808", "99999999999999999999", "-9223372036854775809"]
ALL_TOKENS = INT_TOKENS + FLOAT_TOKENS + BOOL_TOKENS + DATE_TOKENS + TEXT_TOKENS + OVERFLOW_TOKENS

# NUL bytes are excluded: CSV cannot carry them faithfully, so they only test the
# writer, not the conversion. Everything else — newlines, quotes, commas, unicode —
# stays in play and rides through ``write_csv``'s quoting.
_arbitrary_text = st.text(st.characters(codec="utf-8", exclude_characters="\x00"), max_size=8)


@st.composite
def string_frames(draw: st.DrawFn) -> pl.DataFrame:
    """An all-String DataFrame whose ``write_csv`` output is the file under test."""
    n_rows = draw(st.integers(min_value=1, max_value=30))
    n_cols = draw(st.integers(min_value=1, max_value=4))
    cols: dict[str, list[str | None]] = {}
    for i in range(n_cols):
        pool = draw(
            st.sampled_from([INT_TOKENS, FLOAT_TOKENS, BOOL_TOKENS, DATE_TOKENS, TEXT_TOKENS, ALL_TOKENS])
        )
        elem = st.none() | st.sampled_from(pool)
        if pool is TEXT_TOKENS or pool is ALL_TOKENS:
            elem |= _arbitrary_text
        cols[f"c{i}"] = draw(st.lists(elem, min_size=n_rows, max_size=n_rows))
    return pl.DataFrame(cols, schema=dict.fromkeys(cols, pl.String))


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
