"""Tests for dftly integration in MEDS_extract.

Tests the dftly_bridge module and the dftly-powered features in convert_to_MEDS_events.
"""

import polars as pl
import pytest
from dftly import Parser

from MEDS_extract.convert_to_MEDS_events.convert_to_MEDS_events import convert_to_events, extract_event
from MEDS_extract.dftly_bridge import compile_subject_id_expr

_ = pl.Config.set_tbl_width_chars(600)


# ── Compilation functions ──────────────────────────────────────────────


class TestParserToPolars:
    def test_basic_arithmetic(self):
        exprs = Parser.to_polars({"total": "$a + $b"})
        df = pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        result = df.select(**exprs)
        assert result["total"].to_list() == [11, 22, 33]

    def test_string_interpolation(self):
        exprs = Parser.to_polars({"code": 'f"{$prefix}//{$suffix}"'})
        df = pl.DataFrame({"prefix": ["LAB", "VITAL"], "suffix": ["mg", "mmHg"]})
        result = df.select(**exprs)
        assert result["code"].to_list() == ["LAB//mg", "VITAL//mmHg"]

    def test_multiple_transforms(self):
        exprs = Parser.to_polars({"sum": "$a + $b", "label": "$name"})
        df = pl.DataFrame({"a": [1], "b": [2], "name": ["test"]})
        result = df.with_columns(**exprs)
        assert result["sum"].to_list() == [3]
        assert result["label"].to_list() == ["test"]


class TestParserExprToPolars:
    def test_interpolation(self):
        node = Parser()('f"{$test}//{$unit}"')
        expr, cols = node.polars_expr, node.referenced_columns
        df = pl.DataFrame({"test": ["Lab", "Vital"], "unit": ["mg", "mmHg"]})
        result = df.select(code=expr)
        assert result["code"].to_list() == ["Lab//mg", "Vital//mmHg"]
        assert cols == {"test", "unit"}

    def test_literal_no_columns(self):
        node = Parser()('"ADMISSION"')
        assert node.referenced_columns == set()

    def test_column_ref(self):
        expr = Parser.expr_to_polars("$ts")
        df = pl.DataFrame({"ts": ["2021-01-01"]})
        result = df.select(val=expr)
        assert result["val"].to_list() == ["2021-01-01"]

    def test_type_cast(self):
        expr = Parser.expr_to_polars("$x::float")
        df = pl.DataFrame({"x": ["1.5", "2.7"]})
        result = df.select(val=expr)
        assert result["val"].dtype in (pl.Float32, pl.Float64)
        assert abs(result["val"][0] - 1.5) < 1e-6
        assert abs(result["val"][1] - 2.7) < 1e-6

    def test_time_format_parse(self):
        expr = Parser.expr_to_polars('$ts::"%Y-%m-%d"')
        df = pl.DataFrame({"ts": ["2021-01-01", "2021-06-15"]})
        result = df.select(time=expr)
        assert result.schema["time"] == pl.Date


class TestCompileSubjectIdExpr:
    def test_hash(self):
        expr, cols = compile_subject_id_expr("hash($mrn)")
        assert cols == {"mrn"}
        df = pl.DataFrame({"mrn": ["ABC", "DEF", "ABC"]})
        result = df.select(subject_id=expr)
        assert result.schema["subject_id"] == pl.Int64
        assert result["subject_id"][0] == result["subject_id"][2]
        assert result["subject_id"][0] != result["subject_id"][1]

    def test_hash_deterministic(self):
        expr, _ = compile_subject_id_expr("hash($mrn)")
        df = pl.DataFrame({"mrn": ["ABC"]})
        r1 = df.select(subject_id=expr)["subject_id"][0]
        r2 = df.select(subject_id=expr)["subject_id"][0]
        assert r1 == r2


# ── extract_event() ────────────────────────────────────────────────────


class TestExtractEvent:
    def test_code_interpolation(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1, 2],
                "test_name": ["Lab", "Vital"],
                "units": ["mg", "mmHg"],
                "time": ["2021-01-01", "2021-01-02"],
            }
        )
        cfg = {"code": 'f"{$test_name}//{$units}"', "time": '$time::"%Y-%m-%d"'}
        result = extract_event(raw, cfg)
        assert result.shape == (2, 3)
        assert result["code"].to_list() == ["Lab//mg", "Vital//mmHg"]

    def test_literal_code(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1, 2],
                "time": ["2021-01-01", "2021-01-02"],
            }
        )
        cfg = {"code": '"ADMISSION"', "time": '$time::"%Y-%m-%d"'}
        result = extract_event(raw, cfg)
        assert result["code"].to_list() == ["ADMISSION", "ADMISSION"]

    def test_static_event(self):
        raw = pl.DataFrame({"subject_id": [1, 2], "color": ["blue", "green"]})
        cfg = {"code": '"EYE_COLOR"', "time": None, "color": "$color"}
        result = extract_event(raw, cfg)
        assert result["time"].null_count() == 2
        assert result["color"].to_list() == ["blue", "green"]

    def test_missing_code_raises(self):
        raw = pl.DataFrame({"subject_id": [1]})
        with pytest.raises(KeyError, match="code"):
            extract_event(raw, {"time": None})

    def test_missing_time_raises(self):
        raw = pl.DataFrame({"subject_id": [1]})
        with pytest.raises(KeyError, match="time"):
            extract_event(raw, {"code": '"X"'})

    def test_non_string_value_raises(self):
        raw = pl.DataFrame({"subject_id": [1], "time": ["2021-01-01"]})
        with pytest.raises(ValueError, match="must be a string"):
            extract_event(raw, {"code": '"X"', "time": '$time::"%Y-%m-%d"', "val": 42})

    def test_null_code_rows_filtered(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "name": ["A", None, "C"],
                "time": ["2021-01-01", "2021-01-02", "2021-01-03"],
            }
        )
        cfg = {"code": 'f"{$name}"', "time": '$time::"%Y-%m-%d"'}
        result = extract_event(raw, cfg)
        # Row with null name should be filtered out
        assert result.shape[0] == 2
        assert result["code"].to_list() == ["A", "C"]

    def test_null_time_rows_filtered(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "time": ["2021-01-01", None, "2021-01-03"],
            }
        )
        cfg = {"code": '"EVENT"', "time": '$time::"%Y-%m-%d"'}
        result = extract_event(raw, cfg)
        assert result.shape[0] == 2

    def test_dedup_text_and_numeric(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1, 2],
                "time": ["2021-01-01", "2021-01-02"],
                "val": [1.5, 2.0],
                "text": ["1.5", "other"],
            }
        )
        cfg = {
            "code": '"MEAS"',
            "time": '$time::"%Y-%m-%d"',
            "numeric_value": "$val",
            "text_value": "$text",
        }
        result = extract_event(raw, cfg, do_dedup_text_and_numeric=True)
        # "1.5" matches 1.5 numerically → should be nulled
        assert result["text_value"][0] is None
        # "other" doesn't match 2.0 → should remain
        assert result["text_value"][1] == "other"

    def test_dedup_text_and_numeric_no_dedup(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1],
                "time": ["2021-01-01"],
                "val": [1.5],
                "text": ["1.5"],
            }
        )
        cfg = {
            "code": '"MEAS"',
            "time": '$time::"%Y-%m-%d"',
            "numeric_value": "$val",
            "text_value": "$text",
        }
        result = extract_event(raw, cfg, do_dedup_text_and_numeric=False)
        # Without dedup, text_value should remain
        assert result["text_value"][0] == "1.5"

    def test_duplicate_rows_removed(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1, 1],
                "time": ["2021-01-01", "2021-01-01"],
            }
        )
        cfg = {"code": '"EVENT"', "time": '$time::"%Y-%m-%d"'}
        result = extract_event(raw, cfg)
        assert result.shape[0] == 1

    def test_meta_keys_ignored(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1],
                "time": ["2021-01-01"],
            }
        )
        cfg = {
            "code": '"EVENT"',
            "time": '$time::"%Y-%m-%d"',
            "_metadata": {"some": "thing"},
        }
        result = extract_event(raw, cfg)
        assert "_metadata" not in result.columns

    def test_code_bare_name_is_literal_not_column(self):
        """Bare identifiers in code field should be literals, not column references."""
        raw = pl.DataFrame(
            {
                "subject_id": [1, 2],
                "HR": [100.5, 80.2],
                "time": ["2021-01-01", "2021-01-02"],
            }
        )
        cfg = {"code": '"HR"', "time": '$time::"%Y-%m-%d"'}
        result = extract_event(raw, cfg)
        # "HR" should be a literal string, NOT the column values
        assert result["code"].to_list() == ["HR", "HR"]

    def test_dftly_expression_in_value(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1, 2],
                "val1": [10, 20],
                "val2": [3, 7],
                "time": ["2021-01-01", "2021-01-02"],
            }
        )
        cfg = {
            "code": '"MEAS"',
            "time": '$time::"%Y-%m-%d"',
            "numeric_value": "$val1 + $val2",
        }
        result = extract_event(raw, cfg)
        assert result["numeric_value"].to_list() == [13, 27]

    def test_type_cast_in_value(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1],
                "val": [1.5],
                "time": ["2021-01-01"],
            }
        )
        cfg = {
            "code": '"MEAS"',
            "time": '$time::"%Y-%m-%d"',
            "text_value": "$val::str",
        }
        result = extract_event(raw, cfg)
        assert result.schema["text_value"] == pl.String
        assert result["text_value"][0] == "1.5"

    def test_time_with_nulls_via_scan_parquet(self, tmp_path):
        """Regression test: strptime on null-heavy time columns must not crash with scan_parquet.

        Polars predicate pushdown can cause strptime(strict=True) to evaluate on empty strings
        during parquet scanning. The fix filters on source column nulls before applying strptime.
        """
        raw = pl.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "dod": ["2018-11-01T00:00:00", None, None],
            }
        )
        fp = tmp_path / "test.parquet"
        raw.write_parquet(fp)

        lf = pl.scan_parquet(fp, glob=False)
        cfg = {"code": '"MEDS_DEATH"', "time": '$dod::"%Y-%m-%dT%H:%M:%S"'}
        result = extract_event(lf, cfg).collect()

        assert len(result) == 1
        assert result["subject_id"][0] == 1
        assert result["code"][0] == "MEDS_DEATH"


# ── convert_to_events() ───────────────────────────────────────────────


class TestConvertToEvents:
    def test_empty_config_raises(self):
        df = pl.DataFrame({"subject_id": [1]})
        with pytest.raises(ValueError, match="No event configurations"):
            convert_to_events(df, {})

    def test_invalid_event_raises(self):
        df = pl.DataFrame({"subject_id": [1]})
        with pytest.raises(ValueError, match="Error extracting event"):
            convert_to_events(df, {"bad": {}})

    def test_multiple_events_concatenated(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1, 2],
                "dept": ["CARDIAC", "PULM"],
                "ts": ["2021-01-01", "2021-01-02"],
                "color": ["blue", "green"],
            }
        )
        cfgs = {
            "admit": {"code": '"ADMISSION"', "time": '$ts::"%Y-%m-%d"'},
            "color": {"code": '"EYE_COLOR"', "time": None, "eye_color": "$color"},
        }
        result = convert_to_events(raw, cfgs)
        assert result.shape[0] == 4
        codes = sorted(result["code"].to_list())
        assert codes == ["ADMISSION", "ADMISSION", "EYE_COLOR", "EYE_COLOR"]

    def test_meta_keys_skipped_in_event_loop(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1],
                "ts": ["2021-01-01"],
            }
        )
        cfgs = {
            "_metadata": {"should": "be skipped"},
            "event": {"code": '"X"', "time": '$ts::"%Y-%m-%d"'},
        }
        result = convert_to_events(raw, cfgs)
        assert result.shape[0] == 1

    def test_diagonal_concat_fills_nulls(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1],
                "ts": ["2021-01-01"],
                "val": [1.5],
                "color": ["blue"],
            }
        )
        cfgs = {
            "with_val": {"code": '"A"', "time": '$ts::"%Y-%m-%d"', "numeric_value": "$val"},
            "with_color": {"code": '"B"', "time": None, "eye_color": "$color"},
        }
        result = convert_to_events(raw, cfgs)
        assert result.shape == (2, 5)
        # Columns from different events should have nulls
        assert result.filter(pl.col("code") == "A")["eye_color"][0] is None
        assert result.filter(pl.col("code") == "B")["numeric_value"][0] is None


# ── Integration: transforms + subject_id_expr ──────────────────────────


class TestIntegrationWithTransforms:
    def test_transforms_derive_column(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1, 1, 2],
                "val1": [10, 20, 30],
                "val2": [1, 2, 3],
                "time": ["2021-01-01", "2021-01-02", "2021-01-03"],
            }
        )

        transform_exprs = Parser.to_polars({"total": "$val1 + $val2"})
        df_with_transforms = raw.with_columns(**transform_exprs)

        event_cfgs = {
            "measurement": {
                "code": '"MEASUREMENT"',
                "time": '$time::"%Y-%m-%d"',
                "numeric_value": "$total",
            }
        }
        result = convert_to_events(df_with_transforms, event_cfgs)
        assert result["numeric_value"].to_list() == [11, 22, 33]

    def test_subject_id_expr_hash(self):
        raw = pl.DataFrame(
            {
                "mrn": ["ABC", "DEF", "ABC"],
                "time": ["2021-01-01", "2021-01-02", "2021-01-03"],
            }
        )

        sid_expr, _ = compile_subject_id_expr("hash($mrn)")
        df_with_sid = raw.with_columns(subject_id=sid_expr)

        result = convert_to_events(df_with_sid, {"event": {"code": '"X"', "time": '$time::"%Y-%m-%d"'}})
        assert result.schema["subject_id"] == pl.Int64
        assert result["subject_id"][0] == result["subject_id"][2]

    def test_mixed_event_types(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1, 2],
                "test_name": ["Lab", "Vital"],
                "units": ["mg", "mmHg"],
                "time": ["2021-01-01", "2021-01-02"],
                "value": [1.0, 2.0],
            }
        )

        event_cfgs = {
            "interpolation_event": {
                "code": 'f"PREFIX//{$test_name}"',
                "time": '$time::"%Y-%m-%d"',
                "numeric_value": "$value",
            },
            "dftly_event": {
                "code": 'f"{$test_name}//{$units}"',
                "time": '$time::"%Y-%m-%d"',
            },
        }
        result = convert_to_events(raw, event_cfgs)
        assert result.shape[0] == 4
        codes = sorted(result["code"].to_list())
        assert "Lab//mg" in codes
        assert "PREFIX//Lab" in codes
