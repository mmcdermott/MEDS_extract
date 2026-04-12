"""Tests for dftly integration in MEDS_extract.

Tests the config module and the dftly-powered features in convert_to_MEDS_events.
"""

import polars as pl
import pytest
from dftly import Parser

from MEDS_extract.config import FileConfig
from MEDS_extract.convert_to_MEDS_events.convert_to_MEDS_events import convert_to_events, extract_event

_ = pl.Config.set_tbl_width_chars(600)


# ── FileConfig.subject_id_polars_expr ─────────────────────────────────


class TestSubjectIdExpr:
    def test_hash(self):
        fc = FileConfig(defaults={"subject_id": "hash($mrn)"})
        assert fc.subject_id_column == "mrn"
        df = pl.DataFrame({"mrn": ["ABC", "DEF", "ABC"]})
        result = df.select(subject_id=fc.subject_id_polars_expr)
        assert result.schema["subject_id"] == pl.Int64
        assert result["subject_id"][0] == result["subject_id"][2]
        assert result["subject_id"][0] != result["subject_id"][1]

    def test_hash_deterministic(self):
        fc = FileConfig(defaults={"subject_id": "hash($mrn)"})
        df = pl.DataFrame({"mrn": ["ABC"]})
        r1 = df.select(subject_id=fc.subject_id_polars_expr)["subject_id"][0]
        r2 = df.select(subject_id=fc.subject_id_polars_expr)["subject_id"][0]
        assert r1 == r2

    def test_non_hash_column_ref(self):
        """Non-hash expressions should pass through without reinterpret."""
        fc = FileConfig(defaults={"subject_id": "$patient_id"})
        assert fc.subject_id_column == "patient_id"
        df = pl.DataFrame({"patient_id": [100, 200]})
        result = df.select(subject_id=fc.subject_id_polars_expr)
        assert result["subject_id"].to_list() == [100, 200]

    def test_no_subject_id(self):
        fc = FileConfig(defaults={})
        assert fc.subject_id_polars_expr is None
        assert fc.subject_id_column == "subject_id"


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
        assert result["code"].to_list() == ["Lab//mg", "Vital//mmHg"]
        assert "code_components" in result.columns
        assert result["code_components"].struct.field("test_name").to_list() == ["Lab", "Vital"]

    def test_literal_code(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1, 2],
                "time": ["2021-01-01", "2021-01-02"],
            }
        )
        cfg = {"code": "ADMISSION", "time": '$time::"%Y-%m-%d"'}
        result = extract_event(raw, cfg)
        assert result["code"].to_list() == ["ADMISSION", "ADMISSION"]

    def test_static_event(self):
        raw = pl.DataFrame({"subject_id": [1, 2], "color": ["blue", "green"]})
        cfg = {"code": "EYE_COLOR", "time": None, "color": "$color"}
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
            extract_event(raw, {"code": "X"})

    def test_non_string_value_raises(self):
        raw = pl.DataFrame({"subject_id": [1], "time": ["2021-01-01"]})
        with pytest.raises(ValueError, match="must be a string"):
            extract_event(raw, {"code": "X", "time": '$time::"%Y-%m-%d"', "val": 42})

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
        cfg = {"code": "EVENT", "time": '$time::"%Y-%m-%d"'}
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
            "code": "MEAS",
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
            "code": "MEAS",
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
        cfg = {"code": "EVENT", "time": '$time::"%Y-%m-%d"'}
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
            "code": "EVENT",
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
        cfg = {"code": "HR", "time": '$time::"%Y-%m-%d"'}
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
            "code": "MEAS",
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
            "code": "MEAS",
            "time": '$time::"%Y-%m-%d"',
            "text_value": "$val::str",
        }
        result = extract_event(raw, cfg)
        assert result.schema["text_value"] == pl.String
        assert result["text_value"][0] == "1.5"

    def test_literal_time_expression(self):
        """Tests a time expression that references no columns (e.g., a hardcoded date literal)."""
        raw = pl.DataFrame({"subject_id": [1, 2], "color": ["blue", "green"]})
        cfg = {"code": "EYE_COLOR", "time": '"2021-01-01"::"%Y-%m-%d"'}
        result = extract_event(raw, cfg)
        assert len(result) == 2
        assert result["time"][0] == result["time"][1]

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
        cfg = {"code": "MEDS_DEATH", "time": '$dod::"%Y-%m-%dT%H:%M:%S"'}
        result = extract_event(lf, cfg).collect()

        assert len(result) == 1
        assert result["subject_id"][0] == 1
        assert result["code"][0] == "MEDS_DEATH"

    def test_time_from_non_string_column(self):
        """Regression: time null filter must not compare non-string columns with empty string (#68)."""
        raw = pl.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "year_of_birth": [1980, None, 1990],
            }
        )
        cfg = {"code": "MEDS_BIRTH", "time": "$year_of_birth::year"}
        result = extract_event(raw, cfg)
        # Should filter out the null row, not crash with ComputeError
        assert len(result) == 2
        assert set(result["subject_id"].to_list()) == {1, 3}


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
            "admit": {"code": "ADMISSION", "time": '$ts::"%Y-%m-%d"'},
            "color": {"code": "EYE_COLOR", "time": None, "eye_color": "$color"},
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
            "event": {"code": "X", "time": '$ts::"%Y-%m-%d"'},
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
            "with_val": {"code": "A", "time": '$ts::"%Y-%m-%d"', "numeric_value": "$val"},
            "with_color": {"code": "B", "time": None, "eye_color": "$color"},
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
                "code": "MEASUREMENT",
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

        fc = FileConfig(defaults={"subject_id": "hash($mrn)"})
        df_with_sid = raw.with_columns(subject_id=fc.subject_id_polars_expr)

        result = convert_to_events(df_with_sid, {"event": {"code": "X", "time": '$time::"%Y-%m-%d"'}})
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
