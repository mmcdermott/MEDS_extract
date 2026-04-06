"""Tests for dftly integration in MEDS_extract.

Tests the dftly_bridge module and the dftly-powered features in convert_to_MEDS_events.
"""

import polars as pl

from MEDS_extract.convert_to_MEDS_events.convert_to_MEDS_events import convert_to_events, extract_event
from MEDS_extract.dftly_bridge import (
    compile_code_interpolation,
    compile_subject_id_expr,
    compile_transforms,
    extract_columns_from_dftly_value_schemaless,
    is_dftly_expr,
    polars_schema_to_dftly_schema,
)

_ = pl.Config.set_tbl_width_chars(600)


class TestPolarsSchemaMapping:
    def test_basic_types(self):
        schema = {"a": pl.Int64, "b": pl.Utf8, "c": pl.Float64, "d": pl.Boolean, "e": pl.Date}
        result = polars_schema_to_dftly_schema(schema)
        assert result == {"a": "int", "b": "str", "c": "float", "d": "bool", "e": "date"}

    def test_unknown_type(self):
        schema = {"a": pl.Categorical}
        result = polars_schema_to_dftly_schema(schema)
        assert result == {"a": None}

    def test_unsigned_ints(self):
        schema = {"a": pl.UInt8, "b": pl.UInt64}
        result = polars_schema_to_dftly_schema(schema)
        assert result == {"a": "int", "b": "int"}


class TestIsDftlyExpr:
    def test_legacy_col_ref(self):
        assert not is_dftly_expr("col(timestamp)")

    def test_plain_column_name(self):
        assert not is_dftly_expr("timestamp")

    def test_literal_string(self):
        assert not is_dftly_expr("ADMISSION")

    def test_arithmetic(self):
        assert is_dftly_expr("col1 + col2")

    def test_type_cast(self):
        assert is_dftly_expr("col1 as float")

    def test_hash(self):
        assert is_dftly_expr("hash(mrn)")

    def test_string_interpolation(self):
        assert is_dftly_expr("{test_name} // {units}")

    def test_timestamp_resolution(self):
        assert is_dftly_expr("date_col @ time_col")

    def test_conditional(self):
        assert is_dftly_expr("val if flag else 0")


class TestSchemalessColumnExtraction:
    def test_arithmetic(self):
        assert extract_columns_from_dftly_value_schemaless("a + b") == {"a", "b"}

    def test_hash(self):
        assert extract_columns_from_dftly_value_schemaless("hash(mrn)") == {"mrn"}

    def test_interpolation(self):
        assert extract_columns_from_dftly_value_schemaless("{test} // {unit}") == {"test", "unit"}

    def test_cast_filters_type_keyword(self):
        result = extract_columns_from_dftly_value_schemaless("col1 as float")
        assert "col1" in result
        assert "float" not in result

    def test_conditional_filters_keywords(self):
        result = extract_columns_from_dftly_value_schemaless("val if flag else default_col")
        assert result == {"val", "flag", "default_col"}


class TestCompileTransforms:
    def test_basic_arithmetic(self):
        exprs = compile_transforms({"total": "a + b"}, {"a": "int", "b": "int"})
        df = pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        result = df.select(**exprs)
        assert result["total"].to_list() == [11, 22, 33]

    def test_string_interpolation(self):
        exprs = compile_transforms(
            {"code": "{prefix} // {suffix}"},
            {"prefix": "str", "suffix": "str"},
        )
        df = pl.DataFrame({"prefix": ["LAB", "VITAL"], "suffix": ["mg", "mmHg"]})
        result = df.select(**exprs)
        assert result["code"].to_list() == ["LAB // mg", "VITAL // mmHg"]


class TestCompileCodeInterpolation:
    def test_basic(self):
        expr, null_filter, cols = compile_code_interpolation(
            "{test} // {unit}", {"test": "str", "unit": "str"}
        )
        df = pl.DataFrame({"test": ["Lab", "Vital"], "unit": ["mg", "mmHg"]})
        result = df.select(code=expr)
        assert result["code"].to_list() == ["Lab // mg", "Vital // mmHg"]
        assert cols == {"test", "unit"}
        assert null_filter is not None

    def test_with_null_data(self):
        expr, null_filter, _ = compile_code_interpolation(
            "{test} // {unit}", {"test": "str", "unit": "str"}
        )
        df = pl.DataFrame({"test": [None, "Vital"], "unit": ["mg", "mmHg"]})
        if null_filter is not None:
            df = df.filter(null_filter)
        result = df.select(code=expr)
        assert len(result) == 1


class TestCompileSubjectIdExpr:
    def test_hash(self):
        expr, cols = compile_subject_id_expr("hash(mrn)", {"mrn": "str"})
        assert cols == {"mrn"}
        df = pl.DataFrame({"mrn": ["ABC", "DEF", "ABC"]})
        result = df.select(subject_id=expr)
        assert result.schema["subject_id"] == pl.Int64
        # Same input should produce same hash
        assert result["subject_id"][0] == result["subject_id"][2]
        # Different inputs should produce different hashes
        assert result["subject_id"][0] != result["subject_id"][1]


class TestExtractEventWithDftly:
    """Tests that extract_event works with dftly expressions in config values."""

    def test_dftly_code_interpolation(self):
        """Test code field as dftly string interpolation."""
        raw_data = pl.DataFrame({
            "subject_id": [1, 2],
            "test_name": ["Lab", "Vital"],
            "units": ["mg", "mmHg"],
            "time": ["2021-01-01", "2021-01-02"],
        })
        event_cfg = {
            "code": "{test_name} // {units}",
            "time": "col(time)",
            "time_format": "%Y-%m-%d",
        }
        result = extract_event(raw_data, event_cfg)
        assert result.shape == (2, 3)
        codes = result["code"].to_list()
        assert codes == ["Lab // mg", "Vital // mmHg"]

    def test_dftly_value_expression(self):
        """Test value field as dftly expression (e.g., arithmetic)."""
        raw_data = pl.DataFrame({
            "subject_id": [1, 2],
            "code_col": ["A", "B"],
            "val1": [10, 20],
            "val2": [3, 7],
            "time": ["2021-01-01", "2021-01-02"],
        })
        event_cfg = {
            "code": "col(code_col)",
            "time": "col(time)",
            "time_format": "%Y-%m-%d",
            "numeric_value": "val1 + val2",
        }
        result = extract_event(raw_data, event_cfg)
        assert result.shape == (2, 4)
        assert result["numeric_value"].to_list() == [13, 27]

    def test_legacy_config_unchanged(self):
        """Verify legacy configs still work identically."""
        raw_data = pl.DataFrame({
            "subject_id": [1, 2],
            "code_col": ["A", "B"],
            "time_col": ["2021-01-01", "2021-01-02"],
            "value": [1.0, 2.0],
        })
        event_cfg = {
            "code": ["PREFIX", "col(code_col)"],
            "time": "col(time_col)",
            "time_format": "%Y-%m-%d",
            "numeric_value": "value",
        }
        result = extract_event(raw_data, event_cfg)
        assert result.shape == (2, 4)
        codes = result["code"].to_list()
        assert codes == ["PREFIX//A", "PREFIX//B"]


class TestConvertToEventsWithTransforms:
    """Tests that convert_to_events works with dftly features at the compute_fn level.

    These test the transforms and subject_id_expr handling that happens in main()'s compute_fn.
    We simulate what compute_fn does by manually applying transforms before calling convert_to_events.
    """

    def test_transforms_derive_column(self):
        """Test that a transforms block can derive a column used by events."""
        raw_data = pl.DataFrame({
            "subject_id": [1, 1, 2],
            "val1": [10, 20, 30],
            "val2": [1, 2, 3],
            "time": ["2021-01-01", "2021-01-02", "2021-01-03"],
        })

        # Simulate what main() does: apply transforms, then extract events
        dftly_schema = polars_schema_to_dftly_schema(raw_data.schema)
        transform_exprs = compile_transforms({"total": "val1 + val2"}, dftly_schema)
        df_with_transforms = raw_data.with_columns(**transform_exprs)

        event_cfgs = {
            "measurement": {
                "code": "MEASUREMENT",
                "time": "col(time)",
                "time_format": "%Y-%m-%d",
                "numeric_value": "total",
            }
        }
        result = convert_to_events(df_with_transforms, event_cfgs)
        assert result["numeric_value"].to_list() == [11, 22, 33]

    def test_subject_id_expr_hash(self):
        """Test that subject_id_expr with hash produces int64 subject IDs."""
        raw_data = pl.DataFrame({
            "mrn": ["ABC", "DEF", "ABC"],
            "code_col": ["X", "Y", "Z"],
            "time": ["2021-01-01", "2021-01-02", "2021-01-03"],
        })

        # Simulate compute_fn: apply subject_id_expr
        dftly_schema = polars_schema_to_dftly_schema(raw_data.schema)
        sid_expr, _ = compile_subject_id_expr("hash(mrn)", dftly_schema)
        df_with_sid = raw_data.with_columns(subject_id=sid_expr)

        event_cfgs = {
            "event": {
                "code": "col(code_col)",
                "time": "col(time)",
                "time_format": "%Y-%m-%d",
            }
        }
        result = convert_to_events(df_with_sid, event_cfgs)
        assert result.schema["subject_id"] == pl.Int64
        # Same MRN → same subject_id
        assert result["subject_id"][0] == result["subject_id"][2]

    def test_mixed_legacy_and_dftly(self):
        """Test config with both legacy and dftly syntax in different events."""
        raw_data = pl.DataFrame({
            "subject_id": [1, 2],
            "code_col": ["A", "B"],
            "test_name": ["Lab", "Vital"],
            "units": ["mg", "mmHg"],
            "time": ["2021-01-01", "2021-01-02"],
            "value": [1.0, 2.0],
        })

        event_cfgs = {
            # Legacy syntax
            "legacy_event": {
                "code": ["PREFIX", "col(code_col)"],
                "time": "col(time)",
                "time_format": "%Y-%m-%d",
                "numeric_value": "value",
            },
            # dftly syntax
            "dftly_event": {
                "code": "{test_name} // {units}",
                "time": "col(time)",
                "time_format": "%Y-%m-%d",
            },
        }
        result = convert_to_events(raw_data, event_cfgs)
        # Should have 4 rows: 2 from each event
        assert result.shape[0] == 4
        codes = sorted(result["code"].to_list())
        assert "Lab // mg" in codes
        assert "PREFIX//A" in codes
