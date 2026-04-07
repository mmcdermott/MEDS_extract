"""Tests for dftly integration in MEDS_extract.

Tests the dftly_bridge module and the dftly-powered features in convert_to_MEDS_events.
"""

import polars as pl
import pytest
from dftly import Column, Expression, Literal

from MEDS_extract.convert_to_MEDS_events.convert_to_MEDS_events import convert_to_events, extract_event
from MEDS_extract.dftly_bridge import (
    compile_field_expr,
    compile_field_expr_with_columns,
    compile_subject_id_expr,
    compile_transforms,
    extract_columns_schemaless,
    extract_columns_schemaless_code,
    get_referenced_columns,
    polars_schema_to_dftly_schema,
)

_ = pl.Config.set_tbl_width_chars(600)


# ── Schema mapping ──────────────────────────────────────────────────────


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

    def test_datetime_and_duration(self):
        schema = {"a": pl.Datetime, "b": pl.Duration}
        result = polars_schema_to_dftly_schema(schema)
        assert result == {"a": "datetime", "b": "duration"}

    def test_empty_schema(self):
        assert polars_schema_to_dftly_schema({}) == {}


# ── AST column extraction ──────────────────────────────────────────────


class TestGetReferencedColumns:
    def test_column_node(self):
        assert get_referenced_columns(Column(name="a", type="int")) == {"a"}

    def test_literal_node(self):
        assert get_referenced_columns(Literal(value=42)) == set()

    def test_expression_with_list_args(self):
        node = Expression(
            type="ADD",
            arguments=[Column(name="x", type="int"), Column(name="y", type="int")],
        )
        assert get_referenced_columns(node) == {"x", "y"}

    def test_expression_with_dict_args(self):
        node = Expression(
            type="CONDITIONAL",
            arguments={
                "condition": Column(name="flag", type="bool"),
                "true_value": Column(name="a", type="int"),
                "false_value": Literal(value=0),
            },
        )
        assert get_referenced_columns(node) == {"a", "flag"}

    def test_nested_expression(self):
        inner = Expression(type="ADD", arguments=[Column(name="a", type="int"), Literal(value=1)])
        outer = Expression(
            type="CONDITIONAL",
            arguments={
                "condition": Column(name="flag", type="bool"),
                "true_value": inner,
                "false_value": Column(name="b", type="int"),
            },
        )
        assert get_referenced_columns(outer) == {"a", "b", "flag"}

    def test_dict_of_nodes(self):
        nodes = {"x": Column(name="a", type="int"), "y": Literal(value=5)}
        assert get_referenced_columns(nodes) == {"a"}

    def test_list_of_nodes(self):
        nodes = [Column(name="a", type="int"), Column(name="b", type="str")]
        assert get_referenced_columns(nodes) == {"a", "b"}

    def test_empty_inputs(self):
        assert get_referenced_columns({}) == set()
        assert get_referenced_columns([]) == set()


# ── Schemaless column extraction ───────────────────────────────────────


class TestSchemalessColumnExtraction:
    def test_arithmetic(self):
        assert extract_columns_schemaless("a + b") == {"a", "b"}

    def test_hash(self):
        assert extract_columns_schemaless("hash(mrn)") == {"mrn"}

    def test_interpolation(self):
        assert extract_columns_schemaless("{test} // {unit}") == {"test", "unit"}

    def test_cast_filters_type_keyword(self):
        result = extract_columns_schemaless("col1 as float")
        assert "col1" in result
        assert "float" not in result

    def test_conditional_filters_keywords(self):
        result = extract_columns_schemaless("val if flag else default_col")
        assert result == {"val", "flag", "default_col"}

    def test_format_string_ignored(self):
        result = extract_columns_schemaless('charttime as "%m/%d/%Y, %H:%M:%S"')
        assert result == {"charttime"}

    def test_bare_identifier(self):
        assert extract_columns_schemaless("ADMISSION") == {"ADMISSION"}

    def test_all_keywords_filtered(self):
        result = extract_columns_schemaless("not true and false or null")
        assert result == set()


class TestSchemalessCodeColumnExtraction:
    def test_literal_code(self):
        assert extract_columns_schemaless_code("ADMISSION") == set()

    def test_interpolation_code(self):
        assert extract_columns_schemaless_code("{test} // {unit}") == {"test", "unit"}

    def test_bare_column_name_treated_as_literal(self):
        assert extract_columns_schemaless_code("test_name") == set()

    def test_mixed_literal_and_interpolation(self):
        assert extract_columns_schemaless_code("PREFIX // {col}") == {"col"}


# ── Compilation functions ──────────────────────────────────────────────


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

    def test_multiple_transforms(self):
        exprs = compile_transforms(
            {"sum": "a + b", "label": "{name}"},
            {"a": "int", "b": "int", "name": "str"},
        )
        df = pl.DataFrame({"a": [1], "b": [2], "name": ["test"]})
        result = df.with_columns(**exprs)
        assert result["sum"].to_list() == [3]
        assert result["label"].to_list() == ["test"]


class TestCompileFieldExpr:
    def test_interpolation(self):
        expr, cols = compile_field_expr_with_columns(
            "code", "{test} // {unit}", {"test": "str", "unit": "str"}
        )
        df = pl.DataFrame({"test": ["Lab", "Vital"], "unit": ["mg", "mmHg"]})
        result = df.select(code=expr)
        assert result["code"].to_list() == ["Lab // mg", "Vital // mmHg"]
        assert cols == {"test", "unit"}

    def test_literal_no_columns(self):
        _, cols = compile_field_expr_with_columns("code", "ADMISSION", {"ts": "str"})
        assert cols == set()

    def test_column_ref(self):
        expr = compile_field_expr("val", "ts", {"ts": "str"})
        df = pl.DataFrame({"ts": ["2021-01-01"]})
        result = df.select(val=expr)
        assert result["val"].to_list() == ["2021-01-01"]

    def test_type_cast(self):
        expr = compile_field_expr("val", "x as float", {"x": "str"})
        df = pl.DataFrame({"x": ["1.5", "2.7"]})
        result = df.select(val=expr)
        assert result["val"].to_list() == [1.5, 2.7]

    def test_time_format_parse(self):
        expr = compile_field_expr("time", 'ts as "%Y-%m-%d"', {"ts": "str"})
        df = pl.DataFrame({"ts": ["2021-01-01", "2021-06-15"]})
        result = df.select(time=expr)
        assert result.schema["time"] == pl.Date


class TestCompileSubjectIdExpr:
    def test_hash(self):
        expr, cols = compile_subject_id_expr("hash(mrn)", {"mrn": "str"})
        assert cols == {"mrn"}
        df = pl.DataFrame({"mrn": ["ABC", "DEF", "ABC"]})
        result = df.select(subject_id=expr)
        assert result.schema["subject_id"] == pl.Int64
        assert result["subject_id"][0] == result["subject_id"][2]
        assert result["subject_id"][0] != result["subject_id"][1]

    def test_hash_deterministic(self):
        expr, _ = compile_subject_id_expr("hash(mrn)", {"mrn": "str"})
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
        cfg = {"code": "{test_name} // {units}", "time": 'time as "%Y-%m-%d"'}
        result = extract_event(raw, cfg)
        assert result.shape == (2, 3)
        assert result["code"].to_list() == ["Lab // mg", "Vital // mmHg"]

    def test_literal_code(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1, 2],
                "time": ["2021-01-01", "2021-01-02"],
            }
        )
        cfg = {"code": "ADMISSION", "time": 'time as "%Y-%m-%d"'}
        result = extract_event(raw, cfg)
        assert result["code"].to_list() == ["ADMISSION", "ADMISSION"]

    def test_static_event(self):
        raw = pl.DataFrame({"subject_id": [1, 2], "color": ["blue", "green"]})
        cfg = {"code": "EYE_COLOR", "time": None, "color": "color"}
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
            extract_event(raw, {"code": "X", "time": 'time as "%Y-%m-%d"', "val": 42})

    def test_null_code_rows_filtered(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "name": ["A", None, "C"],
                "time": ["2021-01-01", "2021-01-02", "2021-01-03"],
            }
        )
        cfg = {"code": "{name}", "time": 'time as "%Y-%m-%d"'}
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
        cfg = {"code": "EVENT", "time": 'time as "%Y-%m-%d"'}
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
            "time": 'time as "%Y-%m-%d"',
            "numeric_value": "val",
            "text_value": "text",
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
            "time": 'time as "%Y-%m-%d"',
            "numeric_value": "val",
            "text_value": "text",
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
        cfg = {"code": "EVENT", "time": 'time as "%Y-%m-%d"'}
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
            "time": 'time as "%Y-%m-%d"',
            "_metadata": {"some": "thing"},
        }
        result = extract_event(raw, cfg)
        assert "_metadata" not in result.columns

    def test_code_bare_name_is_literal_not_column(self):
        """Bare identifiers in code field should be literals, even if they match a column name."""
        raw = pl.DataFrame(
            {
                "subject_id": [1, 2],
                "HR": [100.5, 80.2],
                "time": ["2021-01-01", "2021-01-02"],
            }
        )
        cfg = {"code": "HR", "time": 'time as "%Y-%m-%d"'}
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
            "time": 'time as "%Y-%m-%d"',
            "numeric_value": "val1 + val2",
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
            "time": 'time as "%Y-%m-%d"',
            "text_value": "val as str",
        }
        result = extract_event(raw, cfg)
        assert result.schema["text_value"] == pl.String
        assert result["text_value"][0] == "1.5"


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
            "admit": {"code": "ADMISSION", "time": 'ts as "%Y-%m-%d"'},
            "color": {"code": "EYE_COLOR", "time": None, "eye_color": "color"},
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
            "event": {"code": "X", "time": 'ts as "%Y-%m-%d"'},
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
            "with_val": {"code": "A", "time": 'ts as "%Y-%m-%d"', "numeric_value": "val"},
            "with_color": {"code": "B", "time": None, "eye_color": "color"},
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

        dftly_schema = polars_schema_to_dftly_schema(raw.schema)
        transform_exprs = compile_transforms({"total": "val1 + val2"}, dftly_schema)
        df_with_transforms = raw.with_columns(**transform_exprs)

        event_cfgs = {
            "measurement": {
                "code": "MEASUREMENT",
                "time": 'time as "%Y-%m-%d"',
                "numeric_value": "total",
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

        dftly_schema = polars_schema_to_dftly_schema(raw.schema)
        sid_expr, _ = compile_subject_id_expr("hash(mrn)", dftly_schema)
        df_with_sid = raw.with_columns(subject_id=sid_expr)

        result = convert_to_events(df_with_sid, {"event": {"code": "X", "time": 'time as "%Y-%m-%d"'}})
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
                "code": "PREFIX // {test_name}",
                "time": 'time as "%Y-%m-%d"',
                "numeric_value": "value",
            },
            "dftly_event": {
                "code": "{test_name} // {units}",
                "time": 'time as "%Y-%m-%d"',
            },
        }
        result = convert_to_events(raw, event_cfgs)
        assert result.shape[0] == 4
        codes = sorted(result["code"].to_list())
        assert "Lab // mg" in codes
        assert "PREFIX // Lab" in codes
