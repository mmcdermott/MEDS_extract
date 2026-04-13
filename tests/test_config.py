"""Tests for MEDS_extract.config and the extract_event / convert_to_events helpers.

Library-contract tests only. dftly DSL semantics (interpolation, casts, bare-literal parsing) are tested in
the dftly package itself and should not be duplicated here.
"""

import polars as pl
import pytest

from MEDS_extract.config import EventConfig, MessyConfig, TableConfig
from MEDS_extract.convert_to_MEDS_events.convert_to_MEDS_events import convert_to_events, extract_event

_ = pl.Config.set_tbl_width_chars(600)


# ── TableConfig subject_id handling ──────────────────────────────────


class TestSubjectIdExpr:
    def test_hash_reinterpret(self):
        tc = TableConfig("t", "hash($mrn)")
        assert tc.subject_id_column == "mrn"
        df = pl.DataFrame({"mrn": ["ABC", "DEF", "ABC"]})
        result = df.select(subject_id=tc.subject_id_polars_expr)
        assert result.schema["subject_id"] == pl.Int64
        assert result["subject_id"][0] == result["subject_id"][2]
        assert result["subject_id"][0] != result["subject_id"][1]

    def test_hash_deterministic(self):
        tc = TableConfig("t", "hash($mrn)")
        df = pl.DataFrame({"mrn": ["ABC"]})
        r1 = df.select(subject_id=tc.subject_id_polars_expr)["subject_id"][0]
        r2 = df.select(subject_id=tc.subject_id_polars_expr)["subject_id"][0]
        assert r1 == r2

    def test_column_ref_passthrough(self):
        tc = TableConfig("t", "$patient_id")
        assert tc.subject_id_column == "patient_id"
        df = pl.DataFrame({"patient_id": [100, 200]})
        result = df.select(subject_id=tc.subject_id_polars_expr)
        assert result["subject_id"].to_list() == [100, 200]

    def test_no_subject_id(self):
        tc = TableConfig("t", None)
        assert tc.subject_id_polars_expr is None
        assert tc.subject_id_column == "subject_id"


# ── MessyConfig parse-time validation (#73) ──────────────────────────


class TestParseValidation:
    def test_rejects_multi_column_subject_id(self):
        with pytest.raises(ValueError, match="must reference exactly one source column"):
            MessyConfig.parse(
                {"t": {"_defaults": {"subject_id": "$a + $b"}, "e": {"code": "X", "time": None}}}
            )

    def test_allows_hash_wrapped_subject_id(self):
        cfg = MessyConfig.parse(
            {"t": {"_defaults": {"subject_id": "hash($mrn)"}, "e": {"code": "X", "time": None}}}
        )
        assert cfg.tables[0].subject_id_column == "mrn"

    def test_rejects_per_event_subject_id(self):
        with pytest.raises(ValueError, match="subject_id is a table-level concept"):
            MessyConfig.parse({"t": {"e": {"code": "X", "time": None, "subject_id": "$sid"}}})

    def test_missing_code_raises_with_location(self):
        with pytest.raises(KeyError, match=r"Event 'labs\.lab' must contain a 'code' key"):
            MessyConfig.parse({"labs": {"lab": {"time": None}}})

    def test_missing_time_treated_as_static(self):
        """``time`` is optional — missing is equivalent to ``time: null`` (static event)."""
        cfg = MessyConfig.parse({"labs": {"lab": {"code": "X"}}})
        assert cfg.tables[0].events[0].time is None

    def test_non_string_event_field_raises(self):
        with pytest.raises(ValueError, match="must be a string"):
            MessyConfig.parse({"t": {"e": {"code": "X", "time": None, "val": 42}}})

    def test_global_defaults_merged(self):
        cfg = MessyConfig.parse(
            {
                "_defaults": {"subject_id": "$MRN"},
                "a": {"e": {"code": "X", "time": None}},
                "b": {"_defaults": {"subject_id": "$patient_id"}, "e": {"code": "X", "time": None}},
            }
        )
        assert cfg.tables[0].subject_id_expr == "$MRN"
        assert cfg.tables[1].subject_id_expr == "$patient_id"


# ── extract_event() library contract ─────────────────────────────────


def _ev(code: str, time: str | None = None, **extras) -> EventConfig:
    return EventConfig(name="e", table_prefix="t", code=code, time=time, extras=dict(extras), metadata={})


class TestExtractEvent:
    def test_static_event(self):
        raw = pl.DataFrame({"subject_id": [1, 2], "color": ["blue", "green"]})
        result = extract_event(raw, _ev("EYE_COLOR", None, color="$color"))
        assert result["time"].null_count() == 2
        assert result["color"].to_list() == ["blue", "green"]

    def test_null_code_rows_filtered(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "name": ["A", None, "C"],
                "time": ["2021-01-01", "2021-01-02", "2021-01-03"],
            }
        )
        result = extract_event(raw, _ev('f"{$name}"', '$time::"%Y-%m-%d"'))
        assert result.shape[0] == 2
        assert result["code"].to_list() == ["A", "C"]

    def test_null_time_rows_filtered(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "time": ["2021-01-01", None, "2021-01-03"],
            }
        )
        result = extract_event(raw, _ev("EVENT", '$time::"%Y-%m-%d"'))
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
        cfg = _ev("MEAS", '$time::"%Y-%m-%d"', numeric_value="$val", text_value="$text")
        result = extract_event(raw, cfg, do_dedup_text_and_numeric=True)
        assert result["text_value"][0] is None
        assert result["text_value"][1] == "other"

    def test_dedup_disabled_preserves_text(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1],
                "time": ["2021-01-01"],
                "val": [1.5],
                "text": ["1.5"],
            }
        )
        cfg = _ev("MEAS", '$time::"%Y-%m-%d"', numeric_value="$val", text_value="$text")
        result = extract_event(raw, cfg, do_dedup_text_and_numeric=False)
        assert result["text_value"][0] == "1.5"

    def test_duplicate_rows_removed(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1, 1],
                "time": ["2021-01-01", "2021-01-01"],
            }
        )
        result = extract_event(raw, _ev("EVENT", '$time::"%Y-%m-%d"'))
        assert result.shape[0] == 1

    def test_time_with_nulls_via_scan_parquet(self, tmp_path):
        """Regression: strptime on null-heavy time columns must not crash with scan_parquet."""
        raw = pl.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "dod": ["2018-11-01T00:00:00", None, None],
            }
        )
        fp = tmp_path / "test.parquet"
        raw.write_parquet(fp)

        lf = pl.scan_parquet(fp, glob=False)
        result = extract_event(lf, _ev("MEDS_DEATH", '$dod::"%Y-%m-%dT%H:%M:%S"')).collect()

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
        result = extract_event(raw, _ev("MEDS_BIRTH", "$year_of_birth::year"))
        assert len(result) == 2
        assert set(result["subject_id"].to_list()) == {1, 3}


# ── convert_to_events() library contract ─────────────────────────────


class TestConvertToEvents:
    def test_empty_iterable_raises(self):
        df = pl.DataFrame({"subject_id": [1]})
        with pytest.raises(ValueError, match="No event configurations"):
            convert_to_events(df, [])

    def test_invalid_event_raises(self):
        df = pl.DataFrame({"subject_id": [1]})
        bad = EventConfig(name="bad", table_prefix="t", code="$missing_col", time=None)
        with pytest.raises(ValueError, match="Error extracting event"):
            convert_to_events(df, [bad]).collect()

    def test_multiple_events_concatenated(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1, 2],
                "dept": ["CARDIAC", "PULM"],
                "ts": ["2021-01-01", "2021-01-02"],
                "color": ["blue", "green"],
            }
        )
        cfgs = [
            EventConfig(name="admit", table_prefix="t", code="ADMISSION", time='$ts::"%Y-%m-%d"'),
            EventConfig(
                name="color",
                table_prefix="t",
                code="EYE_COLOR",
                time=None,
                extras={"eye_color": "$color"},
            ),
        ]
        result = convert_to_events(raw, cfgs)
        assert result.shape[0] == 4
        codes = sorted(result["code"].to_list())
        assert codes == ["ADMISSION", "ADMISSION", "EYE_COLOR", "EYE_COLOR"]

    def test_diagonal_concat_fills_nulls(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1],
                "ts": ["2021-01-01"],
                "val": [1.5],
                "color": ["blue"],
            }
        )
        cfgs = [
            EventConfig(
                name="with_val",
                table_prefix="t",
                code="A",
                time='$ts::"%Y-%m-%d"',
                extras={"numeric_value": "$val"},
            ),
            EventConfig(
                name="with_color",
                table_prefix="t",
                code="B",
                time=None,
                extras={"eye_color": "$color"},
            ),
        ]
        result = convert_to_events(raw, cfgs)
        assert result.shape == (2, 6)  # subject_id, code, time, source_block, numeric_value, eye_color
        assert result.filter(pl.col("code") == "A")["eye_color"][0] is None
        assert result.filter(pl.col("code") == "B")["numeric_value"][0] is None
        assert set(result["source_block"].to_list()) == {"t/with_val", "t/with_color"}


# ── Integration: MessyConfig → extraction ────────────────────────────


class TestIntegration:
    def test_hash_subject_id_through_table(self):
        raw = pl.DataFrame(
            {
                "mrn": ["ABC", "DEF", "ABC"],
                "time": ["2021-01-01", "2021-01-02", "2021-01-03"],
            }
        )

        cfg = MessyConfig.parse(
            {
                "patients": {
                    "_defaults": {"subject_id": "hash($mrn)"},
                    "event": {"code": "X", "time": '$time::"%Y-%m-%d"'},
                }
            }
        )
        table = cfg.tables[0]
        df_with_sid = raw.with_columns(subject_id=table.subject_id_polars_expr)

        result = convert_to_events(df_with_sid, table.events)
        assert result.schema["subject_id"] == pl.Int64
        assert result["subject_id"][0] == result["subject_id"][2]

    def test_derived_column_via_table_cols(self):
        from dftly import Parser

        raw = pl.DataFrame(
            {
                "subject_id": [1, 1, 2],
                "val1": [10, 20, 30],
                "val2": [1, 2, 3],
                "time": ["2021-01-01", "2021-01-02", "2021-01-03"],
            }
        )
        cfg = MessyConfig.parse(
            {
                "t": {
                    "_table": {"cols": {"total": "$val1 + $val2"}},
                    "measurement": {
                        "code": "MEASUREMENT",
                        "time": '$time::"%Y-%m-%d"',
                        "numeric_value": "$total",
                    },
                }
            }
        )
        table = cfg.tables[0]
        df_with_derived = raw.with_columns(**Parser.to_polars(dict(table.cols)))

        result = convert_to_events(df_with_derived, table.events)
        assert result["numeric_value"].to_list() == [11, 22, 33]
