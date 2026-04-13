"""Library-behavior tests for MEDS_extract config parsing and extraction.

dftly DSL semantics (f-string interpolation, bare-literal parsing, ``::`` casts,
arithmetic) are tested in the dftly package itself — nothing in this file should
exercise them except incidentally.

Parser-validation errors (missing code, per-event subject_id, non-string values)
are demonstrated as doctests on :class:`EventConfig.parse` and :class:`TableConfig.parse`
in ``src/MEDS_extract/config.py``; this file keeps the runtime-extraction and
schema-dependent tests that would be awkward as doctests.
"""

import polars as pl
import pytest

from MEDS_extract.config import EventConfig, MessyConfig, TableConfig

_ = pl.Config.set_tbl_width_chars(600)


def _event(code: str, time: str | None = None, **extras) -> EventConfig:
    return EventConfig(name="e", columns={"code": code, "time": time, **extras})


# ── TableConfig.subject_id_polars_expr: always Int64 output ───────────


class TestSubjectIdInt64Output:
    """Whatever the input expression, the materialized ``subject_id`` must be Int64."""

    def test_int_column_ref(self):
        tc = TableConfig.parse(
            "t",
            {"_defaults": {"subject_id": "$patient_id"}, "e": {"code": "X", "time": None}},
        )
        df = pl.DataFrame({"patient_id": [1, 2]}, schema={"patient_id": pl.Int32})
        assert df.select(subject_id=tc.subject_id_polars_expr).schema["subject_id"] == pl.Int64

    def test_hash_expression(self):
        tc = TableConfig.parse(
            "t",
            {"_defaults": {"subject_id": "hash($mrn)"}, "e": {"code": "X", "time": None}},
        )
        df = pl.DataFrame({"mrn": ["ABC", "DEF", "ABC"]})
        result = df.select(subject_id=tc.subject_id_polars_expr)
        assert result.schema["subject_id"] == pl.Int64
        # Same input → same hash (the only invariant the Int64 conversion must preserve)
        assert result["subject_id"][0] == result["subject_id"][2]


# ── EventConfig.extract: library contract ─────────────────────────────


class TestEventExtract:
    def test_static_event_has_null_time(self):
        raw = pl.DataFrame({"subject_id": [1, 2], "color": ["blue", "green"]})
        ev = _event("EYE_COLOR", None, eye_color="$color")
        result = ev.extract(raw.lazy(), "patients/eye").collect()
        assert result["time"].null_count() == 2
        assert result["eye_color"].to_list() == ["blue", "green"]

    def test_null_code_rows_filtered(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "name": ["A", None, "C"],
                "time": ["2021-01-01", "2021-01-02", "2021-01-03"],
            }
        )
        result = _event('f"{$name}"', '$time::"%Y-%m-%d"').extract(raw.lazy(), "t/e").collect()
        assert result.shape[0] == 2
        assert result["code"].to_list() == ["A", "C"]

    def test_null_time_rows_filtered(self):
        raw = pl.DataFrame({"subject_id": [1, 2, 3], "time": ["2021-01-01", None, "2021-01-03"]})
        result = _event("EVENT", '$time::"%Y-%m-%d"').extract(raw.lazy(), "t/e").collect()
        assert result.shape[0] == 2

    def test_dedup_text_matches_numeric(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1, 2],
                "time": ["2021-01-01", "2021-01-02"],
                "val": [1.5, 2.0],
                "text": ["1.5", "other"],
            }
        )
        ev = _event("MEAS", '$time::"%Y-%m-%d"', numeric_value="$val", text_value="$text")
        result = ev.extract(raw.lazy(), "t/e", do_dedup_text_and_numeric=True).collect()
        assert result["text_value"][0] is None  # "1.5" == 1.5 → nulled
        assert result["text_value"][1] == "other"

    def test_duplicate_rows_removed(self):
        raw = pl.DataFrame({"subject_id": [1, 1], "time": ["2021-01-01", "2021-01-01"]})
        result = _event("EVENT", '$time::"%Y-%m-%d"').extract(raw.lazy(), "t/e").collect()
        assert result.shape[0] == 1

    def test_source_block_always_present(self):
        """``source_block`` is an unconditional output column (no legacy conditional path)."""
        raw = pl.DataFrame({"subject_id": [1], "time": ["2021-01-01"]})
        result = _event("X", '$time::"%Y-%m-%d"').extract(raw.lazy(), "patients/dob").collect()
        assert result["source_block"].to_list() == ["patients/dob"]

    def test_scan_parquet_null_time_regression(self, tmp_path):
        """Regression: strptime on null-heavy time columns must not crash with scan_parquet.

        Polars predicate pushdown can cause strptime(strict=True) to evaluate on empty
        strings during parquet scanning. The null filter uses source columns, not the
        parsed time expression, to sidestep that bug.
        """
        raw = pl.DataFrame({"subject_id": [1, 2, 3], "dod": ["2018-11-01T00:00:00", None, None]})
        fp = tmp_path / "test.parquet"
        raw.write_parquet(fp)

        lf = pl.scan_parquet(fp, glob=False)
        result = _event("MEDS_DEATH", '$dod::"%Y-%m-%dT%H:%M:%S"').extract(lf, "patients/dod").collect()
        assert len(result) == 1
        assert result["subject_id"][0] == 1

    def test_non_string_time_column_regression(self):
        """Regression: time null filter must not compare non-string columns with ``!= ""`` (#68)."""
        raw = pl.DataFrame({"subject_id": [1, 2, 3], "year_of_birth": [1980, None, 1990]})
        result = _event("MEDS_BIRTH", "$year_of_birth::year").extract(raw.lazy(), "patients/dob").collect()
        assert len(result) == 2
        assert set(result["subject_id"].to_list()) == {1, 3}


# ── TableConfig.extract_events ─────────────────────────────────────────


class TestTableExtract:
    def test_empty_events_raises(self):
        tc = TableConfig(input_prefix="empty")
        df = pl.DataFrame({"subject_id": [1]}).lazy()
        with pytest.raises(ValueError, match="has no events"):
            tc.extract_events(df)

    def test_multiple_events_concatenated(self):
        raw = pl.DataFrame(
            {
                "subject_id": [1, 2],
                "dept": ["CARDIAC", "PULM"],
                "ts": ["2021-01-01", "2021-01-02"],
                "color": ["blue", "green"],
            }
        )
        tc = TableConfig.parse(
            "data",
            {
                "admit": {"code": "ADMISSION", "time": '$ts::"%Y-%m-%d"'},
                "color": {"code": "EYE_COLOR", "time": None, "eye_color": "$color"},
            },
        )
        result = tc.extract_events(raw.lazy()).collect()
        assert result.shape[0] == 4
        assert sorted(result["code"].to_list()) == [
            "ADMISSION",
            "ADMISSION",
            "EYE_COLOR",
            "EYE_COLOR",
        ]
        assert set(result["source_block"].to_list()) == {"data/admit", "data/color"}

    def test_prepare_applies_subject_id_and_derived_columns(self):
        raw = pl.DataFrame(
            {
                "mrn": ["ABC", "DEF"],
                "val1": [10, 20],
                "val2": [1, 2],
                "time": ["2021-01-01", "2021-01-02"],
            }
        )
        tc = TableConfig.parse(
            "t",
            {
                "_defaults": {"subject_id": "hash($mrn)"},
                "_table": {"cols": {"total": "$val1 + $val2"}},
                "m": {"code": "M", "time": '$time::"%Y-%m-%d"', "numeric_value": "$total"},
            },
        )
        result = tc.extract_events(raw.lazy()).collect()
        assert result.schema["subject_id"] == pl.Int64
        assert sorted(result["numeric_value"].to_list()) == [11, 22]


# ── MessyConfig.load ─────────────────────────────────────────────────


class TestMessyConfigLoad:
    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Event conversion config file not found"):
            MessyConfig.load(tmp_path / "nope.yaml")

    def test_load_parses_yaml(self, tmp_path):
        fp = tmp_path / "cfg.yaml"
        fp.write_text("_defaults:\n  subject_id: $MRN\npatients:\n  dob:\n    code: BIRTH\n    time: null\n")
        cfg = MessyConfig.load(fp)
        assert cfg.table_prefixes == ["patients"]
        assert cfg.tables[0].subject_id_expr == "$MRN"
