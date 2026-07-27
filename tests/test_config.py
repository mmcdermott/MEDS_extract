"""Library-behavior tests for MEDS_extract config.

Happy-path extraction, parsing, validation, and source-column planning are demonstrated as
doctests on :class:`EventConfig`, :class:`TableConfig`, and :class:`MessyConfig` in
``src/MEDS_extract/config.py`` — this file keeps only regression tests that are awkward to
express as doctests: polars scan/dtype/plan-shape edge cases (null-heavy time columns,
typed non-string time columns, cross-column time coalesces, predicate-pushdown safety) and
log-capture assertions (drop-count warnings, ``sources:`` credential redaction).
"""

import logging

import polars as pl
import pytest

from MEDS_extract.config import EventConfig, MessyConfig

_ = pl.Config.set_tbl_width_chars(600)


def _event(code: str, time: str | None = None, **extras) -> EventConfig:
    return EventConfig.parse("e", {"code": code, "time": time, **extras})


# ── EventConfig.extract regressions (tied to polars scan_parquet / type quirks) ───────────────────


def test_scan_parquet_null_time_regression(tmp_path):
    """Regression: strptime on null-heavy time columns must not crash with scan_parquet.

    A strptime(strict=True) predicate pushed into the parquet scan evaluates on
    validity-unmasked data, where null slots surface as "" and panic the process
    (pola-rs/polars#28521). The null filter runs on the computed ``time`` column after
    the select, which polars keeps above the scan, sidestepping that bug.
    """
    raw = pl.DataFrame({"subject_id": [1, 2, 3], "dod": ["2018-11-01T00:00:00", None, None]})
    fp = tmp_path / "test.parquet"
    raw.write_parquet(fp)

    lf = pl.scan_parquet(fp, glob=False)
    result = _event("MEDS_DEATH", '$dod::"%Y-%m-%dT%H:%M:%S"').extract(lf, "patients/dod").collect()
    assert len(result) == 1
    assert result["subject_id"][0] == 1


def test_non_string_time_column_regression():
    """Regression: time null filter must not compare non-string columns with ``!= ""`` (#68)."""
    raw = pl.DataFrame({"subject_id": [1, 2, 3], "year_of_birth": [1980, None, 1990]})
    result = _event("MEDS_BIRTH", "$year_of_birth::year").extract(raw.lazy(), "patients/dob").collect()
    assert len(result) == 2
    assert set(result["subject_id"].to_list()) == {1, 3}


def test_coalesce_across_columns_time_keeps_partial_rows(tmp_path):
    """Regression: a time coalescing across columns must keep rows where only one column is set (#149).

    The old raw-column null pre-filter ANDed ``is_not_null`` over ALL time source columns,
    silently dropping every row where either column was null — on the MIMIC-IV demo this
    lost 16 of 31 deaths (all out-of-hospital deaths, with ``dod`` but no admissions
    ``deathtime``). The filter now runs on the computed ``time`` column, so coalesce
    semantics apply. Verified both on an in-memory frame and through ``scan_parquet`` (the
    path where a fallible predicate pushed into the scan would panic).
    """
    raw = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "deathtime": ["2021-01-01 12:00:00", None, None],
            "dod": ["2021-01-01", "2021-01-02", None],
        }
    )
    ev = _event("MEDS_DEATH", 'coalesce($deathtime::?"%Y-%m-%d %H:%M:%S", $dod::?"%Y-%m-%d")')

    result = ev.extract(raw.lazy(), "patients/death").collect()
    assert set(result["subject_id"].to_list()) == {1, 2}

    fp = tmp_path / "patients.parquet"
    raw.write_parquet(fp)
    result = ev.extract(pl.scan_parquet(fp, glob=False), "patients/death").collect()
    assert set(result["subject_id"].to_list()) == {1, 2}


def test_coalesce_nested_inside_strptime_time_keeps_partial_rows():
    """A coalesce nested inside another node (``coalesce($a, $b)::?fmt``) also keeps partial rows (#149)."""
    raw = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "a": ["2021-01-01", None, None],
            "b": ["2021-06-01", "2021-01-02", None],
        }
    )
    ev = _event("MEDS_DEATH", 'coalesce($a, $b)::?"%Y-%m-%d"')
    result = ev.extract(raw.lazy(), "patients/death").collect()
    assert set(result["subject_id"].to_list()) == {1, 2}


def test_strict_cast_empty_string_time_errors(tmp_path):
    """A strict (``::"fmt"``) time cast over an unparsable ``""`` raises a clean error naming the value.

    Previously such rows were silently pre-filtered away by the raw-column ``!= ""`` guard.
    That guard is gone: strict means strict, and the failure is a normal
    ``InvalidOperationError`` (not a pushdown panic), including through ``scan_parquet``.
    Authors who want unparsable values dropped opt in with a lenient ``::?`` cast.
    """
    raw = pl.DataFrame({"subject_id": [1, 2], "dod": ["2018-11-01T00:00:00", ""]})
    fp = tmp_path / "patients.parquet"
    raw.write_parquet(fp)

    ev = _event("MEDS_DEATH", '$dod::"%Y-%m-%dT%H:%M:%S"')
    with pytest.raises(pl.exceptions.InvalidOperationError, match=r'\[""\]'):
        ev.extract(pl.scan_parquet(fp, glob=False), "patients/death").collect()


def test_lenient_unparsable_time_row_dropped():
    """Regression: a lenient (``::?``) cast of an unparsable value must DROP the row.

    On dev before this fix, junk rows slipped past the raw-column pre-filter (non-null,
    non-empty) and leaked downstream with ``time=null`` — invalid MEDS output. The computed
    ``time`` filter now drops them.
    """
    raw = pl.DataFrame({"subject_id": [1, 2, 3], "dod": ["2018-11-01T00:00:00", "not a date", None]})
    ev = _event("MEDS_DEATH", '$dod::?"%Y-%m-%dT%H:%M:%S"')
    result = ev.extract(raw.lazy(), "patients/death").collect()
    assert set(result["subject_id"].to_list()) == {1}
    assert result["time"].null_count() == 0


def test_time_filter_not_pushed_into_scan(tmp_path):
    """Plan shape: the computed time/code null filters must stay above the SELECT, not enter the scan.

    Polars evaluates predicates pushed into ``scan_parquet`` on validity-unmasked data
    (null slots of a String column surface as ``""``), so a fallible predicate there
    panics on polars >= 1.28 (pola-rs/polars#28521). A pushed predicate shows up as a
    ``SELECTION`` line inside the scan node of ``explain()``; assert it never appears and
    that the filter sits above the select while column projection is still pushed down.
    """
    raw = pl.DataFrame({"subject_id": [1, 2], "dod": ["2018-11-01T00:00:00", None], "junk": ["x", "y"]})
    fp = tmp_path / "patients.parquet"
    raw.write_parquet(fp)

    ev = _event("MEDS_DEATH", '$dod::"%Y-%m-%dT%H:%M:%S"')
    plan = ev.extract(pl.scan_parquet(fp, glob=False), "patients/death").explain()

    assert "SELECTION" not in plan  # no predicate inside the scan node
    assert 'FILTER col("time").is_not_null()' in plan
    assert plan.index('FILTER col("time")') < plan.index("Parquet SCAN")
    assert "PROJECT 2/3 COLUMNS" in plan  # projection pushdown intact (junk not read)


def test_extract_drop_accounting_warning(caplog):
    """Dropped-row counts surface as one WARNING per event; zero-drop events stay silent (#26)."""
    ev = _event("$name", '$ts::?"%Y-%m-%d"')

    lossy = pl.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "name": ["A", "B", None, "D"],
            "ts": ["2021-01-01", "junk", "2021-01-03", None],
        }
    )
    with caplog.at_level(logging.WARNING, logger="MEDS_extract.config"):
        result = ev.extract(lossy.lazy(), "patients/e").collect()
    assert set(result["subject_id"].to_list()) == {1}
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == [
        "`patients/e`: dropped 2/4 rows with null time "
        "(unparsable or missing under the configured formats) and 1 with null code"
    ]

    caplog.clear()
    clean = pl.DataFrame({"subject_id": [1], "name": ["A"], "ts": ["2021-01-01"]})
    with caplog.at_level(logging.WARNING, logger="MEDS_extract.config"):
        result = ev.extract(clean.lazy(), "patients/e").collect()
    assert len(result) == 1
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


# ── Combined-MESSY ``sources:`` hygiene: credential redaction in logs ─────────────────────────────


def test_messy_config_load_does_not_log_sources_block(tmp_path, caplog):
    """The ``sources:`` block can carry credentials (literal API keys, passwords); ``MessyConfig.load``'s INFO
    dump of the config must strip it so secrets never land in per-stage logs."""
    import logging

    cfg_fp = tmp_path / "messy.yaml"
    cfg_fp.write_text(
        """
sources:
  dataset:
    - type: http
      headers: {X-Dataverse-key: super-secret-token}
      urls: [https://example.com/x.csv]
patients:
  dob: {code: BIRTH, time: null}
"""
    )
    with caplog.at_level(logging.INFO, logger="MEDS_extract.config"):
        cfg = MessyConfig.load(cfg_fp)

    assert cfg.table_prefixes == ["patients"]
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "patients" in logged  # the event-conversion side is still logged
    assert "super-secret-token" not in logged
    assert "sources" not in logged


# ── Aggregated joins: String-dtype hazards surfaced at apply time ─────────────────────────────────


def test_aggregated_join_string_dtype_checks(tmp_path, caplog):
    """``min``/``max`` on a String column warns (lexicographic comparison is only right for ISO-style
    timestamp text); ``sum``/``mean`` on a String column raises (polars would fail cryptically for ``sum`` and
    silently return all-null for ``mean``); typed columns stay silent."""
    from datetime import datetime

    from MEDS_extract.config import JoinConfig

    pl.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "deathtime_str": ["2020-03-05", "2020-03-01", None],
            "deathtime_dt": [datetime(2020, 3, 5), datetime(2020, 3, 1), None],
        }
    ).write_parquet(tmp_path / "admissions.parquet")
    left = pl.LazyFrame({"subject_id": [1, 2, 3]})

    def _apply(cols: dict) -> pl.DataFrame:
        jc = JoinConfig.parse({"admissions": {"key": "subject_id", "cols": cols}})
        caplog.clear()  # discard the construction-time use-with-care warning; this test is apply-time only
        return jc.apply(left, tmp_path).collect()

    # String min: warns but still computes (values ARE correct for ISO-ordered text).
    with caplog.at_level(logging.WARNING, logger="MEDS_extract.config"):
        out = _apply({"deathtime_str": "min"})
    assert out.sort("subject_id")["deathtime_str"].to_list() == ["2020-03-01", None, None]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "lexicographically" in warnings[0]

    # Datetime-typed min: no warning.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="MEDS_extract.config"):
        out = _apply({"deathtime_dt": "min"})
    assert out.sort("subject_id")["deathtime_dt"].to_list() == [datetime(2020, 3, 1), None, None]
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    # sum/mean on String: rejected with the join table + column named.
    for agg in ("sum", "mean"):
        with pytest.raises(ValueError, match=rf"'admissions'.*{agg}.*'deathtime_str'"):
            _apply({"deathtime_str": agg})


def test_aggregated_join_construction_warning(caplog):
    """Constructing an aggregated JoinConfig logs exactly one use-with-care WARNING naming the join and its
    col→agg pairs (aggregation silently absorbs data conflicts and breaks row-level provenance); flat joins
    stay silent."""
    from MEDS_extract.config import JoinConfig

    with caplog.at_level(logging.WARNING, logger="MEDS_extract.config"):
        JoinConfig.parse(
            {
                "hosp/admissions": {
                    "key": "subject_id",
                    "cols": {"deathtime": "min", "admittime": "max"},
                }
            }
        )
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "'hosp/admissions'" in warnings[0]
    assert "min(deathtime), max(admittime)" in warnings[0]
    assert "provenance" in warnings[0]
    assert "Use with care" in warnings[0]

    # Flat (non-aggregated) joins construct silently.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="MEDS_extract.config"):
        JoinConfig.parse({"stays": {"key": "stay_id", "cols": ["patient_id", "dischtime"]}})
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
