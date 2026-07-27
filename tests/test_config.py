"""Library-behavior tests for MEDS_extract config.

Happy-path extraction, parsing, validation, and source-column planning are demonstrated as
doctests on :class:`EventConfig`, :class:`TableConfig`, and :class:`MessyConfig` in
``src/MEDS_extract/config.py`` — this file keeps only regression tests that are awkward to
express as doctests: polars scan/dtype edge cases (null-heavy time columns, typed
non-string time columns) and log-capture assertions (``sources:`` credential redaction).
"""

import polars as pl

from MEDS_extract.config import EventConfig, MessyConfig

_ = pl.Config.set_tbl_width_chars(600)


def _event(code: str, time: str | None = None, **extras) -> EventConfig:
    return EventConfig.parse("e", {"code": code, "time": time, **extras})


# ── EventConfig.extract regressions (tied to polars scan_parquet / type quirks) ───────────────────


def test_scan_parquet_null_time_regression(tmp_path):
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


def test_non_string_time_column_regression():
    """Regression: time null filter must not compare non-string columns with ``!= ""`` (#68)."""
    raw = pl.DataFrame({"subject_id": [1, 2, 3], "year_of_birth": [1980, None, 1990]})
    result = _event("MEDS_BIRTH", "$year_of_birth::year").extract(raw.lazy(), "patients/dob").collect()
    assert len(result) == 2
    assert set(result["subject_id"].to_list()) == {1, 3}


def test_coalesce_across_columns_time_keeps_partial_rows(tmp_path):
    """Regression: a time coalescing across columns must keep rows where only one column is set (#149).

    The null pre-filter used to AND ``is_not_null`` over ALL time source columns, silently
    dropping every row where either column was null — on the MIMIC-IV demo this lost 16 of
    31 deaths (all out-of-hospital deaths, with ``dod`` but no admissions ``deathtime``).
    The filter is now derived from the time expression's structure: an OR across coalesce
    branches. Verified both on an in-memory frame and through ``scan_parquet`` (the
    pre-filter exists to keep strict strptime off unfiltered scan batches, so the parquet
    path is the one that must stay safe).
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


def test_coalesce_same_column_multi_format_time_unchanged():
    """A same-column multi-format coalesce keeps rows parseable by either format, drops null/unparsable."""
    raw = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "t": ["2021-01-01 12:00:00", "2021-01-02", None],
        }
    )
    ev = _event("MEDS_DEATH", 'coalesce($t::?"%Y-%m-%d %H:%M:%S", $t::?"%Y-%m-%d")')
    result = ev.extract(raw.lazy(), "patients/death").collect()
    assert set(result["subject_id"].to_list()) == {1, 2}


def test_coalesce_nested_inside_strptime_time_keeps_partial_rows():
    """A coalesce nested inside another node (``coalesce($a, $b)::?fmt``) also ORs its branches (#149)."""
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
