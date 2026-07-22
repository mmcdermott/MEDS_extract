"""Library-behavior tests for MEDS_extract config.

Happy-path extraction, parsing, and validation are demonstrated as doctests on
:class:`EventConfig`, :class:`TableConfig`, and :class:`MessyConfig` in
``src/MEDS_extract/config.py`` — this file keeps only regression tests for
schema-dependent edge cases that are awkward to express as doctests (e.g.,
scanning parquet with null-heavy columns, typed non-string time columns).
"""

import polars as pl
from omegaconf import OmegaConf
from yaml import load as load_yaml

from MEDS_extract.config import EventConfig, MessyConfig

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

_ = pl.Config.set_tbl_width_chars(600)


def _event(code: str, time: str | None = None, **extras) -> EventConfig:
    return EventConfig.parse("e", {"code": code, "time": time, **extras})


# ── MessyConfig.needed_source_columns edge cases ───────────────────────────────────────────


def test_needed_source_columns_join():
    cfg_yaml = """\
vitals:
  _table:
    join:
      stays:
        key: stay_id
        cols: [subject_id]
  HR:
    code: HR
    time: '$charttime::"%m/%d/%Y %H:%M:%S"'
    numeric_value: $HR
"""
    cfg = OmegaConf.create(load_yaml(cfg_yaml, Loader=Loader))
    cols = MessyConfig.parse(cfg).needed_source_columns()
    assert set(cols["vitals"]) == {"HR", "charttime", "stay_id"}
    assert set(cols["stays"]) == {"stay_id", "subject_id"}


def test_needed_source_columns_with_transforms():
    """Tests needed_source_columns handles transforms and event field values."""
    cfg_yaml = """\
data:
  _table:
    cols:
      derived: "$a + $b"
  event:
    code: $code_col
    time: null
    numeric_value: $val
    _metadata:
      meta_file:
        description: desc_col
"""
    cfg = OmegaConf.create(load_yaml(cfg_yaml, Loader=Loader))
    cols = MessyConfig.parse(cfg).needed_source_columns()
    # Should extract columns from _table.cols (a, b) and event fields (code_col, val)
    # but skip null time and _metadata.
    assert "a" in cols["data"]
    assert "b" in cols["data"]
    assert "code_col" in cols["data"]
    assert "val" in cols["data"]


def test_needed_source_columns_excludes_transform_outputs():
    """Regression: transform output columns must not appear in the source file extraction plan (#67)."""
    cfg_yaml = """\
hosp/patients:
  _table:
    cols:
      year_of_birth: "$anchor_year - $anchor_age"
  dob:
    code: MEDS_BIRTH
    time: '$year_of_birth::year'
"""
    cfg = OmegaConf.create(load_yaml(cfg_yaml, Loader=Loader))
    cols = MessyConfig.parse(cfg).needed_source_columns()
    assert "year_of_birth" not in cols["hosp/patients"], (
        f"Transform output 'year_of_birth' should not be in extraction plan: {cols['hosp/patients']}"
    )
    assert "anchor_year" in cols["hosp/patients"]
    assert "anchor_age" in cols["hosp/patients"]


def test_needed_source_columns_excludes_joined_columns_referenced_in_events():
    """Regression: joined columns must not be re-added to left-side extraction plan (#66)."""
    cfg_yaml = """\
hosp/drgcodes:
  _table:
    join:
      hosp/admissions:
        key: hadm_id
        cols: [dischtime]
  drg:
    code: 'f"DRG//{$drg_type}//{$drg_code}"'
    time: '$dischtime::"%Y-%m-%d %H:%M:%S"'
"""
    cfg = OmegaConf.create(load_yaml(cfg_yaml, Loader=Loader))
    cols = MessyConfig.parse(cfg).needed_source_columns()
    assert "dischtime" not in cols["hosp/drgcodes"], (
        f"Joined column 'dischtime' should not be in left-side extraction plan: {cols['hosp/drgcodes']}"
    )
    assert "dischtime" in cols["hosp/admissions"]
    assert "drg_type" in cols["hosp/drgcodes"]
    assert "drg_code" in cols["hosp/drgcodes"]
    assert "hadm_id" in cols["hosp/drgcodes"]


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


def test_messy_config_strips_sources_before_resolve():
    """The combined-MESSY pattern requires ``MessyConfig.parse`` to ignore ``sources:`` BEFORE
    ``OmegaConf.to_container(resolve=True)`` — otherwise a download-only ``${oc.env:...}`` inside a
    ``sources:`` block would fail when the pipeline loads the event-conversion side of the same file without
    that env var set.

    Lock the behavior explicitly with an unset env-var interpolation that would raise
    if resolve ran over the ``sources:`` subtree.
    """
    cfg = OmegaConf.create(
        {
            "sources": {
                "dataset": [{"type": "fsspec", "root": "${oc.env:UNSET_DOWNLOAD_ROOT}"}],
            },
            "_defaults": {"subject_id": "$patient_id"},
            "patients": {"dob": {"code": "DOB", "time": "$dob"}},
        }
    )
    parsed = MessyConfig.parse(cfg)
    # Only ``patients`` becomes a TableConfig; ``sources`` is silently skipped.
    assert [t.input_prefix for t in parsed.tables] == ["patients"]


def test_messy_config_load_does_not_log_sources_block(tmp_path, caplog):
    """The ``sources:`` block can carry credentials (literal API keys, passwords);
    ``MessyConfig.load``'s INFO dump of the config must strip it so secrets never
    land in per-stage logs."""
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
