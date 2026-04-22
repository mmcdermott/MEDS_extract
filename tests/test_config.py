"""Library-behavior tests for MEDS_extract config.

Happy-path extraction, parsing, and validation are demonstrated as doctests on
:class:`EventConfig`, :class:`TableConfig`, and :class:`MessyConfig` in
``src/MEDS_extract/config.py`` — this file keeps only regression tests for
schema-dependent edge cases that are awkward to express as doctests (e.g.,
scanning parquet with null-heavy columns, typed non-string time columns).
"""

import polars as pl

from MEDS_extract.config import EventConfig

_ = pl.Config.set_tbl_width_chars(600)


def _event(code: str, time: str | None = None, **extras) -> EventConfig:
    return EventConfig.parse("e", {"code": code, "time": time, **extras})


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
