"""Demonstration / regression test for issue #110 — ``extract_code_metadata`` DuplicateError.

https://github.com/mmcdermott/MEDS_extract/issues/110

When a ``code`` expression references a source column literally named ``code`` (idiomatic
for ICD/OMOP vocabulary tables, e.g. ``code: f"ICD//{$code}"``), the ``code_components``
struct that ``convert_to_MEDS_events`` attaches has a field named ``code``.
``extract_code_metadata.main`` then runs (unconditionally, whenever any ``_metadata`` block
exists)::

    code_component_map = (
        all_data.select("code", "code_components").unique().collect().unnest("code_components")
    )

and the unnested ``code`` field collides with the existing output ``code`` column, raising
``polars.exceptions.DuplicateError`` and aborting the whole stage.

The first test establishes provenance (the struct really does carry a ``code`` field); the
second drives the *real* stage and is ``xfail(strict, raises=DuplicateError)`` — it fails
today via that exact exception, any *other* failure is a hard error rather than a silent
xfail, and the fix turns it green (``strict=True`` then flags the stale marker).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import polars as pl
import pytest
from omegaconf import OmegaConf

from MEDS_extract.config import EventConfig

# A diagnoses table whose code is built from a source column named ``code``, plus a
# ``_metadata`` block so the stage runs past its "no metadata -> exit" early return.
_MESSY = """\
diagnoses:
  dx:
    code: 'f"ICD//{$code}"'
    _metadata:
      icd_descriptions:
        description: long_title
"""


def _make_cfg(overrides: dict) -> OmegaConf:
    """Minimal DictConfig mimicking what MEDS-Transforms hands a stage (worker 0)."""
    base = {
        "do_overwrite": True,
        "seed": 1,
        "worker": 0,
        "polling_time": 0.1,
        "stage": "test",
        "stage_cfg": {},
        "etl_metadata": {
            "dataset_name": "TEST",
            "dataset_version": "1.0",
            "package_name": "MEDS_extract",
            "package_version": "0.0.0",
        },
    }
    base.update(overrides)
    cfg = OmegaConf.create(base)
    OmegaConf.set_struct(cfg, False)
    return cfg


def _run_extract_code_metadata(root: Path) -> Path:
    """Lay out a minimal extract_code_metadata invocation and run it; return the reducer dir.

    The events parquet carries a ``code_components`` struct with a field named ``code`` —
    exactly the shape ``convert_to_MEDS_events`` emits for ``code: f"ICD//{$code}"`` (proven
    by ``test_convert_produces_code_components_with_a_code_field``).
    """
    from MEDS_extract.extract_code_metadata.extract_code_metadata import main as ecm_stage

    events_dir = root / "events" / "train" / "0"
    events_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "subject_id": [1, 2],
            "time": [None, None],
            "code": ["ICD//250.00", "ICD//401.9"],
            "code_components": [{"code": "250.00"}, {"code": "401.9"}],
            "numeric_value": [None, None],
        }
    ).cast({"subject_id": pl.Int64, "time": pl.Datetime("us"), "numeric_value": pl.Float32}).write_parquet(
        events_dir / "data.parquet"
    )

    raw_dir = root / "raw"
    raw_dir.mkdir()
    (raw_dir / "icd_descriptions.csv").write_text(
        "code,long_title\n250.00,Diabetes mellitus\n401.9,Essential hypertension\n"
    )

    metadata_in = root / "metadata_in" / "metadata"
    metadata_in.mkdir(parents=True)
    pl.DataFrame({"code": ["EXISTING"], "description": ["pre-existing code"]}).write_parquet(
        metadata_in / "codes.parquet", use_pyarrow=True
    )

    event_cfg_fp = root / "messy.yaml"
    event_cfg_fp.write_text(_MESSY)
    shards_fp = root / "metadata" / ".shards.json"
    shards_fp.parent.mkdir(parents=True)
    shards_fp.write_text(json.dumps({"train/0": [1, 2]}))

    out_dir = root / "metadata_out" / "metadata"
    out_dir.mkdir(parents=True)

    cfg = _make_cfg(
        {
            "input_dir": str(raw_dir),
            "stage_cfg": {
                "data_input_dir": str(root / "events"),
                "output_dir": str(out_dir),
                "metadata_input_dir": str(metadata_in),
                "reducer_output_dir": str(out_dir),
                "description_separator": "\n",
            },
            "event_conversion_config_fp": str(event_cfg_fp),
            "shards_map_fp": str(shards_fp),
        }
    )
    ecm_stage.main_fn(cfg)
    return out_dir


def test_convert_produces_code_components_with_a_code_field():
    """Provenance: the real convert stage attaches a ``code_components`` struct whose field
    is literally ``code`` when the code references a ``$code`` source column — this is what
    makes the ``extract_code_metadata`` unnest collide.
    """
    raw = pl.DataFrame({"subject_id": [1], "code": ["250.00"], "ts": ["2020-01-01"]})
    out = (
        EventConfig.parse("dx", {"code": 'f"ICD//{$code}"', "time": '$ts::"%Y-%m-%d"'})
        .extract(raw.lazy(), "diagnoses/dx")
        .collect()
    )
    assert "code_components" in out.columns
    assert "code" in [f.name for f in out.schema["code_components"].fields]


@pytest.mark.xfail(
    strict=True,
    raises=pl.exceptions.DuplicateError,
    reason="#110: extract_code_metadata unnests `code_components` (which has a `code` field) "
    "alongside the existing `code` column, colliding -> DuplicateError aborts the stage.",
)
def test_extract_code_metadata_handles_code_named_source_column():
    """The stage should run to completion when a ``code`` references a ``code`` column.

    Today it raises ``DuplicateError`` at the ``code_components`` unnest. Scoping the marker
    to ``raises=DuplicateError`` means a regression elsewhere surfaces as a real failure, and
    the fix flips this to a passing regression test.
    """
    with tempfile.TemporaryDirectory() as d:
        out_dir = _run_extract_code_metadata(Path(d))
        assert (out_dir / "codes.parquet").exists()  # reached only once the bug is fixed
