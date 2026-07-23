"""Stage-level tests for ``extract_code_metadata`` — end-to-end runs of the real stage.

Covers behavior that needs the full mapper/reducer machinery rather than a single
helper (those are doctested in place): here, codes built from a source column literally
named ``code`` — the idiomatic ICD/OMOP vocabulary-table shape — must flow through the
``code_components`` map build, full-match extraction, and reduction without colliding
with the output ``code`` column (regression for
https://github.com/mmcdermott/MEDS_extract/issues/110; the struct-shape provenance for
that scenario is a doctest on ``EventConfig.extract``).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import polars as pl
from omegaconf import OmegaConf

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
            "source_block": ["diagnoses/dx", "diagnoses/dx"],
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
    # A non-conflicting column name: how overlapping metadata columns merge with pre-existing
    # metadata is orthogonal to #110 (overlap coalescing is covered by the #137 reducer tests).
    pl.DataFrame({"code": ["EXISTING"], "old_description": ["pre-existing code"]}).write_parquet(
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


def test_extract_code_metadata_handles_code_named_source_column():
    """The stage runs to completion when a ``code`` expression references a ``code`` column.

    Before the #110 fix this raised ``DuplicateError`` at the ``code_components`` unnest
    (this test carried a strict xfail marker). Now it asserts the stage completes and the
    full-match metadata lands on the reconstructed codes.
    """
    with tempfile.TemporaryDirectory() as d:
        out_dir = _run_extract_code_metadata(Path(d))
        codes_df = pl.read_parquet(out_dir / "codes.parquet")
        by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
        assert by_code.get("ICD//250.00") == "Diabetes mellitus"
        assert by_code.get("ICD//401.9") == "Essential hypertension"
