"""In-process tests for stage edge cases not covered by the subprocess-based stage tests.

With coverage.py's subprocess patch enabled, the subprocess tests cover the main code paths.
These tests target specific edge cases in individual stage ``main_fn``s: subject_id_expr,
transforms, and new-style config syntax in ``convert_to_MEDS_events``; file skipping in
``shard_events``; and output-dir validation plus overwrite handling in
``finalize_MEDS_metadata``. Stage-level ``extract_code_metadata`` scenarios live in
``tests/test_extract_code_metadata.py``; single-function behavior is doctested on the
functions themselves.
"""

import json
import tempfile
from pathlib import Path

import polars as pl
import pytest
from omegaconf import OmegaConf

_ = pl.Config.set_tbl_width_chars(600)


def _make_cfg(overrides: dict) -> OmegaConf:
    """Build a minimal DictConfig mimicking what MEDS-Transforms provides to stages."""
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


# ── convert_to_MEDS_events: subject_id_expr, transforms, new-style config syntax ──


def test_convert_to_MEDS_events_subject_id_expr():
    """A ``subject_id`` expression (``hash($MRN)``) produces distinct Int64 subject IDs."""
    from MEDS_extract.convert_to_MEDS_events.convert_to_MEDS_events import main as cme_stage

    event_cfg = """\
subjects:
  _defaults:
    subject_id: "hash($MRN)"
  eye_color:
    code: 'f"EYE_COLOR//{$eye_color}"'
    time: null
"""

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        shard_dir = root / "input" / "train" / "0"
        shard_dir.mkdir(parents=True)
        pl.DataFrame({"MRN": ["ABC", "DEF"], "eye_color": ["BROWN", "BLUE"]}).write_parquet(
            shard_dir / "subjects.parquet"
        )

        event_cfg_fp = root / "event_cfgs.yaml"
        event_cfg_fp.write_text(event_cfg)
        shards_fp = root / ".shards.json"
        shards_fp.write_text(json.dumps({"train/0": [1]}))

        cfg = _make_cfg(
            {
                "stage_cfg": {
                    "data_input_dir": str(root / "input"),
                    "output_dir": str(root / "output"),
                    "do_dedup_text_and_numeric": False,
                },
                "event_conversion_config_fp": str(event_cfg_fp),
                "shards_map_fp": str(shards_fp),
            }
        )
        cme_stage.main_fn(cfg)

        df = pl.read_parquet(root / "output" / "train" / "0" / "subjects.parquet")
        assert df["subject_id"].dtype == pl.Int64
        # Two source rows -> two distinct, non-null hashed subject IDs.
        assert df["subject_id"].null_count() == 0
        assert df["subject_id"].n_unique() == 2
        assert set(df["code"].to_list()) == {"EYE_COLOR//BROWN", "EYE_COLOR//BLUE"}


def test_convert_to_MEDS_events_with_transforms():
    """``_table.cols`` transform outputs are computed and usable as event fields."""
    from MEDS_extract.convert_to_MEDS_events.convert_to_MEDS_events import main as cme_stage

    event_cfg = """\
data:
  _table:
    cols:
      doubled: "$value * 2"
  measurement:
    code: MEAS
    time: null
    numeric_value: "$doubled"
"""

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        shard_dir = root / "input" / "train" / "0"
        shard_dir.mkdir(parents=True)
        pl.DataFrame({"subject_id": [1, 2], "value": [10.0, 20.0]}).write_parquet(shard_dir / "data.parquet")

        event_cfg_fp = root / "event_cfgs.yaml"
        event_cfg_fp.write_text(event_cfg)
        shards_fp = root / ".shards.json"
        shards_fp.write_text(json.dumps({"train/0": [1, 2]}))

        cfg = _make_cfg(
            {
                "stage_cfg": {
                    "data_input_dir": str(root / "input"),
                    "output_dir": str(root / "output"),
                    "do_dedup_text_and_numeric": False,
                },
                "event_conversion_config_fp": str(event_cfg_fp),
                "shards_map_fp": str(shards_fp),
            }
        )
        cme_stage.main_fn(cfg)

        df = pl.read_parquet(root / "output" / "train" / "0" / "data.parquet")
        vals = sorted(df["numeric_value"].drop_nulls().to_list())
        assert vals == [20.0, 40.0]


def test_convert_to_MEDS_events_new_style_config():
    """Top-level ``_defaults`` and ``_table`` config syntax both take effect in one run."""
    from MEDS_extract.convert_to_MEDS_events.convert_to_MEDS_events import main as cme_stage

    event_cfg = """\
_defaults:
  subject_id: "hash($MRN)"
data:
  _table:
    cols:
      doubled: "$value * 2"
  measurement:
    code: MEAS
    time: null
    numeric_value: "$doubled"
"""

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        shard_dir = root / "input" / "train" / "0"
        shard_dir.mkdir(parents=True)
        pl.DataFrame({"MRN": ["ABC", "DEF"], "value": [10.0, 20.0]}).write_parquet(shard_dir / "data.parquet")

        event_cfg_fp = root / "event_cfgs.yaml"
        event_cfg_fp.write_text(event_cfg)
        shards_fp = root / ".shards.json"
        shards_fp.write_text(json.dumps({"train/0": [1, 2]}))

        cfg = _make_cfg(
            {
                "stage_cfg": {
                    "data_input_dir": str(root / "input"),
                    "output_dir": str(root / "output"),
                    "do_dedup_text_and_numeric": False,
                },
                "event_conversion_config_fp": str(event_cfg_fp),
                "shards_map_fp": str(shards_fp),
            }
        )
        cme_stage.main_fn(cfg)

        df = pl.read_parquet(root / "output" / "train" / "0" / "data.parquet")
        # _defaults.subject_id should produce distinct, non-null hashed Int64 subject IDs.
        assert df["subject_id"].dtype == pl.Int64
        assert df["subject_id"].null_count() == 0
        assert df["subject_id"].n_unique() == 2
        # _table.cols.doubled should be computed before event extraction.
        vals = sorted(df["numeric_value"].drop_nulls().to_list())
        assert vals == [20.0, 40.0]


# ── shard_events: skip files absent from the event config ──


def test_shard_events_skips_unconfigured_files():
    """Files in the raw input that no event config references are not sharded."""
    from MEDS_extract.shard_events.shard_events import main as shard_stage

    minimal_cfg = """\
data:
  event:
    code: X
    time: null
"""

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        raw_dir = root / "raw_cohort"
        raw_dir.mkdir()
        pl.DataFrame({"subject_id": [1]}).write_parquet(raw_dir / "data.parquet")
        pl.DataFrame({"a": [1]}).write_parquet(raw_dir / "extra.parquet")

        event_cfg_fp = root / "event_cfgs.yaml"
        event_cfg_fp.write_text(minimal_cfg)

        cfg = _make_cfg(
            {
                "stage": "shard_events",
                "stage_cfg": {
                    "data_input_dir": str(raw_dir / "data"),
                    "output_dir": str(root / "output" / "data"),
                    "row_chunksize": 100,
                    "infer_schema_length": 10000,
                },
                "event_conversion_config_fp": str(event_cfg_fp),
            }
        )
        shard_stage.main_fn(cfg)

        assert (root / "output" / "data" / "data").exists()
        assert not (root / "output" / "data" / "extra").exists()


# ── finalize_MEDS_metadata: output-dir validation and overwrite handling ──


def test_finalize_MEDS_metadata_output_dir_validation():
    """The stage rejects a reducer output dir whose basename is not ``metadata``."""
    from MEDS_extract.finalize_MEDS_metadata.finalize_MEDS_metadata import main as fmm_stage

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        metadata_in = root / "metadata_in" / "metadata"
        metadata_in.mkdir(parents=True)
        shards_fp = root / "metadata" / ".shards.json"
        shards_fp.parent.mkdir(parents=True)
        shards_fp.write_text(json.dumps({"train/0": [1]}))

        out_dir = root / "output" / "wrong_name"
        out_dir.mkdir(parents=True)

        cfg = _make_cfg(
            {
                "stage_cfg": {"metadata_input_dir": str(metadata_in), "reducer_output_dir": str(out_dir)},
                "shards_map_fp": str(shards_fp),
            }
        )

        with pytest.raises(ValueError, match="metadata"):
            fmm_stage.main_fn(cfg)


def test_finalize_MEDS_metadata_overwrite_error():
    """Existing output files raise FileExistsError when do_overwrite is False."""
    from MEDS_extract.finalize_MEDS_metadata.finalize_MEDS_metadata import main as fmm_stage

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        metadata_in = root / "metadata_in" / "metadata"
        metadata_in.mkdir(parents=True)
        shards_fp = root / "metadata" / ".shards.json"
        shards_fp.parent.mkdir(parents=True)
        shards_fp.write_text(json.dumps({"train/0": [1]}))

        out_dir = root / "output" / "metadata"
        out_dir.mkdir(parents=True)
        (out_dir / "codes.parquet").write_bytes(b"dummy")

        cfg = _make_cfg(
            {
                "do_overwrite": False,
                "stage_cfg": {"metadata_input_dir": str(metadata_in), "reducer_output_dir": str(out_dir)},
                "shards_map_fp": str(shards_fp),
            }
        )

        with pytest.raises(FileExistsError):
            fmm_stage.main_fn(cfg)


def test_finalize_MEDS_metadata_overwrite_succeeds():
    """With do_overwrite=True, existing output files are deleted and rewritten."""
    from MEDS_extract.finalize_MEDS_metadata.finalize_MEDS_metadata import main as fmm_stage

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        metadata_in = root / "metadata_in" / "metadata"
        metadata_in.mkdir(parents=True)

        shards = {"train/0": [1, 2]}
        shards_fp = root / "metadata" / ".shards.json"
        shards_fp.parent.mkdir(parents=True)
        shards_fp.write_text(json.dumps(shards))

        out_dir = root / "output" / "metadata"
        out_dir.mkdir(parents=True)

        # Pre-create output files
        (out_dir / "codes.parquet").write_bytes(b"dummy")
        (out_dir / "dataset.json").write_text("{}")
        (out_dir / "subject_splits.parquet").write_bytes(b"dummy")

        cfg = _make_cfg(
            {
                "do_overwrite": True,
                "stage_cfg": {"metadata_input_dir": str(metadata_in), "reducer_output_dir": str(out_dir)},
                "shards_map_fp": str(shards_fp),
            }
        )
        fmm_stage.main_fn(cfg)

        # Verify files were rewritten (not the dummy content)
        meta = json.loads((out_dir / "dataset.json").read_text())
        assert meta["dataset_name"] == "TEST"
