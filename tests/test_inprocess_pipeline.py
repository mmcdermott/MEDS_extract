"""In-process tests for edge cases not covered by the subprocess-based stage tests.

With coverage.py's subprocess patch enabled, the subprocess tests cover the main code paths.
These tests target specific edge cases: subject_id_expr, transforms, file globbing, missing
configs, external splits, metadata joining, and error paths.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
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


# ── convert_to_MEDS_events: subject_id_expr path (lines 315-316) ────


def test_convert_to_MEDS_events_subject_id_expr():
    """Tests the subject_id_expr path in convert_to_MEDS_events main()."""
    from MEDS_extract.convert_to_MEDS_events.convert_to_MEDS_events import main as cme_stage

    event_cfg = """\
subjects:
  subject_id_expr: "hash($MRN)"
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
        assert "subject_id" in df.columns
        assert df["subject_id"].dtype == pl.Int64


# ── convert_to_MEDS_events: transforms path (lines 321-322) ─────────


def test_convert_to_MEDS_events_with_transforms():
    """Tests the transforms path in convert_to_MEDS_events main()."""
    from MEDS_extract.convert_to_MEDS_events.convert_to_MEDS_events import main as cme_stage

    event_cfg = """\
subject_id_col: subject_id
data:
  transforms:
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


# ── shard_events: skip unconfigured files (lines 358-359) ────────────


def test_shard_events_skips_unconfigured_files():
    """Tests that shard_events skips files not in the event config."""
    from MEDS_extract.shard_events.shard_events import main as shard_stage

    minimal_cfg = """\
subject_id_col: subject_id
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


# ── extract_code_metadata: missing column error (line 165) ───────────


def test_extract_metadata_missing_column_error():
    """Tests that extract_metadata raises KeyError for missing columns."""
    from MEDS_extract.extract_code_metadata.extract_code_metadata import extract_metadata

    metadata_df = pl.DataFrame({"code": ["A"], "name": ["Code A"]}).lazy()
    event_cfg = {"code": "$code", "_metadata": {"desc": "nonexistent_column"}}

    with pytest.raises(KeyError, match="nonexistent_column"):
        extract_metadata(metadata_df, event_cfg)


# ── extract_code_metadata: multiple files + existing codes (lines 397-402, 468-470) ──


def test_extract_code_metadata_with_existing_codes():
    """Tests extract_code_metadata with existing codes.parquet for joining."""
    from MEDS_extract.extract_code_metadata.extract_code_metadata import main as ecm_stage

    metadata_cfg = """\
subject_id_col: subject_id
data:
  measurement:
    code: $lab_code
    _metadata:
      lab_meta:
        description: title
"""

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        events_dir = root / "events" / "train" / "0"
        events_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "subject_id": [1, 1],
                "time": [None, None],
                "code": ["HR", "TEMP"],
                "numeric_value": [None, None],
            }
        ).cast(
            {"subject_id": pl.Int64, "time": pl.Datetime("us"), "numeric_value": pl.Float32}
        ).write_parquet(events_dir / "data.parquet")

        raw_dir = root / "raw"
        raw_dir.mkdir()
        (raw_dir / "lab_meta.csv").write_text(
            "lab_code,title,loinc\nHR,Heart Rate,8867-4\nTEMP,Temperature,8310-5\n"
        )

        metadata_in = root / "metadata_in" / "metadata"
        metadata_in.mkdir(parents=True)
        pl.DataFrame({"code": ["EXISTING_CODE"], "description": ["An existing code"]}).write_parquet(
            metadata_in / "codes.parquet", use_pyarrow=True
        )

        event_cfg_fp = root / "event_cfgs.yaml"
        event_cfg_fp.write_text(metadata_cfg)
        shards_fp = root / "metadata" / ".shards.json"
        shards_fp.parent.mkdir(parents=True)
        shards_fp.write_text(json.dumps({"train/0": [1]}))

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

        codes_df = pl.read_parquet(out_dir / "codes.parquet")
        all_codes = set(codes_df["code"].to_list())
        assert "EXISTING_CODE" in all_codes
        assert "HR" in all_codes or "TEMP" in all_codes


# ── extract_code_metadata/utils: multiple files matching prefix (lines 127-128) ──


def test_get_supported_fp_multiple_files():
    """Tests get_supported_fp when multiple files match a prefix."""
    from MEDS_extract.extract_code_metadata.utils import get_supported_fp

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "data_part1.csv").write_text("a,b\n1,2\n")
        (root / "data_part2.csv").write_text("a,b\n3,4\n")

        fps, reader = get_supported_fp(root, "data")
        assert isinstance(fps, list)
        assert len(fps) == 2


# ── finalize_MEDS_metadata: output dir validation (line 61) ──────────


def test_finalize_MEDS_metadata_output_dir_validation():
    """Tests that finalize_MEDS_metadata validates the output dir ends in 'metadata'."""
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


# ── finalize_MEDS_metadata: overwrite error (lines 70, 72) ──────────


def test_finalize_MEDS_metadata_overwrite_error():
    """Tests that existing output files raise FileExistsError when do_overwrite is False."""
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


# ── split_and_shard_subjects: external splits edge cases ─────────────


def test_shard_subjects_external_splits_cover_all():
    """Tests shard_subjects when external splits cover all subjects (line 160)."""
    from MEDS_extract.split_and_shard_subjects.split_and_shard_subjects import shard_subjects

    subjects = np.array([1, 2, 3, 4])
    result = shard_subjects(
        subjects=subjects,
        external_splits={"train": np.array([1, 2, 3]), "test": np.array([4])},
        split_fracs_dict={},
        n_subjects_per_shard=100,
        seed=42,
    )
    assert set(result.keys()) == {"train/0", "test/0"}


def test_shard_subjects_external_splits_list_conversion():
    """Tests that non-numpy external splits are converted to numpy arrays."""
    from MEDS_extract.split_and_shard_subjects.split_and_shard_subjects import shard_subjects

    subjects = np.array([1, 2, 3, 4])
    result = shard_subjects(
        subjects=subjects,
        external_splits={"train": [1, 2, 3], "test": [4]},
        split_fracs_dict={},
        n_subjects_per_shard=100,
        seed=42,
    )
    all_ids = [i for ids in result.values() for i in ids]
    assert set(all_ids) == {1, 2, 3, 4}
