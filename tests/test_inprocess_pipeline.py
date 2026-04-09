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


# ── extract_code_metadata: multiple metadata files for one prefix (lines 397-402) ──


def test_extract_code_metadata_multiple_files_per_prefix():
    """Tests that multiple CSV files matching a metadata prefix are concatenated."""
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

        # Two CSV files matching the "lab_meta" prefix — triggers multi-file concat
        raw_dir = root / "raw"
        raw_dir.mkdir()
        (raw_dir / "lab_meta_part1.csv").write_text("lab_code,title\nHR,Heart Rate\n")
        (raw_dir / "lab_meta_part2.csv").write_text("lab_code,title\nTEMP,Body Temperature\n")

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
                    "metadata_input_dir": str(root / "empty_meta"),
                    "reducer_output_dir": str(out_dir),
                    "description_separator": "\n",
                },
                "event_conversion_config_fp": str(event_cfg_fp),
                "shards_map_fp": str(shards_fp),
            }
        )
        ecm_stage.main_fn(cfg)

        codes_df = pl.read_parquet(out_dir / "codes.parquet")
        codes = set(codes_df["code"].to_list())
        # Both codes from both files should be present
        assert "HR" in codes
        assert "TEMP" in codes


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


# ── finalize_MEDS_metadata: do_overwrite=True (line 70) ─────────────


def test_finalize_MEDS_metadata_overwrite_succeeds():
    """Tests that do_overwrite=True deletes and rewrites existing output files."""
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


# ── extract_code_metadata: duplicate codes aggregation (lines 453-460) ──


def test_extract_code_metadata_duplicate_codes_aggregation():
    """Tests description concatenation for duplicate codes from multiple metadata sources.

    Two different _metadata blocks (source_a, source_b) both produce a "description" column for the same code
    "HR". The reducer must aggregate them via str.join.
    """
    from MEDS_extract.extract_code_metadata.extract_code_metadata import main as ecm_stage

    # Two _metadata blocks pointing to different source files, both producing description for HR
    metadata_cfg = """\
subject_id_col: subject_id
data:
  measurement:
    code: $lab_code
    _metadata:
      source_a:
        description: title_a
      source_b:
        description: title_b
"""

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        events_dir = root / "events" / "train" / "0"
        events_dir.mkdir(parents=True)
        pl.DataFrame({"subject_id": [1], "time": [None], "code": ["HR"], "numeric_value": [None]}).cast(
            {"subject_id": pl.Int64, "time": pl.Datetime("us"), "numeric_value": pl.Float32}
        ).write_parquet(events_dir / "data.parquet")

        raw_dir = root / "raw"
        raw_dir.mkdir()
        (raw_dir / "source_a.csv").write_text("lab_code,title_a\nHR,Heart Rate\n")
        (raw_dir / "source_b.csv").write_text("lab_code,title_b\nHR,Pulse Rate\n")

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
                    "metadata_input_dir": str(root / "empty_meta"),
                    "reducer_output_dir": str(out_dir),
                    "description_separator": "; ",
                },
                "event_conversion_config_fp": str(event_cfg_fp),
                "shards_map_fp": str(shards_fp),
            }
        )
        ecm_stage.main_fn(cfg)

        codes_df = pl.read_parquet(out_dir / "codes.parquet")
        hr_rows = codes_df.filter(pl.col("code") == "HR")
        assert len(hr_rows) == 1
        desc = hr_rows["description"][0]
        # Both descriptions should be joined with the separator
        assert "Heart Rate" in desc
        assert "Pulse Rate" in desc
        assert "; " in desc


def test_extract_code_metadata_duplicate_codes_no_description():
    """Tests aggregation of duplicate codes when metadata has no description column.

    Covers the branch at line 454 where "description" is not in metadata_cols.
    """
    from MEDS_extract.extract_code_metadata.extract_code_metadata import main as ecm_stage

    # Two sources producing a custom property (not description) for the same code
    metadata_cfg = """\
subject_id_col: subject_id
data:
  measurement:
    code: $lab_code
    _metadata:
      source_a:
        custom_prop: val_a
      source_b:
        custom_prop: val_b
"""

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        events_dir = root / "events" / "train" / "0"
        events_dir.mkdir(parents=True)
        pl.DataFrame({"subject_id": [1], "time": [None], "code": ["HR"], "numeric_value": [None]}).cast(
            {"subject_id": pl.Int64, "time": pl.Datetime("us"), "numeric_value": pl.Float32}
        ).write_parquet(events_dir / "data.parquet")

        raw_dir = root / "raw"
        raw_dir.mkdir()
        (raw_dir / "source_a.csv").write_text("lab_code,val_a\nHR,value_1\n")
        (raw_dir / "source_b.csv").write_text("lab_code,val_b\nHR,value_2\n")

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
                    "metadata_input_dir": str(root / "empty_meta"),
                    "reducer_output_dir": str(out_dir),
                    "description_separator": "\n",
                },
                "event_conversion_config_fp": str(event_cfg_fp),
                "shards_map_fp": str(shards_fp),
            }
        )
        ecm_stage.main_fn(cfg)

        codes_df = pl.read_parquet(out_dir / "codes.parquet")
        assert "HR" in codes_df["code"].to_list()
        # custom_prop should be present
        assert "custom_prop" in codes_df.columns


def test_extract_code_metadata_code_template_survives_aggregation():
    """Tests that code_template remains a scalar string (not a list) after duplicate code aggregation."""
    from MEDS_extract.extract_code_metadata.extract_code_metadata import main as ecm_stage

    metadata_cfg = """\
subject_id_col: subject_id
data:
  measurement:
    code: $lab_code
    _metadata:
      source_a:
        description: title_a
      source_b:
        description: title_b
"""

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        events_dir = root / "events" / "train" / "0"
        events_dir.mkdir(parents=True)
        pl.DataFrame({"subject_id": [1], "time": [None], "code": ["HR"], "numeric_value": [None]}).cast(
            {"subject_id": pl.Int64, "time": pl.Datetime("us"), "numeric_value": pl.Float32}
        ).write_parquet(events_dir / "data.parquet")

        raw_dir = root / "raw"
        raw_dir.mkdir()
        (raw_dir / "source_a.csv").write_text("lab_code,title_a\nHR,Heart Rate\n")
        (raw_dir / "source_b.csv").write_text("lab_code,title_b\nHR,Pulse Rate\n")

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
                    "metadata_input_dir": str(root / "empty_meta"),
                    "reducer_output_dir": str(out_dir),
                    "description_separator": "; ",
                },
                "event_conversion_config_fp": str(event_cfg_fp),
                "shards_map_fp": str(shards_fp),
            }
        )
        ecm_stage.main_fn(cfg)

        codes_df = pl.read_parquet(out_dir / "codes.parquet")
        assert "code_template" in codes_df.columns
        # code_template must be a String, not a List — regression test for aggregation bug
        assert codes_df.schema["code_template"] == pl.String
        hr_row = codes_df.filter(pl.col("code") == "HR")
        assert hr_row["code_template"][0] == "$lab_code"


def test_extract_metadata_invalid_match_on():
    """Tests that _match_on raises KeyError when column isn't referenced by the code expression."""
    from MEDS_extract.extract_code_metadata.extract_code_metadata import extract_metadata

    metadata_df = pl.DataFrame({"medication_name": ["X"], "desc": ["Y"]}).lazy()
    event_cfg = {
        "code": 'f"{$medication_name}//{$dose}"',
        "_metadata": {"_match_on": "typo_column", "description": "desc"},
    }

    with pytest.raises(KeyError, match="not referenced by the code expression"):
        extract_metadata(metadata_df, event_cfg)


def test_extract_metadata_partial_match_multi_column():
    """Tests _match_on with multiple columns."""
    from MEDS_extract.extract_code_metadata.extract_code_metadata import extract_metadata

    metadata_df = pl.DataFrame({"a": ["X", "Y"], "b": ["1", "2"], "desc": ["X-1", "Y-2"]}).lazy()
    event_cfg = {
        "code": 'f"{$a}//{$b}//{$c}"',
        "_metadata": {"_match_on": ["a", "b"], "description": "desc"},
    }

    result = extract_metadata(metadata_df, event_cfg)
    collected = result.collect()
    assert set(collected.columns) == {"a", "b", "code_template", "description"}
    assert len(collected) == 2
    assert collected["code_template"][0] == 'f"{$a}//{$b}//{$c}"'
