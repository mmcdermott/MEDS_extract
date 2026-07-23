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
        assert "subject_id" in df.columns
        assert df["subject_id"].dtype == pl.Int64


# ── convert_to_MEDS_events: transforms path (lines 321-322) ─────────


def test_convert_to_MEDS_events_with_transforms():
    """Tests the transforms path in convert_to_MEDS_events main()."""
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


# ── convert_to_MEDS_events: new-style _defaults and _table syntax ────


def test_convert_to_MEDS_events_new_style_config():
    """Tests the new _defaults and _table config syntax."""
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
        # _defaults.subject_id should produce hashed Int64 subject IDs
        assert "subject_id" in df.columns
        assert df["subject_id"].dtype == pl.Int64
        # _table.cols.doubled should be computed before event extraction
        vals = sorted(df["numeric_value"].drop_nulls().to_list())
        assert vals == [20.0, 40.0]


# ── shard_events: skip unconfigured files (lines 358-359) ────────────


def test_shard_events_skips_unconfigured_files():
    """Tests that shard_events skips files not in the event config."""
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
        # Overlapping columns are coalesced, never forked into `*_right` (#137).
        assert "description_right" not in codes_df.columns
        by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
        assert by_code["EXISTING_CODE"] == "An existing code"
        assert by_code["HR"] == "Heart Rate"


# ── extract_code_metadata: multiple metadata files for one prefix (lines 397-402) ──


def test_extract_code_metadata_multiple_files_per_prefix():
    """Tests that multiple CSV files matching a metadata prefix are concatenated."""
    from MEDS_extract.extract_code_metadata.extract_code_metadata import main as ecm_stage

    metadata_cfg = """\
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

        # Two CSV files in a sub-sharded `lab_meta/` directory — triggers multi-file concat
        raw_dir = root / "raw"
        raw_dir.mkdir()
        (raw_dir / "lab_meta").mkdir()
        (raw_dir / "lab_meta" / "part1.csv").write_text("lab_code,title\nHR,Heart Rate\n")
        (raw_dir / "lab_meta" / "part2.csv").write_text("lab_code,title\nTEMP,Body Temperature\n")

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


def test_resolve_source_files_ambiguity_errors():
    """Resolving a prefix with both a bare file and a subsharded dir raises."""
    from MEDS_extract.io import resolve_source_files

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "labs.parquet").touch()
        (root / "labs").mkdir()
        (root / "labs" / "shard_0.parquet").touch()

        with pytest.raises(ValueError, match="Ambiguous source layout"):
            resolve_source_files(root, "labs")


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
        # custom_prop is aggregated to the canonical sorted List(String) shape (#137).
        assert codes_df.schema["custom_prop"] == pl.List(pl.String)
        hr_row = codes_df.filter(pl.col("code") == "HR")
        assert hr_row["custom_prop"][0].to_list() == ["value_1", "value_2"]


def test_extract_code_metadata_code_template_survives_aggregation():
    """Tests that code_template aggregates to a sorted unique List(String) (#137).

    ``.first()`` used to keep only whichever source's template happened to arrive first —
    nondeterministic under worker shuffle, and silently dropping the other sources'
    provenance. The canonical shape preserves every contributing template exactly once.
    """
    from MEDS_extract.extract_code_metadata.extract_code_metadata import main as ecm_stage

    metadata_cfg = """\
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
        # code_template is a sorted unique List(String): both sources share the same code
        # expression, so exactly one template survives — no provenance is dropped and no
        # duplicate is kept.
        assert codes_df.schema["code_template"] == pl.List(pl.String)
        hr_row = codes_df.filter(pl.col("code") == "HR")
        templates = hr_row["code_template"][0].to_list()
        assert len(templates) == 1
        # Since the config now holds parsed dftly nodes, the code_template column renders
        # the node's repr form ("Column('lab_code')") rather than the original user string
        # ("$lab_code"). Pending upstream dftly support for a stable string form.
        assert "lab_code" in templates[0]


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


def test_extract_metadata_partial_match_missing_match_col():
    """Tests that _match_on raises KeyError when the column doesn't exist in the metadata table."""
    from MEDS_extract.extract_code_metadata.extract_code_metadata import extract_metadata

    # medication_name is referenced by the code expression but missing from the metadata table
    metadata_df = pl.DataFrame({"dose": ["500mg"], "desc": ["some desc"]}).lazy()
    event_cfg = {
        "code": 'f"{$medication_name}//{$dose}"',
        "_metadata": {"_match_on": "medication_name", "description": "desc"},
    }

    with pytest.raises(KeyError, match="_match_on columns"):
        extract_metadata(metadata_df, event_cfg)


def test_extract_metadata_partial_match_missing_metadata_col():
    """Tests that partial match raises KeyError when metadata output column doesn't exist."""
    from MEDS_extract.extract_code_metadata.extract_code_metadata import extract_metadata

    metadata_df = pl.DataFrame({"medication_name": ["X"]}).lazy()
    event_cfg = {
        "code": 'f"{$medication_name}//{$dose}"',
        "_metadata": {"_match_on": "medication_name", "description": "nonexistent_col"},
    }

    with pytest.raises(KeyError, match="nonexistent_col"):
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


# ── Bug regression: mixed-schema parquet scan crashes on heterogeneous event files ──


def test_mixed_schema_parquet_scan_with_and_without_code_components():
    """Regression guard: metadata extraction must handle heterogeneous event parquet schemas.

    Some event files have a code_components column (dynamic codes like f"{$test_name}//{$units}")
    and others don't (literal codes like "ADMISSION"). The reducer must scan these mixed-schema
    files without crashing. Previously a single glob scan_parquet raised on schema mismatches.
    """
    from MEDS_extract.extract_code_metadata.extract_code_metadata import main as ecm_stage

    # Two input prefixes: "labs" has a dynamic code (produces code_components),
    # "admissions" has a literal code (no code_components).
    metadata_cfg = """\
labs:
  measurement:
    code: 'f"{$test_name}//{$units}"'
    _metadata:
      lab_meta:
        description: title
admissions:
  admit:
    code: ADMISSION
    time: null
"""

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        events_dir = root / "events" / "train" / "0"
        events_dir.mkdir(parents=True)

        # Event file WITH code_components (dynamic code)
        pl.DataFrame(
            {
                "subject_id": [1, 2],
                "time": [None, None],
                "code": ["Glucose//mg/dL", "BUN//mg/dL"],
                "code_components": [
                    {"test_name": "Glucose", "units": "mg/dL"},
                    {"test_name": "BUN", "units": "mg/dL"},
                ],
                "source_block": ["labs/measurement", "labs/measurement"],
                "numeric_value": [100.0, 20.0],
            }
        ).cast(
            {"subject_id": pl.Int64, "time": pl.Datetime("us"), "numeric_value": pl.Float32}
        ).write_parquet(events_dir / "labs.parquet")

        # Event file WITHOUT code_components (literal code)
        pl.DataFrame(
            {
                "subject_id": [1],
                "time": [None],
                "code": ["ADMISSION"],
                "source_block": ["admissions/admit"],
                "numeric_value": [None],
            }
        ).cast(
            {"subject_id": pl.Int64, "time": pl.Datetime("us"), "numeric_value": pl.Float32}
        ).write_parquet(events_dir / "admissions.parquet")

        raw_dir = root / "raw"
        raw_dir.mkdir()
        (raw_dir / "lab_meta.csv").write_text("test_name,units,title\nGlucose,mg/dL,Blood Glucose\n")

        event_cfg_fp = root / "event_cfgs.yaml"
        event_cfg_fp.write_text(metadata_cfg)
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
        assert "Glucose//mg/dL" in codes_df["code"].to_list()
        assert codes_df.filter(pl.col("code") == "Glucose//mg/dL")["description"][0] == "Blood Glucose"


# ── Bug regression: partial-match join keys inferred from schema intersection ──


def test_partial_match_join_key_not_inferred_from_schema_intersection():
    """Regression guard: partial-match join keys must use explicit _match_on, not schema intersection.

    If a metadata output column shares a name with a code component column, it must not be treated
    as a join key. Only the explicit _match_on columns should be used. Previously the reducer
    inferred join keys from schema intersection, which over-constrained the join and silently
    dropped matches when column names collided.

    All event files here use dynamic codes (uniform schema) to isolate this from the mixed-schema bug.
    """
    from MEDS_extract.extract_code_metadata.extract_code_metadata import main as ecm_stage

    # Code is f"{$category}//{$item}" — so code_component_map has columns: code, category, item.
    # Metadata is keyed on _match_on: category, and has an output column ALSO named "item"
    # (e.g., the metadata table has its own "item" column with different values).
    # The reducer will incorrectly treat "item" as a join key too, because it appears in both
    # the partial metadata shard and code_component_map.columns.
    metadata_cfg = """\
data:
  event:
    code: 'f"{$category}//{$item}"'
    _metadata:
      category_meta:
        _match_on: category
        item: item_description
"""

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        events_dir = root / "events" / "train" / "0"
        events_dir.mkdir(parents=True)

        # ALL event files have code_components (uniform schema — avoids mixed-schema bug).
        # Events: category=Drug, item=Aspirin; category=Drug, item=Ibuprofen
        pl.DataFrame(
            {
                "subject_id": [1, 2],
                "time": [None, None],
                "code": ["Drug//Aspirin", "Drug//Ibuprofen"],
                "code_components": [
                    {"category": "Drug", "item": "Aspirin"},
                    {"category": "Drug", "item": "Ibuprofen"},
                ],
                "source_block": ["data/event", "data/event"],
                "numeric_value": [None, None],
            }
        ).cast(
            {"subject_id": pl.Int64, "time": pl.Datetime("us"), "numeric_value": pl.Float32}
        ).write_parquet(events_dir / "data.parquet")

        raw_dir = root / "raw"
        raw_dir.mkdir()
        # Metadata: category_meta maps category -> item_description.
        # The output column is named "item" (matching a code component name!),
        # but contains description text, not actual item values.
        (raw_dir / "category_meta.csv").write_text(
            "category,item_description\nDrug,Pharmaceutical compound\n"
        )

        event_cfg_fp = root / "event_cfgs.yaml"
        event_cfg_fp.write_text(metadata_cfg)
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
        # The "item" column should be present in the output — it's a metadata output column.
        assert "item" in codes_df.columns, (
            f"Expected 'item' column in output (metadata output from partial match), "
            f"but columns are: {codes_df.columns}.\nFull output:\n{codes_df}"
        )
        codes_with_item = codes_df.filter(pl.col("item").is_not_null())
        # Both Drug//Aspirin and Drug//Ibuprofen should get "Pharmaceutical compound"
        # because _match_on is only "category" and both share category=Drug.
        # With the bug, the join also matches on "item" column, so neither row matches
        # (because "Pharmaceutical compound" != "Aspirin" or "Ibuprofen").
        assert len(codes_with_item) == 2, (
            f"Expected 2 codes with item metadata (both Drug codes should match via category), "
            f"got {len(codes_with_item)}.\nFull output:\n{codes_df}"
        )


# ── Bug regression: mixed full-match and partial-match from the same metadata prefix ──


def test_mixed_full_and_partial_match_from_same_metadata_prefix():
    """Regression guard: mixed full-match and partial-match configs sharing a metadata prefix.

    A single metadata file prefix can be referenced by multiple event configs with different
    match modes. Each must be written to a separate intermediate shard so the reducer can
    classify and expand them independently. Previously all configs for one prefix were
    concatenated into one shard, and the reducer treated the whole shard as full-match
    (because "code" was in the schema), silently dropping partial-match rows.

    This test uses "shared_meta" referenced by a full-match config (code: $lab_code) and a
    partial-match config (code: f"{$category}//{$item}", _match_on: category).
    """
    from MEDS_extract.extract_code_metadata.extract_code_metadata import main as ecm_stage

    # Two event configs reference the same metadata prefix "shared_meta":
    # - labs/measurement: full-match on $lab_code
    # - products/product: partial-match on category via _match_on
    metadata_cfg = """\
labs:
  measurement:
    code: $lab_code
    _metadata:
      shared_meta:
        description: desc
products:
  product:
    code: 'f"{$category}//{$item}"'
    _metadata:
      shared_meta:
        _match_on: category
        description: desc
"""

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        events_dir = root / "events" / "train" / "0"
        events_dir.mkdir(parents=True)

        # Lab events (full match — code is a simple column ref, no code_components)
        pl.DataFrame(
            {
                "subject_id": [1],
                "time": [None],
                "code": ["HR"],
                "source_block": ["labs/measurement"],
                "numeric_value": [None],
            }
        ).cast(
            {"subject_id": pl.Int64, "time": pl.Datetime("us"), "numeric_value": pl.Float32}
        ).write_parquet(events_dir / "labs.parquet")

        # Product events (partial match — dynamic code with code_components)
        pl.DataFrame(
            {
                "subject_id": [1, 2],
                "time": [None, None],
                "code": ["Drug//Aspirin", "Drug//Ibuprofen"],
                "code_components": [
                    {"category": "Drug", "item": "Aspirin"},
                    {"category": "Drug", "item": "Ibuprofen"},
                ],
                "source_block": ["products/product", "products/product"],
                "numeric_value": [None, None],
            }
        ).cast(
            {"subject_id": pl.Int64, "time": pl.Datetime("us"), "numeric_value": pl.Float32}
        ).write_parquet(events_dir / "products.parquet")

        raw_dir = root / "raw"
        raw_dir.mkdir()
        # Shared metadata file: has lab_code, category, and desc columns.
        # "HR" matches full-match via lab_code; "Drug" matches partial-match via category.
        (raw_dir / "shared_meta.csv").write_text("lab_code,category,desc\nHR,Drug,Shared description\n")

        event_cfg_fp = root / "event_cfgs.yaml"
        event_cfg_fp.write_text(metadata_cfg)
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
        codes_with_desc = codes_df.filter(pl.col("description").is_not_null())
        matched_codes = set(codes_with_desc["code"].to_list())

        # Full-match: HR should get description from shared_meta via lab_code
        assert "HR" in matched_codes, f"Full-match code 'HR' missing from output.\n{codes_df}"
        # Partial-match: Drug//Aspirin and Drug//Ibuprofen should get description via category=Drug
        assert "Drug//Aspirin" in matched_codes, (
            f"Partial-match code 'Drug//Aspirin' missing from output.\n{codes_df}"
        )
        assert "Drug//Ibuprofen" in matched_codes, (
            f"Partial-match code 'Drug//Ibuprofen' missing from output.\n{codes_df}"
        )


# ── Coverage: no _metadata blocks early return ──


def test_extract_code_metadata_no_metadata_blocks():
    """Covers the early return when no _metadata blocks are found in the event config."""
    from MEDS_extract.extract_code_metadata.extract_code_metadata import main as ecm_stage

    # Event config with no _metadata blocks at all
    metadata_cfg = """\
data:
  measurement:
    code: $lab_code
    time: null
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
        # Should return early without error — no metadata to extract
        ecm_stage.main_fn(cfg)
        # No codes.parquet should be written since we never reach the reducer
        assert not (out_dir / "codes.parquet").exists()


# ── Coverage: partial-match with no code_components in event data ──


def test_partial_match_without_code_components_in_events():
    """Covers the warning path when partial-match metadata exists but event data has no code_components."""
    from MEDS_extract.extract_code_metadata.extract_code_metadata import main as ecm_stage

    metadata_cfg = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      lab_meta:
        _match_on: lab_code
        description: title
"""

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        events_dir = root / "events" / "train" / "0"
        events_dir.mkdir(parents=True)
        # Event file WITHOUT code_components (literal code via $col)
        pl.DataFrame({"subject_id": [1], "time": [None], "code": ["HR"], "numeric_value": [None]}).cast(
            {"subject_id": pl.Int64, "time": pl.Datetime("us"), "numeric_value": pl.Float32}
        ).write_parquet(events_dir / "data.parquet")

        raw_dir = root / "raw"
        raw_dir.mkdir()
        (raw_dir / "lab_meta.csv").write_text("lab_code,title\nHR,Heart Rate\n")

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
        # Should complete without error but partial-match rows won't be expanded
        ecm_stage.main_fn(cfg)
        codes_df = pl.read_parquet(out_dir / "codes.parquet")
        # The output will be empty since partial match can't expand without code_components
        assert len(codes_df) == 0


# ── Coverage: empty metadata result (no matching codes) ──


def test_extract_code_metadata_no_matching_codes():
    """Covers the 'no metadata to reduce' empty output path."""
    from MEDS_extract.extract_code_metadata.extract_code_metadata import main as ecm_stage

    metadata_cfg = """\
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
        # Event data has code "HR" but metadata has no matching lab_code
        pl.DataFrame({"subject_id": [1], "time": [None], "code": ["HR"], "numeric_value": [None]}).cast(
            {"subject_id": pl.Int64, "time": pl.Datetime("us"), "numeric_value": pl.Float32}
        ).write_parquet(events_dir / "data.parquet")

        raw_dir = root / "raw"
        raw_dir.mkdir()
        # Metadata has lab_code "NONEXISTENT" — no match with event codes
        (raw_dir / "lab_meta.csv").write_text("lab_code,title\nNONEXISTENT,No Match\n")

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
        assert len(codes_df) == 0


# ── Partial-match correctness regressions (#110, #134, #135) ──


def _run_ecm_scenario(
    root: Path,
    messy_yaml: str,
    event_frames: dict[str, pl.DataFrame],
    raw_files: dict[str, str | pl.DataFrame],
    existing_codes: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Run the extract_code_metadata stage over synthetic event shards and raw metadata files.

    ``event_frames`` maps parquet basenames to frames of extra columns (code, code_components,
    source_block, ...); the standard subject_id/time/numeric_value columns are added here.
    ``raw_files`` maps raw metadata file paths (which may include subdirectories) to either
    text content (written verbatim) or a DataFrame (written as parquet). ``existing_codes``,
    when given, is written as a pre-existing ``metadata/codes.parquet`` for the reducer to
    merge with. Returns the reduced ``codes.parquet`` as a DataFrame.
    """
    from MEDS_extract.extract_code_metadata.extract_code_metadata import main as ecm_stage

    events_dir = root / "events" / "train" / "0"
    events_dir.mkdir(parents=True)
    for basename, frame in event_frames.items():
        n = len(frame)
        frame.with_columns(
            subject_id=pl.Series(range(1, n + 1), dtype=pl.Int64),
            time=pl.lit(None, dtype=pl.Datetime("us")),
            numeric_value=pl.lit(None, dtype=pl.Float32),
        ).write_parquet(events_dir / f"{basename}.parquet")

    raw_dir = root / "raw"
    raw_dir.mkdir()
    for fname, content in raw_files.items():
        fp = raw_dir / fname
        fp.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, pl.DataFrame):
            content.write_parquet(fp)
        else:
            fp.write_text(content)

    metadata_in = root / "empty_meta"
    if existing_codes is not None:
        metadata_in = root / "metadata_in" / "metadata"
        metadata_in.mkdir(parents=True)
        existing_codes.write_parquet(metadata_in / "codes.parquet", use_pyarrow=True)

    event_cfg_fp = root / "event_cfgs.yaml"
    event_cfg_fp.write_text(messy_yaml)
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
    return pl.read_parquet(out_dir / "codes.parquet")


def test_partial_match_on_column_named_code_is_expanded_not_passed_through():
    """Regression guard (#110): a partial output keyed on a column named ``code`` is partial.

    ``_match_on: code`` makes the intermediate shard carry a ``code`` column of raw component
    values (e.g. ``250.00``). The reducer used to classify shards by sniffing for a ``code``
    column in the schema, misclassifying this shard as full-match and passing the raw
    component values through as output codes — silently wrong codes.parquet. Classification
    must come from explicit map-time bookkeeping instead.
    """
    messy = """\
diagnoses:
  dx:
    code: 'f"ICD//{$code}"'
    _metadata:
      icd_meta:
        _match_on: code
        description: long_title
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "diagnoses": pl.DataFrame(
                    {
                        "code": ["ICD//250.00", "ICD//401.9"],
                        "code_components": [{"code": "250.00"}, {"code": "401.9"}],
                        "source_block": ["diagnoses/dx", "diagnoses/dx"],
                    }
                )
            },
            raw_files={"icd_meta.csv": "code,long_title\n250.00,Diabetes mellitus\n401.9,Hypertension\n"},
        )

        by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
        # Partial-match expansion: metadata lands on the FULL codes...
        assert by_code.get("ICD//250.00") == "Diabetes mellitus"
        assert by_code.get("ICD//401.9") == "Hypertension"
        # ...and the raw component values are NOT passed through as codes (the old
        # full-match misclassification symptom).
        assert "250.00" not in by_code
        assert "401.9" not in by_code


def test_partial_match_scoped_to_declaring_event():
    """Regression guard (#134): ``_match_on`` expansion is scoped to the declaring event.

    Two events build codes from a same-named ``itemid`` component with colliding values, but
    only the chartevents event declares the ``d_items`` metadata. The labevents code must NOT
    receive that metadata, and no output row may carry the declaring config's code_template
    as false provenance for a labevents code.
    """
    messy = """\
chartevents:
  chart:
    code: 'f"CHART//{$itemid}"'
    _metadata:
      d_items:
        _match_on: itemid
        description: label
labevents:
  lab:
    code: 'f"LAB//{$itemid}"'
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "chartevents": pl.DataFrame(
                    {
                        "code": ["CHART//1"],
                        "code_components": [{"itemid": "1"}],
                        "source_block": ["chartevents/chart"],
                    }
                ),
                "labevents": pl.DataFrame(
                    {
                        "code": ["LAB//1"],
                        "code_components": [{"itemid": "1"}],
                        "source_block": ["labevents/lab"],
                    }
                ),
            },
            raw_files={"d_items.csv": "itemid,label\n1,Heart Rate (chart)\n"},
        )

        chart_rows = codes_df.filter(pl.col("code") == "CHART//1")
        assert chart_rows["description"].to_list() == ["Heart Rate (chart)"]
        assert chart_rows["code_template"].to_list() == [['f"CHART//{$itemid}"']]

        # The labevents code must not receive the chartevents-declared metadata.
        lab_rows = codes_df.filter(pl.col("code") == "LAB//1")
        assert lab_rows["description"].drop_nulls().to_list() == [], (
            f"LAB//1 must not receive metadata declared on the chartevents event.\n{codes_df}"
        )
        assert lab_rows["code_template"].drop_nulls().to_list() == [], (
            f"LAB//1 must not be stamped with the chartevents code_template.\n{codes_df}"
        )


def test_partial_match_typed_int_components_join_csv_metadata():
    """Regression guard (#135): typed ``Int64`` components join against all-String CSV keys.

    CSV metadata sources are read with ``infer_schema=False`` (all-String), while code
    components keep their raw source dtypes. The reducer join used to crash with
    ``SchemaError: datatypes of join keys don't match``; join keys must be normalized to
    String on both sides.
    """
    messy = """\
chartevents:
  chart:
    code: 'f"CHART//{$itemid}"'
    _metadata:
      d_items:
        _match_on: itemid
        description: label
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "chartevents": pl.DataFrame(
                    {
                        "code": ["CHART//220045", "CHART//220179"],
                        "code_components": [{"itemid": 220045}, {"itemid": 220179}],
                        "source_block": ["chartevents/chart", "chartevents/chart"],
                    }
                )
            },
            raw_files={"d_items.csv": "itemid,label\n220045,Heart Rate\n220179,NBP systolic\n"},
        )

        by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
        assert by_code.get("CHART//220045") == "Heart Rate"
        assert by_code.get("CHART//220179") == "NBP systolic"


def test_partial_match_integer_valued_float_components_join_csv_metadata():
    """Regression guard (#135): integer-valued float components render as ``220045``.

    A ``Float64`` component with value ``220045.0`` must match the metadata string
    ``"220045"`` — a plain String cast would render ``"220045.0"`` and silently zero-match.
    Non-integer float values keep their float rendering.
    """
    messy = """\
chartevents:
  chart:
    code: 'f"CHART//{$itemid}"'
    _metadata:
      d_items:
        _match_on: itemid
        description: label
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "chartevents": pl.DataFrame(
                    {
                        "code": ["CHART//220045", "CHART//1.5"],
                        "code_components": [{"itemid": 220045.0}, {"itemid": 1.5}],
                        "source_block": ["chartevents/chart", "chartevents/chart"],
                    }
                )
            },
            raw_files={"d_items.csv": "itemid,label\n220045,Heart Rate\n1.5,Half Item\n"},
        )

        by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
        assert by_code.get("CHART//220045") == "Heart Rate"
        assert by_code.get("CHART//1.5") == "Half Item"


def test_partial_match_zero_matches_warns(caplog):
    """A partial-match join that matches zero codes emits a WARNING (minimal diagnostic).

    Full match-coverage diagnostics are tracked in #138; this only guards the silent-miss case introduced
    alongside the #135 dtype normalization.
    """
    messy = """\
chartevents:
  chart:
    code: 'f"CHART//{$itemid}"'
    _metadata:
      d_items:
        _match_on: itemid
        description: label
"""
    with tempfile.TemporaryDirectory() as d, caplog.at_level("WARNING"):
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "chartevents": pl.DataFrame(
                    {
                        "code": ["CHART//1"],
                        "code_components": [{"itemid": "1"}],
                        "source_block": ["chartevents/chart"],
                    }
                )
            },
            raw_files={"d_items.csv": "itemid,label\n999,No Such Item\n"},
        )

        assert len(codes_df.filter(pl.col("description").is_not_null())) == 0
        assert "matched zero codes" in caplog.text


# ── Full-match null-component rendering asymmetry regressions (#136) ──


def test_full_match_null_component_metadata_links_to_unk_code():
    """Regression guard (#136): a null-component mapping row links to the ``UNK`` code.

    The data side renders a labevents row with ``itemid=51463`` and null ``valueuom`` as
    ``LAB//RESULT//51463//UNK``. The metadata side used to reconstruct codes with the raw
    null-propagating dftly expression, so the matching mapping row (itemid ``51463``, null
    ``valueuom``) evaluated to a null code and was silently dropped — the ``UNK`` code could
    never receive its metadata. Both sides must share one compiled code rendering.
    """
    messy = """\
labevents:
  lab:
    code: 'f"LAB//RESULT//{$itemid}//{$valueuom}"'
    _metadata:
      d_labitems_to_loinc:
        description: label
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "labevents": pl.DataFrame(
                    {
                        "code": ["LAB//RESULT//51463//UNK", "LAB//RESULT//51464//mg/dL"],
                        "code_components": [
                            {"itemid": "51463", "valueuom": None},
                            {"itemid": "51464", "valueuom": "mg/dL"},
                        ],
                        "source_block": ["labevents/lab", "labevents/lab"],
                    }
                )
            },
            raw_files={
                "d_labitems_to_loinc.csv": (
                    "itemid,valueuom,label\n51463,,Yeast [Presence] in Urine\n51464,mg/dL,Bilirubin Total\n"
                )
            },
        )

        by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
        # The null-valueuom mapping row must land on the UNK code the data actually emits.
        assert by_code.get("LAB//RESULT//51463//UNK") == "Yeast [Presence] in Urine", (
            f"Null-component mapping row failed to link to the UNK code.\n{codes_df}"
        )
        # Fully-populated mapping rows keep linking as before.
        assert by_code.get("LAB//RESULT//51464//mg/dL") == "Bilirubin Total"


def test_full_match_bare_column_code_drops_null_metadata_keys():
    """Regression guard (#136): bare-column codes drop null metadata keys, mirroring the data side.

    A bare-column code null-propagates on both sides; the data side drops null-code rows (no identifier means
    no event), so the metadata side must likewise drop mapping rows whose key column is null — not crash and
    not emit a null code.
    """
    messy = """\
diagnoses:
  dx:
    code: $icd_code
    _metadata:
      icd_meta:
        description: long_title
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "diagnoses": pl.DataFrame(
                    {
                        "code": ["I10"],
                        "code_components": [{"icd_code": "I10"}],
                        "source_block": ["diagnoses/dx"],
                    }
                )
            },
            raw_files={"icd_meta.csv": "icd_code,long_title\nI10,Hypertension\n,orphan row\n"},
        )

        by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
        assert by_code.get("I10") == "Hypertension"
        # The null-key mapping row is dropped, never emitted as a null code.
        assert None not in by_code, f"Null metadata key must be dropped, not emitted.\n{codes_df}"


# ── Reducer determinism, canonical schema, coalescing merge, and crash bugs (#137) ──


_TWO_SOURCE_MESSY = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      source_a:
        description: title_a
        vocab: vocab_a
      source_b:
        description: title_b
        vocab: vocab_b
"""

_TWO_SOURCE_EVENTS = {"data": pl.DataFrame({"code": ["HR", "TEMP"]})}

_TWO_SOURCE_RAW = {
    "source_a.csv": "lab_code,title_a,vocab_a\nHR,Heart Rate,LOINC\nTEMP,Temperature,LOINC\n",
    "source_b.csv": "lab_code,title_b,vocab_b\nHR,Pulse Rate,SNOMED\n",
}


def test_reduction_is_deterministic_across_config_orderings(monkeypatch):
    """Regression guard (#137): ``codes.parquet`` is byte-identical regardless of shuffle order.

    Each worker shuffles its metadata configs (for lock-contention spreading), and the shuffle
    order used to flow straight into the reduction: description join order and code_template
    selection depended on which partial file was concatenated first. Two runs over identical
    inputs are forced through *opposite* config orderings here — a no-op shuffle vs. a
    reversing shuffle — and must produce byte-identical output files.
    """
    from MEDS_extract.extract_code_metadata import extract_code_metadata as ecm_mod

    outputs: list[bytes] = []
    frames: list[pl.DataFrame] = []
    for shuffle in (lambda x: None, lambda x: x.reverse()):
        with tempfile.TemporaryDirectory() as d:
            monkeypatch.setattr(ecm_mod.random, "shuffle", shuffle)
            _run_ecm_scenario(Path(d), _TWO_SOURCE_MESSY, _TWO_SOURCE_EVENTS, _TWO_SOURCE_RAW)
            fp = Path(d) / "metadata_out" / "metadata" / "codes.parquet"
            outputs.append(fp.read_bytes())
            frames.append(pl.read_parquet(fp))

    from polars.testing import assert_frame_equal

    # Frame-level identity first (row order, list order, dtypes all strict) for a readable
    # diff on failure; then full byte identity of the on-disk files.
    assert_frame_equal(frames[0], frames[1], check_row_order=True, check_column_order=True)
    assert outputs[0] == outputs[1], "codes.parquet bytes differ across config orderings"

    # The canonical ordering has teeth: descriptions joined in config order, values sorted.
    by_code = {r["code"]: r for r in frames[0].iter_rows(named=True)}
    assert by_code["HR"]["description"] == "Heart Rate\nPulse Rate"
    assert by_code["HR"]["vocab"] == ["LOINC", "SNOMED"]


def test_reduced_schema_is_data_independent():
    """Regression guard (#137): extra metadata columns are always ``List(String)``.

    The old reducer only aggregated when some code was duplicated across metadata rows, so
    the *dtype* of extra columns flipped between ``String`` and ``List(String)`` depending on
    the data. A single-source, unique-code extraction must now yield the same schema as a
    multi-source one — one-element lists — while ``description`` keeps its MEDS-mandated
    separator-joined String form.
    """
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      lab_meta:
        description: title
        vocab: vocab
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": pl.DataFrame({"code": ["HR"]})},
            raw_files={"lab_meta.csv": "lab_code,title,vocab\nHR,Heart Rate,LOINC\n"},
        )

    assert codes_df.schema["description"] == pl.String
    assert codes_df.schema["vocab"] == pl.List(pl.String)
    assert codes_df.schema["code_template"] == pl.List(pl.String)
    row = codes_df.filter(pl.col("code") == "HR").to_dicts()[0]
    assert row["description"] == "Heart Rate"
    assert row["vocab"] == ["LOINC"]
    assert row["code_template"] == ["$lab_code"]


def test_reduced_missing_values_are_null_not_empty():
    """A code with no value for a metadata column gets null — never ``[]`` or ``""`` (#137).

    ``TEMP`` appears only in source_a, so its source_b-only column must be null, and its
    description must be exactly the single source_a value (no stray separator).
    """
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(Path(d), _TWO_SOURCE_MESSY, _TWO_SOURCE_EVENTS, _TWO_SOURCE_RAW)

    temp_row = codes_df.filter(pl.col("code") == "TEMP").to_dicts()[0]
    assert temp_row["description"] == "Temperature"
    assert temp_row["vocab"] == ["LOINC"]


def test_preexisting_codes_merge_coalesces_overlapping_columns():
    """Regression guard (#137): merging with a pre-existing ``codes.parquet`` coalesces columns.

    The full join used to fork overlapping columns into ``description`` + ``description_right``.
    Overlaps must coalesce into a single column with extracted values taking precedence;
    pre-existing values survive wherever nothing was re-extracted.
    """
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      lab_meta:
        description: title
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": pl.DataFrame({"code": ["HR", "NEW"]})},
            raw_files={"lab_meta.csv": "lab_code,title\nHR,Fresh heart rate\nNEW,A new code\n"},
            existing_codes=pl.DataFrame(
                {
                    "code": ["HR", "LEGACY"],
                    "description": ["Stale heart rate", "Legacy-only code"],
                }
            ),
        )

    assert "description_right" not in codes_df.columns
    assert codes_df.columns.count("description") == 1
    by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
    # Extracted value wins on overlap.
    assert by_code["HR"] == "Fresh heart rate"
    # Pre-existing survives where not re-extracted; newly extracted codes appear.
    assert by_code["LEGACY"] == "Legacy-only code"
    assert by_code["NEW"] == "A new code"


def test_preexisting_codes_merge_dtype_conflict_names_column():
    """A dtype conflict between pre-existing and extracted same-named columns raises clearly (#137)."""
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      lab_meta:
        description: title
        vocab: vocab
"""
    with (
        tempfile.TemporaryDirectory() as d,
        pytest.raises(ValueError, match="column 'vocab'"),
    ):
        _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": pl.DataFrame({"code": ["HR"]})},
            raw_files={"lab_meta.csv": "lab_code,title,vocab\nHR,Heart Rate,LOINC\n"},
            # Pre-existing `vocab` is String; the extracted canonical form is List(String).
            existing_codes=pl.DataFrame({"code": ["HR"], "vocab": ["OLD"]}),
        )


def test_match_on_column_that_is_also_a_metadata_output_works():
    """Regression guard (#137): ``_match_on`` on a renamed key column must not crash.

    Declaring the join key as a ``_metadata`` output expression (``itemid: itemid_alias``) is
    the only key-rename mechanism available; it used to raise ``DuplicateError`` at the
    mapper's final select because the column was selected both as key and as output.
    """
    messy = """\
chartevents:
  chart:
    code: 'f"CHART//{$itemid}"'
    _metadata:
      d_items:
        _match_on: itemid
        itemid: itemid_alias
        description: label
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "chartevents": pl.DataFrame(
                    {
                        "code": ["CHART//220045"],
                        "code_components": [{"itemid": "220045"}],
                        "source_block": ["chartevents/chart"],
                    }
                )
            },
            # The metadata table has no `itemid` column — the key is renamed from `itemid_alias`.
            raw_files={"d_items.csv": "itemid_alias,label\n220045,Heart Rate\n"},
        )

    by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
    assert by_code.get("CHART//220045") == "Heart Rate"


def test_mixed_format_metadata_prefix_chunks():
    """Regression guard (#137): a metadata prefix mixing csv and parquet chunks must not crash.

    Read kwargs used to be chosen from the *first* resolved file only, so a csv-first prefix
    passed the csv-only ``infer_schema`` kwarg into ``scan_parquet`` → ``TypeError``. Format
    dispatch now happens per file; typed parquet keys unify with all-String csv keys.
    """
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      lab_meta:
        description: title
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": pl.DataFrame({"code": ["HR", "TEMP"]})},
            raw_files={
                # Sub-sharded prefix directory with one csv chunk and one parquet chunk;
                # `resolve_source_files` sorts by name, so the csv is scanned first.
                "lab_meta/chunk_a.csv": "lab_code,title\nHR,Heart Rate\n",
                "lab_meta/chunk_b.parquet": pl.DataFrame(
                    {"lab_code": ["TEMP"], "title": ["Body Temperature"]}
                ),
            },
        )

    by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
    assert by_code.get("HR") == "Heart Rate"
    assert by_code.get("TEMP") == "Body Temperature"
