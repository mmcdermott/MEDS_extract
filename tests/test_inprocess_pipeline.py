"""In-process tests for pipeline stage main() functions.

The existing subprocess-based tests cover correctness but don't contribute to coverage because they run stages
as separate processes. These tests call main() functions directly with constructed DictConfigs to fill that
gap.
"""

import json
import tempfile
from io import StringIO
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
from omegaconf import OmegaConf

_ = pl.Config.set_tbl_width_chars(600)

# ── Shared test data ──────────────────────────────────────────────────

SUBJECTS_CSV = """\
MRN,dob,eye_color,height
1195293,06/20/1978,BLUE,164.6868838269085
239684,12/28/1980,BROWN,175.271115221764
1500733,07/20/1986,BROWN,158.60131573580904
814703,03/28/1976,HAZEL,156.48559093209357
754281,12/19/1988,BROWN,166.22261567137025
68729,03/09/1978,HAZEL,160.3953106166676
"""

ADMIT_VITALS_CSV = """\
subject_id,admit_date,disch_date,department,vitals_date,HR,temp
239684,"05/11/2010, 17:41:51","05/11/2010, 19:27:19",CARDIAC,"05/11/2010, 17:41:51",102.6,96.0
239684,"05/11/2010, 17:41:51","05/11/2010, 19:27:19",CARDIAC,"05/11/2010, 17:48:48",105.1,96.2
68729,"05/26/2010, 02:30:56","05/26/2010, 04:51:52",PULMONARY,"05/26/2010, 02:30:56",86.0,97.8
1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 19:23:52",109.0,100.0
"""

EVENT_CFGS_YAML = """\
subject_id_col: subject_id
subjects:
  subject_id_col: MRN
  eye_color:
    code: 'f"EYE_COLOR//{$eye_color}"'
    time: null
  height:
    code: '"HEIGHT"'
    time: null
    numeric_value: "$height"
  dob:
    code: '"DOB"'
    time: '$dob::"%m/%d/%Y"'
admit_vitals:
  admissions:
    code: 'f"ADMISSION//{$department}"'
    time: '$admit_date::"%m/%d/%Y, %H:%M:%S"'
  discharge:
    code: '"DISCHARGE"'
    time: '$disch_date::"%m/%d/%Y, %H:%M:%S"'
  HR:
    code: '"HR"'
    time: '$vitals_date::"%m/%d/%Y, %H:%M:%S"'
    numeric_value: "$HR"
  temp:
    code: '"TEMP"'
    time: '$vitals_date::"%m/%d/%Y, %H:%M:%S"'
    numeric_value: "$temp"
"""


def _make_cfg(overrides: dict) -> OmegaConf:
    """Build a minimal DictConfig mimicking what MEDS-Transforms provides to stages.

    Uses struct=False so that cfg.get() calls for optional keys work without errors.
    """
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


# ── Stage 1: shard_events ─────────────────────────────────────────────


def test_shard_events_inprocess():
    from MEDS_extract.shard_events.shard_events import main as shard_stage

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        # shard_events uses data_input_dir.parent as raw_cohort_dir, so raw files go at that parent level
        raw_dir = root / "raw_cohort"
        raw_dir.mkdir()

        pl.read_csv(StringIO(SUBJECTS_CSV)).write_parquet(raw_dir / "subjects.parquet")
        pl.read_csv(StringIO(ADMIT_VITALS_CSV)).write_parquet(raw_dir / "admit_vitals.parquet")

        event_cfg_fp = root / "event_cfgs.yaml"
        event_cfg_fp.write_text(EVENT_CFGS_YAML)

        out_dir = root / "output"

        cfg = _make_cfg(
            {
                "stage": "shard_events",
                "stage_cfg": {
                    # data_input_dir is a subdir of raw_cohort so .parent points to raw files
                    "data_input_dir": str(raw_dir / "data"),
                    "output_dir": str(out_dir / "data"),
                    "row_chunksize": 10,
                    "infer_schema_length": 10000,
                },
                "event_conversion_config_fp": str(event_cfg_fp),
            }
        )

        shard_stage.main_fn(cfg)

        subjects_shards = list((out_dir / "data" / "subjects").glob("*.parquet"))
        admit_shards = list((out_dir / "data" / "admit_vitals").glob("*.parquet"))
        assert len(subjects_shards) >= 1, f"Expected subject shards, found: {subjects_shards}"
        assert len(admit_shards) >= 1, f"Expected admit_vitals shards, found: {admit_shards}"


# ── Stage 2: split_and_shard_subjects ─────────────────────────────────


def test_split_and_shard_subjects_inprocess():
    from MEDS_extract.split_and_shard_subjects.split_and_shard_subjects import main as split_stage

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        data_dir = root / "data"

        # Write pre-sharded data (as if shard_events already ran)
        subj_dir = data_dir / "subjects"
        subj_dir.mkdir(parents=True)
        pl.read_csv(StringIO(SUBJECTS_CSV)).write_parquet(subj_dir / "[0-6).parquet")

        av_dir = data_dir / "admit_vitals"
        av_dir.mkdir(parents=True)
        pl.read_csv(StringIO(ADMIT_VITALS_CSV)).write_parquet(av_dir / "[0-4).parquet")

        # Write event config
        event_cfg_fp = root / "event_cfgs.yaml"
        event_cfg_fp.write_text(EVENT_CFGS_YAML)

        shards_fp = root / "metadata" / ".shards.json"
        shards_fp.parent.mkdir(parents=True)

        cfg = _make_cfg(
            {
                "stage_cfg": {
                    "data_input_dir": str(data_dir),
                    "output_dir": str(root / "split_output"),
                    "n_subjects_per_shard": 3,
                    "external_splits_json_fp": None,
                    "split_fracs": {"train": 0.5, "tuning": 0.25, "held_out": 0.25},
                },
                "event_conversion_config_fp": str(event_cfg_fp),
                "shards_map_fp": str(shards_fp),
            }
        )

        split_stage.main_fn(cfg)

        assert shards_fp.exists(), "Shards map should be written"
        shards = json.loads(shards_fp.read_text())
        all_subjects = []
        for subjects in shards.values():
            all_subjects.extend(subjects)
        assert len(all_subjects) == 6, f"Expected 6 subjects total, got {len(all_subjects)}"


# ── Stage 3: convert_to_subject_sharded ───────────────────────────────


def test_convert_to_subject_sharded_inprocess():
    from MEDS_extract.convert_to_subject_sharded.convert_to_subject_sharded import (
        main as cts_stage,
    )

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        data_dir = root / "data"

        # Write pre-sharded data
        subj_dir = data_dir / "subjects"
        subj_dir.mkdir(parents=True)
        subjects_df = pl.read_csv(StringIO(SUBJECTS_CSV))
        subjects_df.write_parquet(subj_dir / "[0-6).parquet")

        av_dir = data_dir / "admit_vitals"
        av_dir.mkdir(parents=True)
        admit_df = pl.read_csv(StringIO(ADMIT_VITALS_CSV))
        admit_df.write_parquet(av_dir / "[0-4).parquet")

        # Write event config
        event_cfg_fp = root / "event_cfgs.yaml"
        event_cfg_fp.write_text(EVENT_CFGS_YAML)

        # Simple shards map with just 2 shards
        shards = {"train/0": [239684, 1195293, 68729], "test/0": [814703, 754281, 1500733]}
        shards_fp = root / "metadata" / ".shards.json"
        shards_fp.parent.mkdir(parents=True)
        shards_fp.write_text(json.dumps(shards))

        out_dir = root / "subject_sharded"

        cfg = _make_cfg(
            {
                "stage_cfg": {
                    "data_input_dir": str(data_dir),
                    "output_dir": str(out_dir),
                },
                "event_conversion_config_fp": str(event_cfg_fp),
                "shards_map_fp": str(shards_fp),
            }
        )

        cts_stage.main_fn(cfg)

        # Check output files exist for each shard/prefix combo
        for shard in shards:
            for prefix in ["subjects", "admit_vitals"]:
                fp = out_dir / shard / f"{prefix}.parquet"
                assert fp.exists(), f"Expected {fp} to exist"


# ── Stage 4: convert_to_MEDS_events ──────────────────────────────────


def test_convert_to_MEDS_events_inprocess():
    from MEDS_extract.convert_to_MEDS_events.convert_to_MEDS_events import main as cme_stage

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        # Prepare subject-sharded input (as if convert_to_subject_sharded ran)
        subjects_df = pl.read_csv(StringIO(SUBJECTS_CSV))
        admit_df = pl.read_csv(StringIO(ADMIT_VITALS_CSV))

        shards = {"train/0": [239684, 1195293, 68729]}

        for shard in shards:
            shard_dir = root / "input" / shard
            shard_dir.mkdir(parents=True)
            subjects_df.write_parquet(shard_dir / "subjects.parquet")
            admit_df.write_parquet(shard_dir / "admit_vitals.parquet")

        # Write event config and shards map
        event_cfg_fp = root / "event_cfgs.yaml"
        event_cfg_fp.write_text(EVENT_CFGS_YAML)

        shards_fp = root / "metadata" / ".shards.json"
        shards_fp.parent.mkdir(parents=True)
        shards_fp.write_text(json.dumps(shards))

        out_dir = root / "meds_events"

        cfg = _make_cfg(
            {
                "stage_cfg": {
                    "data_input_dir": str(root / "input"),
                    "output_dir": str(out_dir),
                    "do_dedup_text_and_numeric": False,
                },
                "event_conversion_config_fp": str(event_cfg_fp),
                "shards_map_fp": str(shards_fp),
            }
        )

        cme_stage.main_fn(cfg)

        # Check that output events were created
        for shard in shards:
            subjects_fp = out_dir / shard / "subjects.parquet"
            admit_fp = out_dir / shard / "admit_vitals.parquet"
            assert subjects_fp.exists(), f"Expected {subjects_fp}"
            assert admit_fp.exists(), f"Expected {admit_fp}"

            # Verify MEDS columns exist
            df = pl.read_parquet(subjects_fp)
            assert "subject_id" in df.columns
            assert "code" in df.columns


# ── Stage 5: merge_to_MEDS_cohort ────────────────────────────────────


def test_merge_to_MEDS_cohort_inprocess():
    from MEDS_extract.merge_to_MEDS_cohort.merge_to_MEDS_cohort import main as merge_stage

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        # Create MEDS-format event data per prefix per shard (as if convert_to_MEDS_events ran)
        shards = {"train/0": [239684]}

        meds_data_subjects = pl.DataFrame(
            {
                "subject_id": [239684, 239684],
                "time": [None, None],
                "code": ["EYE_COLOR//BROWN", "HEIGHT"],
                "numeric_value": [None, 175.27],
            }
        ).cast({"subject_id": pl.Int64, "time": pl.Datetime("us"), "numeric_value": pl.Float32})

        meds_data_vitals = pl.DataFrame(
            {
                "subject_id": [239684, 239684],
                "time": ["2010-05-11T17:41:51", "2010-05-11T19:27:19"],
                "code": ["ADMISSION//CARDIAC", "DISCHARGE"],
                "numeric_value": [None, None],
            }
        ).cast({"subject_id": pl.Int64, "time": pl.Datetime("us"), "numeric_value": pl.Float32})

        for shard in shards:
            shard_dir = root / "input" / shard
            shard_dir.mkdir(parents=True)
            meds_data_subjects.write_parquet(shard_dir / "subjects.parquet")
            meds_data_vitals.write_parquet(shard_dir / "admit_vitals.parquet")

        event_cfg_fp = root / "event_cfgs.yaml"
        event_cfg_fp.write_text(EVENT_CFGS_YAML)

        shards_fp = root / "metadata" / ".shards.json"
        shards_fp.parent.mkdir(parents=True)
        shards_fp.write_text(json.dumps(shards))

        out_dir = root / "merged"

        cfg = _make_cfg(
            {
                "stage_cfg": {
                    "data_input_dir": str(root / "input"),
                    "output_dir": str(out_dir),
                    "unique_by": "*",
                    "additional_sort_by": None,
                },
                "event_conversion_config_fp": str(event_cfg_fp),
                "shards_map_fp": str(shards_fp),
            }
        )

        merge_stage.main_fn(cfg)

        merged_fp = out_dir / "train/0.parquet"
        assert merged_fp.exists(), f"Expected merged file at {merged_fp}"
        df = pl.read_parquet(merged_fp)
        assert len(df) == 4, f"Expected 4 merged events, got {len(df)}"


# ── Stage 6: extract_code_metadata ───────────────────────────────────

METADATA_EVENT_CFGS_YAML = """\
subject_id_col: subject_id
subjects:
  subject_id_col: MRN
  eye_color:
    code: 'f"EYE_COLOR//{$eye_color}"'
    time: null
    _metadata:
      demo_metadata:
        description: description
admit_vitals:
  HR:
    code: '"HR"'
    time: '$vitals_date::"%m/%d/%Y, %H:%M:%S"'
    numeric_value: "$HR"
    _metadata:
      lab_metadata:
        description: title
"""

DEMO_METADATA_CSV = """\
eye_color,description
BROWN,Brown eyes
BLUE,Blue eyes
HAZEL,Hazel eyes
"""

LAB_METADATA_CSV = """\
lab_code,title
HR,Heart Rate
TEMP,Body Temperature
"""


def test_extract_code_metadata_inprocess():
    from MEDS_extract.extract_code_metadata.extract_code_metadata import main as ecm_stage

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        # Create MEDS event data (as if merge ran)
        events_dir = root / "events"
        events_df = pl.DataFrame(
            {
                "subject_id": [239684, 239684, 239684],
                "time": [None, "2010-05-11T17:41:51", "2010-05-11T17:41:51"],
                "code": ["EYE_COLOR//BROWN", "HR", "TEMP"],
                "numeric_value": [None, 102.6, 96.0],
            }
        ).cast({"subject_id": pl.Int64, "time": pl.Datetime("us"), "numeric_value": pl.Float32})
        shard_dir = events_dir / "train" / "0"
        shard_dir.mkdir(parents=True)
        events_df.write_parquet(shard_dir / "data.parquet")

        # Write raw metadata CSVs
        raw_dir = root / "raw"
        raw_dir.mkdir()
        (raw_dir / "demo_metadata.csv").write_text(DEMO_METADATA_CSV)
        (raw_dir / "lab_metadata.csv").write_text(LAB_METADATA_CSV)

        event_cfg_fp = root / "event_cfgs.yaml"
        event_cfg_fp.write_text(METADATA_EVENT_CFGS_YAML)

        shards_fp = root / "metadata" / ".shards.json"
        shards_fp.parent.mkdir(parents=True)
        shards_fp.write_text(json.dumps({"train/0": [239684]}))

        metadata_input_dir = root / "metadata_in" / "metadata"
        metadata_input_dir.mkdir(parents=True)

        out_dir = root / "metadata_out" / "metadata"
        out_dir.mkdir(parents=True)

        cfg = _make_cfg(
            {
                "input_dir": str(raw_dir),
                "stage_cfg": {
                    "data_input_dir": str(events_dir),
                    "output_dir": str(out_dir),
                    "metadata_input_dir": str(metadata_input_dir),
                    "reducer_output_dir": str(out_dir),
                    "description_separator": "\n",
                },
                "event_conversion_config_fp": str(event_cfg_fp),
                "shards_map_fp": str(shards_fp),
            }
        )

        ecm_stage.main_fn(cfg)

        codes_fp = out_dir / "codes.parquet"
        assert codes_fp.exists(), f"Expected codes.parquet at {codes_fp}"
        codes_df = pl.read_parquet(codes_fp)
        assert "code" in codes_df.columns
        assert "description" in codes_df.columns


# ── Stage 7: finalize_MEDS_metadata ──────────────────────────────────


def test_finalize_MEDS_metadata_inprocess():
    from MEDS_extract.finalize_MEDS_metadata.finalize_MEDS_metadata import main as fmm_stage

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        # Write input code metadata
        metadata_in = root / "metadata_in" / "metadata"
        metadata_in.mkdir(parents=True)

        codes_df = pl.DataFrame(
            {
                "code": ["EYE_COLOR//BROWN", "HR"],
                "description": ["Brown eyes", "Heart Rate"],
                "parent_codes": [None, ["LOINC/8867-4"]],
            }
        )
        codes_df.write_parquet(metadata_in / "codes.parquet", use_pyarrow=True)

        # Write shards map
        shards = {"train/0": [239684, 1195293], "test/0": [68729]}
        shards_fp = root / "metadata" / ".shards.json"
        shards_fp.parent.mkdir(parents=True)
        shards_fp.write_text(json.dumps(shards))

        out_dir = root / "output" / "metadata"
        out_dir.mkdir(parents=True)

        cfg = _make_cfg(
            {
                "stage_cfg": {
                    "metadata_input_dir": str(metadata_in),
                    "reducer_output_dir": str(out_dir),
                },
                "shards_map_fp": str(shards_fp),
            }
        )

        fmm_stage.main_fn(cfg)

        # Check all three output files
        # finalize does: output_metadata_dir.parent / <meds_filepath>
        # where meds filepaths are "metadata/codes.parquet", "metadata/dataset.json", etc.
        code_fp = root / "output" / "metadata" / "codes.parquet"
        dataset_fp = root / "output" / "metadata" / "dataset.json"
        splits_fp = root / "output" / "metadata" / "subject_splits.parquet"

        assert code_fp.exists(), f"Expected {code_fp}"
        assert dataset_fp.exists(), f"Expected {dataset_fp}"
        assert splits_fp.exists(), f"Expected {splits_fp}"

        # Validate dataset metadata
        dataset_meta = json.loads(dataset_fp.read_text())
        assert dataset_meta["dataset_name"] == "TEST"
        assert dataset_meta["dataset_version"] == "1.0"

        # Validate subject splits
        splits_tbl = pq.read_table(splits_fp)
        splits_df = pl.from_arrow(splits_tbl)
        assert set(splits_df.columns) >= {"subject_id", "split"}
        assert len(splits_df) == 3


def test_finalize_MEDS_metadata_no_existing_codes():
    """Tests the branch where no input codes.parquet exists."""
    from MEDS_extract.finalize_MEDS_metadata.finalize_MEDS_metadata import main as fmm_stage

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        metadata_in = root / "metadata_in" / "metadata"
        metadata_in.mkdir(parents=True)
        # Don't create codes.parquet — test the empty branch

        shards = {"train/0": [1, 2]}
        shards_fp = root / "metadata" / ".shards.json"
        shards_fp.parent.mkdir(parents=True)
        shards_fp.write_text(json.dumps(shards))

        out_dir = root / "output" / "metadata"
        out_dir.mkdir(parents=True)

        cfg = _make_cfg(
            {
                "stage_cfg": {
                    "metadata_input_dir": str(metadata_in),
                    "reducer_output_dir": str(out_dir),
                },
                "shards_map_fp": str(shards_fp),
            }
        )

        fmm_stage.main_fn(cfg)

        code_fp = root / "output" / "metadata" / "codes.parquet"
        assert code_fp.exists()
        # Should be empty but schema-compliant
        tbl = pq.read_table(code_fp)
        assert tbl.num_rows == 0


def test_finalize_MEDS_metadata_overwrite_error():
    """Tests that existing output files raise FileExistsError when do_overwrite is False."""
    import pytest

    from MEDS_extract.finalize_MEDS_metadata.finalize_MEDS_metadata import main as fmm_stage

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        metadata_in = root / "metadata_in" / "metadata"
        metadata_in.mkdir(parents=True)

        shards = {"train/0": [1]}
        shards_fp = root / "metadata" / ".shards.json"
        shards_fp.parent.mkdir(parents=True)
        shards_fp.write_text(json.dumps(shards))

        out_dir = root / "output" / "metadata"
        out_dir.mkdir(parents=True)

        # Pre-create output file so it conflicts
        (root / "output" / "metadata" / "codes.parquet").write_bytes(b"dummy")

        cfg = _make_cfg(
            {
                "do_overwrite": False,
                "stage_cfg": {
                    "metadata_input_dir": str(metadata_in),
                    "reducer_output_dir": str(out_dir),
                },
                "shards_map_fp": str(shards_fp),
            }
        )

        with pytest.raises(FileExistsError):
            fmm_stage.main_fn(cfg)


# ── Stage 8: finalize_MEDS_data ──────────────────────────────────────


def test_finalize_MEDS_data_inprocess():
    from MEDS_extract.finalize_MEDS_data.finalize_MEDS_data import finalize_MEDS_data

    df = pl.DataFrame(
        {
            "subject_id": [239684, 239684],
            "time": [None, "2010-05-11T17:41:51"],
            "code": ["EYE_COLOR//BROWN", "HR"],
            "numeric_value": [None, 102.6],
        }
    ).cast({"subject_id": pl.Int64, "time": pl.Datetime("us"), "numeric_value": pl.Float32})

    result = finalize_MEDS_data.map_fn(df.lazy())

    # Should return a PyArrow table with MEDS-compliant schema
    import pyarrow as pa

    assert isinstance(result, pa.Table)
    assert result.num_rows == 2
    schema_names = set(result.schema.names)
    assert {"subject_id", "time", "code", "numeric_value"} <= schema_names


# ── Additional coverage: extract_code_metadata edge cases ────────────


def test_extract_code_metadata_no_metadata_blocks():
    """Tests the early return when no _metadata blocks are present."""
    from MEDS_extract.extract_code_metadata.extract_code_metadata import main as ecm_stage

    no_metadata_cfg = """\
subject_id_col: subject_id
subjects:
  subject_id_col: MRN
  eye_color:
    code: 'f"EYE_COLOR//{$eye_color}"'
    time: null
"""

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        events_dir = root / "events" / "train" / "0"
        events_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "subject_id": [1],
                "time": [None],
                "code": ["X"],
                "numeric_value": [None],
            }
        ).cast(
            {"subject_id": pl.Int64, "time": pl.Datetime("us"), "numeric_value": pl.Float32}
        ).write_parquet(events_dir / "data.parquet")

        event_cfg_fp = root / "event_cfgs.yaml"
        event_cfg_fp.write_text(no_metadata_cfg)

        shards_fp = root / "metadata" / ".shards.json"
        shards_fp.parent.mkdir(parents=True)
        shards_fp.write_text(json.dumps({"train/0": [1]}))

        out_dir = root / "metadata_out" / "metadata"
        out_dir.mkdir(parents=True)

        cfg = _make_cfg(
            {
                "input_dir": str(root / "raw"),
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

        # Should return early without error
        ecm_stage.main_fn(cfg)


# ── Additional coverage: split edge cases ────────────────────────────


def test_shard_subjects_null_split_fracs():
    """Tests shard_subjects when some split fractions are None."""
    import numpy as np

    from MEDS_extract.split_and_shard_subjects.split_and_shard_subjects import shard_subjects

    subjects = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = shard_subjects(
        subjects=subjects,
        split_fracs_dict={"train": 0.8, "test": 0.2, "unused": None},
        n_subjects_per_shard=100,
        seed=42,
    )

    all_ids = []
    for ids in result.values():
        all_ids.extend(ids)
    assert len(all_ids) == 10


def test_shard_subjects_external_splits_cover_all():
    """Tests shard_subjects when external splits cover all subjects."""
    import numpy as np

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


def test_shard_subjects_external_with_nonempty_fracs():
    """Tests the warning when external splits cover all subjects but split_fracs is nonempty."""
    import numpy as np

    from MEDS_extract.split_and_shard_subjects.split_and_shard_subjects import shard_subjects

    subjects = np.array([1, 2, 3, 4])
    result = shard_subjects(
        subjects=subjects,
        external_splits={"train": np.array([1, 2, 3]), "test": np.array([4])},
        split_fracs_dict={"train": 0.8, "test": 0.2},
        n_subjects_per_shard=100,
        seed=42,
    )

    all_ids = []
    for ids in result.values():
        all_ids.extend(ids)
    assert len(all_ids) == 4


# ── Additional coverage: shard_events missing config error ───────────


def test_shard_events_missing_config():
    import pytest

    from MEDS_extract.shard_events.shard_events import main as shard_stage

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        cfg = _make_cfg(
            {
                "stage_cfg": {
                    "data_input_dir": str(root / "data"),
                    "output_dir": str(root / "output"),
                    "row_chunksize": 10,
                    "infer_schema_length": 10000,
                },
                "event_conversion_config_fp": str(root / "nonexistent.yaml"),
                "cloud_io_storage_options": {},
            }
        )

        with pytest.raises(FileNotFoundError):
            shard_stage.main_fn(cfg)


def test_convert_to_MEDS_events_missing_config():
    import pytest

    from MEDS_extract.convert_to_MEDS_events.convert_to_MEDS_events import main as cme_stage

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        shards_fp = root / ".shards.json"
        shards_fp.write_text(json.dumps({"train/0": [1]}))

        cfg = _make_cfg(
            {
                "stage_cfg": {
                    "data_input_dir": str(root),
                    "output_dir": str(root / "output"),
                },
                "event_conversion_config_fp": str(root / "nonexistent.yaml"),
                "shards_map_fp": str(shards_fp),
                "cloud_io_storage_options": {},
            }
        )

        with pytest.raises(FileNotFoundError):
            cme_stage.main_fn(cfg)


def test_convert_to_subject_sharded_missing_config():
    import pytest

    from MEDS_extract.convert_to_subject_sharded.convert_to_subject_sharded import main as cts_stage

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        shards_fp = root / ".shards.json"
        shards_fp.write_text(json.dumps({"train/0": [1]}))

        cfg = _make_cfg(
            {
                "stage_cfg": {
                    "data_input_dir": str(root),
                    "output_dir": str(root / "output"),
                },
                "event_conversion_config_fp": str(root / "nonexistent.yaml"),
                "shards_map_fp": str(shards_fp),
            }
        )

        with pytest.raises(FileNotFoundError):
            cts_stage.main_fn(cfg)


# ── Additional coverage: extract_code_metadata utils ─────────────────


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


def test_finalize_MEDS_metadata_output_dir_validation():
    """Tests that finalize_MEDS_metadata validates the output dir ends in 'metadata'."""
    import pytest

    from MEDS_extract.finalize_MEDS_metadata.finalize_MEDS_metadata import main as fmm_stage

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        metadata_in = root / "metadata_in" / "metadata"
        metadata_in.mkdir(parents=True)

        shards_fp = root / "metadata" / ".shards.json"
        shards_fp.parent.mkdir(parents=True)
        shards_fp.write_text(json.dumps({"train/0": [1]}))

        # Wrong output dir name (not ending in 'metadata')
        out_dir = root / "output" / "wrong_name"
        out_dir.mkdir(parents=True)

        cfg = _make_cfg(
            {
                "stage_cfg": {
                    "metadata_input_dir": str(metadata_in),
                    "reducer_output_dir": str(out_dir),
                },
                "shards_map_fp": str(shards_fp),
            }
        )

        with pytest.raises(ValueError, match="metadata"):
            fmm_stage.main_fn(cfg)


# ── Coverage: convert_to_MEDS_events with subject_id_expr and transforms ──


def test_convert_to_MEDS_events_subject_id_expr():
    """Tests the subject_id_expr path in convert_to_MEDS_events main()."""
    from MEDS_extract.convert_to_MEDS_events.convert_to_MEDS_events import main as cme_stage

    event_cfg_with_expr = """\
subjects:
  subject_id_expr: "hash($MRN)"
  eye_color:
    code: 'f"EYE_COLOR//{$eye_color}"'
    time: null
"""

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        subjects_df = pl.DataFrame({"MRN": ["ABC", "DEF"], "eye_color": ["BROWN", "BLUE"]})

        shards = {"train/0": [1]}  # placeholder - subject_id_expr will hash
        shard_dir = root / "input" / "train" / "0"
        shard_dir.mkdir(parents=True)
        subjects_df.write_parquet(shard_dir / "subjects.parquet")

        event_cfg_fp = root / "event_cfgs.yaml"
        event_cfg_fp.write_text(event_cfg_with_expr)

        shards_fp = root / ".shards.json"
        shards_fp.write_text(json.dumps(shards))

        out_dir = root / "output"

        cfg = _make_cfg(
            {
                "stage_cfg": {
                    "data_input_dir": str(root / "input"),
                    "output_dir": str(out_dir),
                    "do_dedup_text_and_numeric": False,
                },
                "event_conversion_config_fp": str(event_cfg_fp),
                "shards_map_fp": str(shards_fp),
            }
        )

        cme_stage.main_fn(cfg)

        out_fp = out_dir / "train" / "0" / "subjects.parquet"
        assert out_fp.exists()
        df = pl.read_parquet(out_fp)
        assert "subject_id" in df.columns
        assert df["subject_id"].dtype == pl.Int64


def test_convert_to_MEDS_events_with_transforms():
    """Tests the transforms path in convert_to_MEDS_events main()."""
    from MEDS_extract.convert_to_MEDS_events.convert_to_MEDS_events import main as cme_stage

    event_cfg_with_transforms = """\
subject_id_col: subject_id
data:
  transforms:
    doubled: "$value * 2"
  measurement:
    code: '"MEAS"'
    time: null
    numeric_value: "$doubled"
"""

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        data_df = pl.DataFrame({"subject_id": [1, 2], "value": [10.0, 20.0]})

        shards = {"train/0": [1, 2]}
        shard_dir = root / "input" / "train" / "0"
        shard_dir.mkdir(parents=True)
        data_df.write_parquet(shard_dir / "data.parquet")

        event_cfg_fp = root / "event_cfgs.yaml"
        event_cfg_fp.write_text(event_cfg_with_transforms)

        shards_fp = root / ".shards.json"
        shards_fp.write_text(json.dumps(shards))

        out_dir = root / "output"

        cfg = _make_cfg(
            {
                "stage_cfg": {
                    "data_input_dir": str(root / "input"),
                    "output_dir": str(out_dir),
                    "do_dedup_text_and_numeric": False,
                },
                "event_conversion_config_fp": str(event_cfg_fp),
                "shards_map_fp": str(shards_fp),
            }
        )

        cme_stage.main_fn(cfg)

        out_fp = out_dir / "train" / "0" / "data.parquet"
        assert out_fp.exists()
        df = pl.read_parquet(out_fp)
        assert "numeric_value" in df.columns
        # Values should be doubled: 10*2=20, 20*2=40
        vals = sorted(df["numeric_value"].drop_nulls().to_list())
        assert vals == [20.0, 40.0]


def test_convert_to_MEDS_events_file_globbing():
    """Tests the file globbing path when exact file doesn't exist but a glob match does."""
    from MEDS_extract.convert_to_MEDS_events.convert_to_MEDS_events import main as cme_stage

    simple_cfg = """\
subject_id_col: subject_id
data:
  measurement:
    code: '"MEAS"'
    time: null
"""

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        data_df = pl.DataFrame({"subject_id": [1]})

        shards = {"train/0": [1]}
        shard_dir = root / "input" / "train" / "0"
        shard_dir.mkdir(parents=True)
        # Name the file with a suffix so it doesn't match exactly but matches the glob
        data_df.write_parquet(shard_dir / "data_shard1.parquet")

        event_cfg_fp = root / "event_cfgs.yaml"
        event_cfg_fp.write_text(simple_cfg)

        shards_fp = root / ".shards.json"
        shards_fp.write_text(json.dumps(shards))

        out_dir = root / "output"

        cfg = _make_cfg(
            {
                "stage_cfg": {
                    "data_input_dir": str(root / "input"),
                    "output_dir": str(out_dir),
                    "do_dedup_text_and_numeric": False,
                },
                "event_conversion_config_fp": str(event_cfg_fp),
                "shards_map_fp": str(shards_fp),
            }
        )

        cme_stage.main_fn(cfg)

        out_fp = out_dir / "train" / "0" / "data.parquet"
        assert out_fp.exists()


# ── Coverage: convert_to_subject_sharded with join ───────────────────


def test_convert_to_subject_sharded_with_join():
    """Tests convert_to_subject_sharded with a join configuration."""
    from MEDS_extract.convert_to_subject_sharded.convert_to_subject_sharded import main as cts_stage

    join_cfg = """\
vitals:
  join:
    input_prefix: stays
    left_on: stay_id
    right_on: stay_id
    columns_from_right:
      - subject_id
  subject_id_col: subject_id
  HR:
    code: '"HR"'
    time: null
    numeric_value: "$HR"
stays:
  subject_id_col: subject_id
"""

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        data_dir = root / "data"

        vitals_df = pl.DataFrame({"stay_id": [10, 20], "HR": [70.0, 65.0]})
        stays_df = pl.DataFrame({"stay_id": [10, 20], "subject_id": [111, 222]})

        vitals_dir = data_dir / "vitals"
        vitals_dir.mkdir(parents=True)
        vitals_df.write_parquet(vitals_dir / "[0-2).parquet")

        stays_dir = data_dir / "stays"
        stays_dir.mkdir(parents=True)
        stays_df.write_parquet(stays_dir / "[0-2).parquet")

        event_cfg_fp = root / "event_cfg.yaml"
        event_cfg_fp.write_text(join_cfg)

        shards = {"train/0": [111, 222]}
        shards_fp = root / ".shards.json"
        shards_fp.write_text(json.dumps(shards))

        out_dir = root / "output"

        cfg = _make_cfg(
            {
                "stage_cfg": {
                    "data_input_dir": str(data_dir),
                    "output_dir": str(out_dir),
                },
                "event_conversion_config_fp": str(event_cfg_fp),
                "shards_map_fp": str(shards_fp),
            }
        )

        cts_stage.main_fn(cfg)

        vitals_out = out_dir / "train" / "0" / "vitals.parquet"
        assert vitals_out.exists()
        df = pl.read_parquet(vitals_out)
        assert "subject_id" in df.columns


# ── Coverage: split_and_shard_subjects with join and external_splits_json_fp ──


def test_split_and_shard_subjects_with_join():
    """Tests split_and_shard_subjects when the event config includes a join."""
    from MEDS_extract.split_and_shard_subjects.split_and_shard_subjects import main as split_stage

    join_cfg = """\
vitals:
  join:
    input_prefix: stays
    left_on: stay_id
    right_on: stay_id
    columns_from_right:
      - subject_id
  subject_id_col: subject_id
  HR:
    code: '"HR"'
    time: null
stays:
  subject_id_col: subject_id
"""

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        data_dir = root / "data"

        vitals_df = pl.DataFrame({"stay_id": [10, 20], "HR": [70.0, 65.0]})
        stays_df = pl.DataFrame({"stay_id": [10, 20], "subject_id": [111, 222]})

        vitals_dir = data_dir / "vitals"
        vitals_dir.mkdir(parents=True)
        vitals_df.write_parquet(vitals_dir / "[0-2).parquet")

        stays_dir = data_dir / "stays"
        stays_dir.mkdir(parents=True)
        stays_df.write_parquet(stays_dir / "[0-2).parquet")

        event_cfg_fp = root / "event_cfg.yaml"
        event_cfg_fp.write_text(join_cfg)

        shards_fp = root / "metadata" / ".shards.json"
        shards_fp.parent.mkdir(parents=True)

        cfg = _make_cfg(
            {
                "stage_cfg": {
                    "data_input_dir": str(data_dir),
                    "output_dir": str(root / "split_output"),
                    "n_subjects_per_shard": 100,
                    "external_splits_json_fp": None,
                    "split_fracs": {"train": 0.5, "test": 0.5},
                },
                "event_conversion_config_fp": str(event_cfg_fp),
                "shards_map_fp": str(shards_fp),
            }
        )

        split_stage.main_fn(cfg)

        assert shards_fp.exists()
        shards = json.loads(shards_fp.read_text())
        all_subjects = []
        for subjects in shards.values():
            all_subjects.extend(subjects)
        assert set(all_subjects) == {111, 222}


def test_split_and_shard_subjects_with_external_file():
    """Tests split_and_shard_subjects with external_splits_json_fp."""
    from MEDS_extract.split_and_shard_subjects.split_and_shard_subjects import main as split_stage

    simple_cfg = """\
subject_id_col: subject_id
data:
  event:
    code: '"X"'
    time: null
"""

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        data_dir = root / "data"
        data_dir.mkdir()

        data_sub = data_dir / "data"
        data_sub.mkdir()
        pl.DataFrame({"subject_id": [1, 2, 3, 4]}).write_parquet(data_sub / "[0-4).parquet")

        event_cfg_fp = root / "event_cfg.yaml"
        event_cfg_fp.write_text(simple_cfg)

        external_splits = {"train": [1, 2], "test": [3, 4]}
        ext_fp = root / "external_splits.json"
        ext_fp.write_text(json.dumps(external_splits))

        shards_fp = root / "metadata" / ".shards.json"
        shards_fp.parent.mkdir(parents=True)

        cfg = _make_cfg(
            {
                "stage_cfg": {
                    "data_input_dir": str(data_dir),
                    "output_dir": str(root / "output"),
                    "n_subjects_per_shard": 100,
                    "external_splits_json_fp": str(ext_fp),
                    "split_fracs": {},
                },
                "event_conversion_config_fp": str(event_cfg_fp),
                "shards_map_fp": str(shards_fp),
            }
        )

        split_stage.main_fn(cfg)

        shards = json.loads(shards_fp.read_text())
        all_subjects = []
        for subjects in shards.values():
            all_subjects.extend(subjects)
        assert set(all_subjects) == {1, 2, 3, 4}


def test_shard_subjects_external_splits_list_conversion():
    """Tests that non-numpy external splits are converted to numpy arrays."""
    import numpy as np

    from MEDS_extract.split_and_shard_subjects.split_and_shard_subjects import shard_subjects

    subjects = np.array([1, 2, 3, 4])
    # Pass lists instead of numpy arrays
    result = shard_subjects(
        subjects=subjects,
        external_splits={"train": [1, 2, 3], "test": [4]},
        split_fracs_dict={},
        n_subjects_per_shard=100,
        seed=42,
    )

    all_ids = []
    for ids in result.values():
        all_ids.extend(ids)
    assert set(all_ids) == {1, 2, 3, 4}


# ── Coverage: extract_code_metadata with existing codes and parent_codes ──


def test_extract_code_metadata_with_existing_codes():
    """Tests extract_code_metadata when an existing codes.parquet is present for joining."""
    from MEDS_extract.extract_code_metadata.extract_code_metadata import main as ecm_stage

    metadata_cfg = """\
subject_id_col: subject_id
data:
  measurement:
    code: '$lab_code'
    _metadata:
      lab_meta:
        description: title
"""

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        # Create events with codes
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

        # Write raw metadata
        raw_dir = root / "raw"
        raw_dir.mkdir()
        (raw_dir / "lab_meta.csv").write_text(
            "lab_code,title,loinc\nHR,Heart Rate,8867-4\nTEMP,Temperature,8310-5\n"
        )

        # Write existing codes.parquet
        metadata_in = root / "metadata_in" / "metadata"
        metadata_in.mkdir(parents=True)
        existing_codes = pl.DataFrame(
            {
                "code": ["EXISTING_CODE"],
                "description": ["An existing code"],
            }
        )
        existing_codes.write_parquet(metadata_in / "codes.parquet", use_pyarrow=True)

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

        codes_fp = out_dir / "codes.parquet"
        assert codes_fp.exists()
        codes_df = pl.read_parquet(codes_fp)
        # Should contain both existing and new codes
        all_codes = set(codes_df["code"].to_list())
        assert "EXISTING_CODE" in all_codes
        assert "HR" in all_codes or "TEMP" in all_codes


def test_extract_metadata_missing_column_error():
    """Tests that extract_metadata raises KeyError for missing columns."""
    import pytest

    from MEDS_extract.extract_code_metadata.extract_code_metadata import extract_metadata

    metadata_df = pl.DataFrame({"code": ["A"], "name": ["Code A"]}).lazy()
    event_cfg = {
        "code": "$code",
        "_metadata": {"desc": "nonexistent_column"},
    }

    with pytest.raises(KeyError, match="nonexistent_column"):
        extract_metadata(metadata_df, event_cfg)


# ── Coverage: shard_events duplicate file and not-in-config warnings ──


def test_shard_events_skips_unconfigured_files():
    """Tests that shard_events skips files not in the event config."""
    from MEDS_extract.shard_events.shard_events import main as shard_stage

    minimal_cfg = """\
subject_id_col: subject_id
data:
  event:
    code: '"X"'
    time: null
"""

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        raw_dir = root / "raw_cohort"
        raw_dir.mkdir()

        # Write the configured file
        pl.DataFrame({"subject_id": [1]}).write_parquet(raw_dir / "data.parquet")
        # Write an extra file that isn't in the config
        pl.DataFrame({"a": [1]}).write_parquet(raw_dir / "extra.parquet")

        event_cfg_fp = root / "event_cfgs.yaml"
        event_cfg_fp.write_text(minimal_cfg)

        out_dir = root / "output"

        cfg = _make_cfg(
            {
                "stage": "shard_events",
                "stage_cfg": {
                    "data_input_dir": str(raw_dir / "data"),
                    "output_dir": str(out_dir / "data"),
                    "row_chunksize": 100,
                    "infer_schema_length": 10000,
                },
                "event_conversion_config_fp": str(event_cfg_fp),
            }
        )

        shard_stage.main_fn(cfg)

        # Only data should be sharded, not extra
        assert (out_dir / "data" / "data").exists()
        assert not (out_dir / "data" / "extra").exists()


def test_shard_events_prefers_parquet_over_csv():
    """Tests that when both .parquet and .csv exist, parquet is preferred and csv is skipped."""
    from MEDS_extract.shard_events.shard_events import main as shard_stage

    minimal_cfg = """\
subject_id_col: subject_id
data:
  event:
    code: '"X"'
    time: null
"""

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        raw_dir = root / "raw_cohort"
        raw_dir.mkdir()

        df = pl.DataFrame({"subject_id": [1, 2, 3]})
        df.write_parquet(raw_dir / "data.parquet")
        df.write_csv(raw_dir / "data.csv")

        event_cfg_fp = root / "event_cfgs.yaml"
        event_cfg_fp.write_text(minimal_cfg)

        out_dir = root / "output"

        cfg = _make_cfg(
            {
                "stage": "shard_events",
                "stage_cfg": {
                    "data_input_dir": str(raw_dir / "data"),
                    "output_dir": str(out_dir / "data"),
                    "row_chunksize": 100,
                    "infer_schema_length": 10000,
                },
                "event_conversion_config_fp": str(event_cfg_fp),
            }
        )

        shard_stage.main_fn(cfg)

        shards = list((out_dir / "data" / "data").glob("*.parquet"))
        assert len(shards) >= 1
