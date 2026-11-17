"""Finalization must reject shard maps that duplicate subjects."""

import json

import polars as pl
import pytest
from omegaconf import OmegaConf

from MEDS_extract.finalize_MEDS_metadata.finalize_MEDS_metadata import main


def finalize_config(tmp_path, shards):
    shard_file = tmp_path / "shards.json"
    shard_file.write_text(json.dumps(shards))
    return OmegaConf.create(
        {
            "worker": 0,
            "shards_map_fp": str(shard_file),
            "stage_cfg": {
                "metadata_input_dir": str(tmp_path / "input"),
                "reducer_output_dir": str(tmp_path / "metadata"),
            },
            "etl_metadata": {
                "dataset_name": "synthetic",
                "dataset_version": "1",
                "package_name": "synthetic-etl",
                "package_version": "1",
            },
        }
    )


@pytest.mark.parametrize(
    "shards",
    [
        {"taskA/held_out/0": [1, 2], "taskB/held_out/0": [2, 3]},
        {"train/0": [1, 2], "train/1": [2, 3]},
        {"train/0": [1, 1]},
    ],
)
def test_duplicate_subjects_fail_before_replacing_metadata(tmp_path, shards):
    cfg = finalize_config(tmp_path, shards)
    output = tmp_path / "metadata"
    output.mkdir()
    files = [output / name for name in ["codes.parquet", "dataset.json", "subject_splits.parquet"]]
    for path in files:
        path.write_bytes(b"prior output")
    with pytest.raises(ValueError, match="exactly one shard") as error:
        main.main_fn(cfg)
    assert all(shard in str(error.value) for shard in shards)
    assert all(path.read_bytes() == b"prior output" for path in files)


def test_disjoint_shards_keep_one_split_row_per_subject(tmp_path):
    main.main_fn(finalize_config(tmp_path, {"train/0": [1, 2], "task/held_out/0": [3]}))
    splits = pl.read_parquet(tmp_path / "metadata" / "subject_splits.parquet")
    assert splits.to_dicts() == [
        {"subject_id": 1, "split": "train"},
        {"subject_id": 2, "split": "train"},
        {"subject_id": 3, "split": "task/held_out"},
    ]
    assert splits.schema == {"subject_id": pl.Int64, "split": pl.String}
