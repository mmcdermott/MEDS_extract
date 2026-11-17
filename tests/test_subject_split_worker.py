"""Only worker zero should scan subjects and publish the shared shard map."""

import json
from pathlib import Path

import polars as pl
import pytest
from omegaconf import OmegaConf

from MEDS_extract.config import TableConfig
from MEDS_extract.split_and_shard_subjects.split_and_shard_subjects import main


@pytest.mark.parametrize("do_overwrite", [False, True])
def test_multiple_worker_invocations_scan_and_write_once(tmp_path, monkeypatch, do_overwrite):
    source = tmp_path / "input"
    source.mkdir()
    pl.DataFrame({"subject_id": range(1, 11)}).write_parquet(source / "patients.parquet")
    spec = tmp_path / "messy.yaml"
    spec.write_text("patients:\n  birth: {code: BIRTH, time: null}\n")
    shards_file = tmp_path / "metadata" / ".shards.json"
    cfg = OmegaConf.create(
        {
            "worker": 0,
            "seed": 17,
            "do_overwrite": do_overwrite,
            "MESSY_config_fp": str(spec),
            "shards_map_fp": str(shards_file),
            "stage_cfg": {
                "data_input_dir": str(source),
                "external_splits_json_fp": None,
                "n_subjects_per_shard": 4,
                "split_fracs": {"train": 0.8, "tuning": 0.1, "held_out": 0.1},
            },
        }
    )
    calls = {"scans": 0, "writes": 0}
    original_scan = TableConfig.scan
    original_write = Path.write_text

    def scan(table, *args, **kwargs):
        calls["scans"] += 1
        return original_scan(table, *args, **kwargs)

    def write(path, *args, **kwargs):
        if path == shards_file:
            calls["writes"] += 1
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(TableConfig, "scan", scan)
    monkeypatch.setattr(Path, "write_text", write)
    # Controlled invocations emulate a stage multirun; worker zero need not start first.
    for worker in [3, 1, 0, 2]:
        cfg.worker = worker
        main.main_fn(cfg)

    assert calls == {"scans": 1, "writes": 1}
    shards = json.loads(shards_file.read_text())
    assert sorted(subject for ids in shards.values() for subject in ids) == list(range(1, 11))
    assert {name: len(ids) for name, ids in shards.items()} == {
        "train/0": 4,
        "train/1": 4,
        "tuning/0": 1,
        "held_out/0": 1,
    }
