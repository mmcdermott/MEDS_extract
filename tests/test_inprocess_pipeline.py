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
import logging
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

        event_cfg_fp = root / "messy.yaml"
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
                "MESSY_config_fp": str(event_cfg_fp),
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

        event_cfg_fp = root / "messy.yaml"
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
                "MESSY_config_fp": str(event_cfg_fp),
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

        event_cfg_fp = root / "messy.yaml"
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
                "MESSY_config_fp": str(event_cfg_fp),
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


# ── io.scan_source: the .csv.gz read path (MIMIC's native raw format) ──


def test_scan_source_csv_gz_with_shard_events_kwargs(tmp_path):
    """A ``.csv.gz`` source scans through ``gzip.open`` + ``read_csv`` with shard_events' kwargs.

    ``shard_events`` passes ``row_index_name`` and ``infer_schema_length=None``
    (full-file schema inference) to every raw scan regardless of format; the gzip
    branch must honor both — a row-index column is prepended and the schema is
    inferred over all rows (not all-String).
    """
    import gzip

    from MEDS_extract.io import scan_source

    fp = tmp_path / "labs.csv.gz"
    with gzip.open(fp, mode="wt") as f:
        f.write("subject_id,test_name,result\n1,HR,80\n2,TEMP,36.6\n")

    df = scan_source(fp, row_index_name="__row_idx__", infer_schema_length=None).collect()
    assert df.columns == ["__row_idx__", "subject_id", "test_name", "result"]
    assert df["__row_idx__"].to_list() == [0, 1]
    assert df.schema["subject_id"] == pl.Int64  # schema inferred, not all-String
    assert df.schema["result"] == pl.Float64
    assert df["test_name"].to_list() == ["HR", "TEMP"]


# ── convert_to_parquet: raw-table normalization ──


def test_convert_to_parquet_skips_unconfigured_files():
    """Files in the raw input that no event config references are never converted."""
    from MEDS_extract.convert_to_parquet.convert_to_parquet import main as convert_stage

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

        messy_fp = root / "messy.yaml"
        messy_fp.write_text(minimal_cfg)

        cfg = _make_cfg(
            {
                "stage": "convert_to_parquet",
                "input_dir": str(raw_dir),
                "stage_cfg": {
                    "data_input_dir": str(raw_dir / "data"),
                    "output_dir": str(root / "output" / "data"),
                },
                "MESSY_config_fp": str(messy_fp),
            }
        )
        convert_stage.main_fn(cfg)

        assert (root / "output" / "data" / "data.parquet").is_file()
        assert not (root / "output" / "data" / "extra.parquet").exists()


def test_convert_to_parquet_csv_gz_source(tmp_path):
    """End-to-end over a raw ``.csv.gz`` (MIMIC-IV's native distribution format).

    One parquet per input file — no row chunking — with the config's columns projected
    and dtypes inferred over the whole file, so ``result`` is Float64 rather than text.
    """
    import gzip

    from MEDS_extract.convert_to_parquet.convert_to_parquet import main as convert_stage

    event_cfg = """\
labs:
  lab:
    code: $test_name
    time: null
    numeric_value: $result
"""

    root = tmp_path
    raw_dir = root / "raw_cohort"
    raw_dir.mkdir()
    with gzip.open(raw_dir / "labs.csv.gz", mode="wt") as f:
        f.write("subject_id,test_name,result,ignored_col\n1,HR,80,x\n1,TEMP,36.6,x\n2,HR,75,x\n")

    messy_fp = root / "messy.yaml"
    messy_fp.write_text(event_cfg)

    cfg = _make_cfg(
        {
            "stage": "convert_to_parquet",
            "input_dir": str(raw_dir),
            "stage_cfg": {
                "data_input_dir": str(raw_dir / "data"),
                "output_dir": str(root / "output" / "data"),
            },
            "MESSY_config_fp": str(messy_fp),
        }
    )
    convert_stage.main_fn(cfg)

    out_fp = root / "output" / "data" / "labs.parquet"
    assert out_fp.is_file(), "one output per input file, named for the source"
    df = pl.read_parquet(out_fp, glob=False).sort("subject_id", "test_name")
    # Only config-referenced columns are projected; ``ignored_col`` never lands.
    assert df.columns == ["result", "subject_id", "test_name"]
    assert df["subject_id"].to_list() == [1, 1, 2]
    assert df["test_name"].to_list() == ["HR", "TEMP", "HR"]
    assert df["result"].to_list() == [80.0, 36.6, 75.0]
    assert df.schema["result"] == pl.Float64, "full-file inference, not all-String"


def test_convert_to_parquet_hardlinks_parquet_sources(tmp_path):
    """A parquet source is linked, not rewritten — same inode, so no bytes are copied."""
    from MEDS_extract.convert_to_parquet.convert_to_parquet import main as convert_stage

    root = tmp_path
    raw_dir = root / "raw_cohort"
    raw_dir.mkdir()
    src = raw_dir / "labs.parquet"
    pl.DataFrame({"subject_id": [1, 2], "test_name": ["HR", "TEMP"]}).write_parquet(src)

    messy_fp = root / "messy.yaml"
    messy_fp.write_text("labs:\n  lab:\n    code: $test_name\n    time: null\n")

    cfg = _make_cfg(
        {
            "stage": "convert_to_parquet",
            "input_dir": str(raw_dir),
            "stage_cfg": {
                "data_input_dir": str(raw_dir / "data"),
                "output_dir": str(root / "output" / "data"),
            },
            "MESSY_config_fp": str(messy_fp),
        }
    )
    convert_stage.main_fn(cfg)

    out_fp = root / "output" / "data" / "labs.parquet"
    assert out_fp.stat().st_ino == src.stat().st_ino, "parquet input should be hardlinked, not rewritten"


def test_convert_to_parquet_projects_wide_parquet_sources(tmp_path):
    """A parquet source with unread columns is REWRITTEN projected, not linked.

    Reads are unaffected by extra columns (parquet is columnar), so the reason to prune
    is writes: `convert_to_subject_sharded` does not project, so unread columns are
    copied into its output — a one-time cost, but a large one for a wide source, and it
    widens frames in the stage that is already the memory hot spot.
    """
    from MEDS_extract.convert_to_parquet.convert_to_parquet import main as convert_stage

    root = tmp_path
    raw_dir = root / "raw_cohort"
    raw_dir.mkdir()
    src = raw_dir / "labs.parquet"
    pl.DataFrame(
        {
            "subject_id": [1, 2],
            "test_name": ["HR", "TEMP"],
            **{f"unread_{i}": ["x", "y"] for i in range(5)},
        }
    ).write_parquet(src)

    messy_fp = root / "messy.yaml"
    messy_fp.write_text("labs:\n  lab:\n    code: $test_name\n    time: null\n")

    cfg = _make_cfg(
        {
            "stage": "convert_to_parquet",
            "input_dir": str(raw_dir),
            "stage_cfg": {
                "data_input_dir": str(raw_dir / "data"),
                "output_dir": str(root / "output" / "data"),
            },
            "MESSY_config_fp": str(messy_fp),
        }
    )
    convert_stage.main_fn(cfg)

    out_fp = root / "output" / "data" / "labs.parquet"
    assert out_fp.stat().st_ino != src.stat().st_ino, "a wide parquet source must be rewritten"
    assert pl.read_parquet(out_fp).columns == ["subject_id", "test_name"]


# ── split_and_shard_subjects: external splits JSON wiring ──


def test_split_and_shard_subjects_external_splits(tmp_path):
    """``external_splits_json_fp`` flows from ``stage_cfg`` into the written ``.shards.json``.

    Externally-listed subjects land in their own named split and are excluded from the IID
    train/tuning/held_out splits; every subject appears somewhere.
    """
    from MEDS_extract.split_and_shard_subjects.split_and_shard_subjects import main as sss_stage

    root = tmp_path
    input_dir = root / "input"
    input_dir.mkdir()
    pl.DataFrame({"subject_id": list(range(1, 11))}).write_parquet(input_dir / "patients.parquet")

    event_cfg_fp = root / "messy.yaml"
    event_cfg_fp.write_text("patients:\n  e:\n    code: X\n    time: null\n")

    ext_fp = root / "external_splits.json"
    ext_fp.write_text(json.dumps({"prospective_test": [9, 10]}))

    shards_fp = root / "metadata" / ".shards.json"

    cfg = _make_cfg(
        {
            "stage_cfg": {
                "data_input_dir": str(input_dir),
                "external_splits_json_fp": str(ext_fp),
                "split_fracs": {"train": 0.8, "tuning": 0.1, "held_out": 0.1},
                "n_subjects_per_shard": 10,
            },
            "MESSY_config_fp": str(event_cfg_fp),
            "shards_map_fp": str(shards_fp),
        }
    )
    sss_stage.main_fn(cfg)

    shards = json.loads(shards_fp.read_text())
    assert set(shards) == {"prospective_test/0", "train/0", "tuning/0", "held_out/0"}
    # The external split is honored verbatim...
    assert set(shards["prospective_test/0"]) == {9, 10}
    # ...its subjects are excluded from the IID splits, which partition the remainder.
    iid = [s for k, v in shards.items() if k != "prospective_test/0" for s in v]
    assert sorted(iid) == list(range(1, 9))


def test_split_and_shard_subjects_external_splits_file_missing(tmp_path):
    """A configured-but-missing external splits JSON raises ``FileNotFoundError`` naming the path."""
    from MEDS_extract.split_and_shard_subjects.split_and_shard_subjects import main as sss_stage

    root = tmp_path
    input_dir = root / "input"
    input_dir.mkdir()
    pl.DataFrame({"subject_id": [1, 2, 3]}).write_parquet(input_dir / "patients.parquet")

    event_cfg_fp = root / "messy.yaml"
    event_cfg_fp.write_text("patients:\n  e:\n    code: X\n    time: null\n")

    cfg = _make_cfg(
        {
            "stage_cfg": {
                "data_input_dir": str(input_dir),
                "external_splits_json_fp": str(root / "no_such_splits.json"),
                "split_fracs": {"train": 0.8, "tuning": 0.1, "held_out": 0.1},
                "n_subjects_per_shard": 10,
            },
            "MESSY_config_fp": str(event_cfg_fp),
            "shards_map_fp": str(root / "metadata" / ".shards.json"),
        }
    )
    with pytest.raises(FileNotFoundError, match="External splits JSON file not found"):
        sss_stage.main_fn(cfg)


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

        # codes.parquet: the pre-existing dummy bytes were replaced by a VALID parquet
        # carrying the canonical (empty — no input codes) MEDS code-metadata schema.
        codes = pl.read_parquet(out_dir / "codes.parquet")
        assert codes.height == 0
        assert codes.schema["code"] == pl.String
        assert codes.schema["description"] == pl.String
        assert codes.schema["parent_codes"] == pl.List(pl.String)

        # subject_splits.parquet: valid parquet with both shard subjects in the train split.
        splits = pl.read_parquet(out_dir / "subject_splits.parquet")
        assert splits.schema["subject_id"] == pl.Int64
        assert splits.schema["split"] == pl.String
        assert sorted(splits["subject_id"].to_list()) == [1, 2]
        assert splits["split"].to_list() == ["train", "train"]


# ── do_overwrite / parallelism guard ──


@pytest.mark.parametrize(
    ("do_overwrite", "worker", "should_run"),
    [
        (False, 0, True),  # ordinary serial run
        (False, 1, True),  # ordinary parallel run — every worker participates
        (True, 0, True),  # worker 0 always runs, so the stage still completes
        (True, 1, False),  # stands down: do_overwrite disables work-sharing
    ],
)
def test_overwrite_guard_stands_down_only_for_nonzero_workers(
    tmp_path, caplog, do_overwrite: bool, worker: int, should_run: bool
):
    """``do_overwrite=True`` makes every worker but 0 exit before doing any work.

    Guards the fix for #194. Parametrized over the full truth table because the risk is
    a guard that is too eager: a serial run (``worker`` defaults to 0) and any ordinary
    parallel run must be completely unaffected, or the guard would silently halve the
    pipeline. ``convert_to_parquet`` stands in for all four map stages — they share one
    helper, whose own behavior is pinned by its doctests.
    """
    from MEDS_extract.convert_to_parquet.convert_to_parquet import main as convert_stage

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pl.DataFrame({"subject_id": [1], "test_name": ["HR"]}).write_parquet(raw_dir / "labs.parquet")
    messy_fp = tmp_path / "messy.yaml"
    messy_fp.write_text("labs:\n  lab:\n    code: $test_name\n    time: null\n")

    cfg = _make_cfg(
        {
            "stage": "convert_to_parquet",
            "input_dir": str(raw_dir),
            "do_overwrite": do_overwrite,
            "worker": worker,
            "stage_cfg": {
                "data_input_dir": str(raw_dir / "data"),
                "output_dir": str(tmp_path / "out"),
            },
            "MESSY_config_fp": str(messy_fp),
        }
    )
    with caplog.at_level(logging.WARNING):
        convert_stage.main_fn(cfg)

    produced = (tmp_path / "out" / "labs.parquet").exists()
    assert produced is should_run, (
        f"do_overwrite={do_overwrite}, worker={worker}: expected "
        f"{'output' if should_run else 'no output'}, got {'output' if produced else 'none'}"
    )
    if not should_run:
        # The warning must name the parallel alternative, or a user just loses throughput
        # with no idea why.
        assert "standing down" in caplog.text
        assert "delete the stage's output directory" in caplog.text
