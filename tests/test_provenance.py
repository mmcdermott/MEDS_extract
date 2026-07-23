"""In-process end-to-end tests for row-level provenance tracking (issue #132).

Runs shard_events → convert_to_subject_sharded → convert_to_MEDS_events → merge_to_MEDS_cohort
over a small raw dataset twice — once with ``do_track_provenance=true`` and once without — and
asserts the core invariants:

* flag off: no anchor/provenance columns in the MEDS output (byte-identical to today);
* flag on: identical rows in identical order, plus one ``provenance`` column;
* 1→N event fan-out: rows from different event blocks point at the same ``(file, row)``;
* dedup merge: two identical raw rows (necessarily with different row indices) collapse into
  one MEDS row whose provenance is the two-element set of both anchors;
* ``source_file`` values are input-dir-relative paths and ``row_idx`` values are the actual
  0-based positions in the raw files;
* the provenance column survives ``DataSchema.align`` (the finalize_MEDS_data transform).
"""

import json
import tempfile
from pathlib import Path

import polars as pl
from meds import DataSchema
from omegaconf import OmegaConf
from polars.testing import assert_frame_equal

_ = pl.Config.set_tbl_width_chars(600)

PATIENTS_CSV = """\
MRN,dob,eye_color
1,2000-01-01T00:00:00,BROWN
2,2001-02-02T00:00:00,BLUE
"""

# Rows 1 and 2 are identical on purpose: they produce identical MEDS rows whose provenance
# anchors differ (row_idx 1 vs 2), so they only collapse if the dedup key excludes provenance.
LABS_CSV = """\
patient_id,timestamp,test_name,result
1,2020-01-01T10:00:00,HR,80
1,2020-01-01T11:00:00,TEMP,36.6
1,2020-01-01T11:00:00,TEMP,36.6
2,2020-01-02T12:00:00,HR,75
"""

# `patients` fans out 1→2: each raw row feeds both the eye_color and the dob event block.
EVENT_CFG = """\
_defaults:
  subject_id: $MRN
patients:
  eye_color:
    code: 'f"EYE_COLOR//{$eye_color}"'
    time: null
  dob:
    code: MEDS_BIRTH
    time: '$dob::"%Y-%m-%dT%H:%M:%S"'
labs:
  _defaults:
    subject_id: $patient_id
  lab:
    code: $test_name
    time: '$timestamp::"%Y-%m-%dT%H:%M:%S"'
    numeric_value: $result
"""


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


def _run_pipeline(root: Path, do_track_provenance: bool) -> pl.DataFrame:
    """Run the four data stages in-process; return the merged ``train/0`` shard."""
    from MEDS_extract.convert_to_MEDS_events.convert_to_MEDS_events import main as cme_stage
    from MEDS_extract.convert_to_subject_sharded.convert_to_subject_sharded import main as css_stage
    from MEDS_extract.merge_to_MEDS_cohort.merge_to_MEDS_cohort import main as merge_stage
    from MEDS_extract.shard_events.shard_events import main as shard_stage

    raw_dir = root / "raw_cohort"
    raw_dir.mkdir(exist_ok=True)
    (raw_dir / "patients.csv").write_text(PATIENTS_CSV)
    (raw_dir / "labs.csv").write_text(LABS_CSV)

    event_cfg_fp = root / "event_cfgs.yaml"
    event_cfg_fp.write_text(EVENT_CFG)
    shards_fp = root / ".shards.json"
    shards_fp.write_text(json.dumps({"train/0": [1, 2]}))

    tag = "on" if do_track_provenance else "off"
    subsharded = root / f"subsharded_{tag}" / "data"
    subject_sharded = root / f"subject_sharded_{tag}" / "data"
    events = root / f"events_{tag}" / "data"
    merged = root / f"merged_{tag}" / "data"

    shard_stage.main_fn(
        _make_cfg(
            {
                "stage": "shard_events",
                "stage_cfg": {
                    "data_input_dir": str(raw_dir / "data"),
                    "output_dir": str(subsharded),
                    # Chunk size 2 splits the identical labs rows (idx 1 and 2) across
                    # different sub-shards; they must still merge downstream.
                    "row_chunksize": 2,
                    "infer_schema_length": 10000,
                },
                "event_conversion_config_fp": str(event_cfg_fp),
            }
        )
    )

    css_stage.main_fn(
        _make_cfg(
            {
                "stage": "convert_to_subject_sharded",
                "stage_cfg": {"data_input_dir": str(subsharded), "output_dir": str(subject_sharded)},
                "event_conversion_config_fp": str(event_cfg_fp),
                "shards_map_fp": str(shards_fp),
            }
        )
    )

    cme_stage.main_fn(
        _make_cfg(
            {
                "stage": "convert_to_MEDS_events",
                "stage_cfg": {
                    "data_input_dir": str(subject_sharded),
                    "output_dir": str(events),
                    "do_dedup_text_and_numeric": False,
                    "do_track_provenance": do_track_provenance,
                },
                "event_conversion_config_fp": str(event_cfg_fp),
                "shards_map_fp": str(shards_fp),
            }
        )
    )

    merge_stage.main_fn(
        _make_cfg(
            {
                "stage": "merge_to_MEDS_cohort",
                "stage_cfg": {
                    "data_input_dir": str(events),
                    "output_dir": str(merged),
                    "unique_by": "*",
                },
                "event_conversion_config_fp": str(event_cfg_fp),
                "shards_map_fp": str(shards_fp),
            }
        )
    )

    return pl.read_parquet(merged / "train" / "0.parquet")


def test_provenance_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        df_on = _run_pipeline(root, do_track_provenance=True)
        df_off = _run_pipeline(root, do_track_provenance=False)

        # ── Flag off: no provenance, no leaked anchors ──
        for col in ("provenance", "__row_idx__", "__source_file__"):
            assert col not in df_off.columns, f"{col} must not appear in a provenance-off run."

        # ── Flag on: identical rows (content, count, AND order), plus exactly `provenance` ──
        assert set(df_on.columns) == {*df_off.columns, "provenance"}
        assert_frame_equal(df_on.drop("provenance"), df_off)
        assert df_on.schema["provenance"] == pl.List(
            pl.Struct({"source_file": pl.String, "row_idx": pl.UInt32})
        )

        prov_of = {}  # (code, subject_id) → list of (source_file, row_idx) tuples
        for row in df_on.iter_rows(named=True):
            anchors = [(p["source_file"], p["row_idx"]) for p in row["provenance"]]
            prov_of[(row["code"], row["subject_id"])] = anchors

        # ── (a) 1→N fan-out: both blocks' rows point at the same (file, row) ──
        assert prov_of[("EYE_COLOR//BROWN", 1)] == [("patients.csv", 0)]
        assert prov_of[("MEDS_BIRTH", 1)] == [("patients.csv", 0)]
        assert prov_of[("EYE_COLOR//BLUE", 2)] == [("patients.csv", 1)]
        assert prov_of[("MEDS_BIRTH", 2)] == [("patients.csv", 1)]

        # ── (b) dedup merge: identical raw rows 1 and 2 → one MEDS row, 2-element set ──
        assert (df_on["code"] == "TEMP").sum() == 1
        assert sorted(prov_of[("TEMP", 1)]) == [("labs.csv", 1), ("labs.csv", 2)]

        # ── Anchors are input-dir-relative paths with true 0-based row indices ──
        assert prov_of[("HR", 1)] == [("labs.csv", 0)]
        assert prov_of[("HR", 2)] == [("labs.csv", 3)]
        raw = {
            "patients.csv": pl.read_csv(PATIENTS_CSV.encode()),
            "labs.csv": pl.read_csv(LABS_CSV.encode()),
        }
        for row in df_on.iter_rows(named=True):
            for p in row["provenance"]:
                src = raw[p["source_file"]].row(p["row_idx"], named=True)
                if p["source_file"] == "labs.csv":
                    assert src["test_name"] == row["code"]
                    assert src["result"] == row["numeric_value"]
                elif row["code"] == "MEDS_BIRTH":
                    assert src["dob"] == row["time"].isoformat()
                else:
                    assert row["code"] == f"EYE_COLOR//{src['eye_color']}"

        # ── The provenance column survives finalize_MEDS_data's DataSchema.align ──
        aligned = DataSchema.align(df_on.to_arrow())
        assert "provenance" in aligned.schema.names
        final = pl.from_arrow(aligned)
        assert final.height == df_on.height
        assert sorted(
            (p["source_file"], p["row_idx"]) for p in final.filter(pl.col("code") == "TEMP")["provenance"][0]
        ) == [("labs.csv", 1), ("labs.csv", 2)]
