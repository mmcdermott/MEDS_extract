from io import StringIO

import polars as pl

from tests import CONVERT_TO_SUBJECT_SHARDED_SCRIPT, SPLIT_AND_SHARD_SCRIPT
from tests.utils import single_stage_tester

VITALS_CSV = """\
stay_id,charttime,HR
10,01/01/2021 00:00:00,70
10,01/01/2021 01:00:00,75
20,01/01/2021 02:00:00,65
"""

STAYS_CSV = """\
stay_id,subject_id
10,111
20,222
"""

EVENT_CFG_YAML = """\
vitals:
  join:
    input_prefix: stays
    left_on: stay_id
    right_on: stay_id
    columns_from_right:
      - subject_id
  subject_id_col: subject_id
  HR:
    code: HR
    time: col(charttime)
    time_format: "%m/%d/%Y %H:%M:%S"
    numeric_value: HR
stays:
  subject_id_col: subject_id
"""

EXPECTED_SHARDS = {"train/0": [111], "tuning/0": [222]}

TRAIN_DF = (
    pl.read_csv(StringIO(VITALS_CSV))
    .join(pl.read_csv(StringIO(STAYS_CSV)), on="stay_id")
    .filter(pl.col("subject_id") == 111)
)

TUNING_DF = (
    pl.read_csv(StringIO(VITALS_CSV))
    .join(pl.read_csv(StringIO(STAYS_CSV)), on="stay_id")
    .filter(pl.col("subject_id") == 222)
)

STAYS_DF = pl.read_csv(StringIO(STAYS_CSV))
TRAIN_STAYS = STAYS_DF.filter(pl.col("subject_id") == 111)
TUNING_STAYS = STAYS_DF.filter(pl.col("subject_id") == 222)


def test_join_tables_split_and_shard():
    single_stage_tester(
        script=SPLIT_AND_SHARD_SCRIPT,
        stage_name="split_and_shard_subjects",
        stage_kwargs={
            "split_fracs.train": 0.5,
            "split_fracs.tuning": 0.5,
            "split_fracs.held_out": None,
            "n_subjects_per_shard": 10,
        },
        input_files={
            "data/vitals/[0-3).parquet": pl.read_csv(StringIO(VITALS_CSV)),
            "data/stays/[0-2).parquet": pl.read_csv(StringIO(STAYS_CSV)),
            "event_cfg.yaml": EVENT_CFG_YAML,
        },
        event_conversion_config_fp="{input_dir}/event_cfg.yaml",
        shards_map_fp="{output_dir}/metadata/.shards.json",
        want_outputs={"metadata/.shards.json": EXPECTED_SHARDS},
    )


def test_join_tables_convert_to_subject_sharded():
    single_stage_tester(
        script=CONVERT_TO_SUBJECT_SHARDED_SCRIPT,
        stage_name="convert_to_subject_sharded",
        stage_kwargs={},
        input_files={
            "data/vitals/[0-3).parquet": pl.read_csv(StringIO(VITALS_CSV)),
            "data/stays/[0-2).parquet": pl.read_csv(StringIO(STAYS_CSV)),
            "metadata/.shards.json": EXPECTED_SHARDS,
            "event_cfg.yaml": EVENT_CFG_YAML,
        },
        event_conversion_config_fp="{input_dir}/event_cfg.yaml",
        shards_map_fp="{input_dir}/metadata/.shards.json",
        want_outputs={
            "data/train/0/vitals.parquet": TRAIN_DF,
            "data/tuning/0/vitals.parquet": TUNING_DF,
            "data/train/0/stays.parquet": TRAIN_STAYS,
            "data/tuning/0/stays.parquet": TUNING_STAYS,
        },
        df_check_kwargs={"check_row_order": False, "check_column_order": False},
    )
