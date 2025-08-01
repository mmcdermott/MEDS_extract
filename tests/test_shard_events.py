"""Tests the shard events stage in isolation.

Set the bash env variable `DO_USE_LOCAL_SCRIPTS=1` to use the local py files, rather than the installed
scripts.
"""

from io import StringIO

import polars as pl
from omegaconf import OmegaConf
from yaml import load as load_yaml

from MEDS_extract.shard_events.shard_events import retrieve_columns
from tests import SHARD_EVENTS_SCRIPT
from tests.utils import single_stage_tester

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

SUBJECTS_CSV = """
MRN,dob,eye_color,height
1195293,06/20/1978,BLUE,164.6868838269085
239684,12/28/1980,BROWN,175.271115221764
1500733,07/20/1986,BROWN,158.60131573580904
814703,03/28/1976,HAZEL,156.48559093209357
754281,12/19/1988,BROWN,166.22261567137025
68729,03/09/1978,HAZEL,160.3953106166676
"""

EMPTY_SUBJECTS_CSV = """
MRN,dob,eye_color,height
"""

VITALS_JOIN_CSV = """\
stay_id,charttime,HR
10,01/01/2021 00:00:00,70
10,01/01/2021 01:00:00,75
20,01/01/2021 02:00:00,65
"""

STAYS_JOIN_CSV = """\
stay_id,subject_id
10,111
20,222
"""

EVENT_CFG_JOIN_YAML = """\
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

ADMIT_VITALS_CSV = """
subject_id,admit_date,disch_date,department,vitals_date,HR,temp
239684,"05/11/2010, 17:41:51","05/11/2010, 19:27:19",CARDIAC,"05/11/2010, 18:57:18",112.6,95.5
754281,"01/03/2010, 06:27:59","01/03/2010, 08:22:13",PULMONARY,"01/03/2010, 06:27:59",142.0,99.8
814703,"02/05/2010, 05:55:39","02/05/2010, 07:02:30",ORTHOPEDIC,"02/05/2010, 05:55:39",170.2,100.1
239684,"05/11/2010, 17:41:51","05/11/2010, 19:27:19",CARDIAC,"05/11/2010, 18:25:35",113.4,95.8
68729,"05/26/2010, 02:30:56","05/26/2010, 04:51:52",PULMONARY,"05/26/2010, 02:30:56",86.0,97.8
1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 20:12:31",112.5,99.8
1500733,"06/03/2010, 14:54:38","06/03/2010, 16:44:26",ORTHOPEDIC,"06/03/2010, 16:20:49",90.1,100.1
239684,"05/11/2010, 17:41:51","05/11/2010, 19:27:19",CARDIAC,"05/11/2010, 17:48:48",105.1,96.2
239684,"05/11/2010, 17:41:51","05/11/2010, 19:27:19",CARDIAC,"05/11/2010, 17:41:51",102.6,96.0
1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 19:25:32",114.1,100.0
1500733,"06/03/2010, 14:54:38","06/03/2010, 16:44:26",ORTHOPEDIC,"06/03/2010, 14:54:38",91.4,100.0
1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 20:41:33",107.5,100.4
1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 20:24:44",107.7,100.0
1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 19:45:19",119.8,99.9
1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 19:23:52",109.0,100.0
1500733,"06/03/2010, 14:54:38","06/03/2010, 16:44:26",ORTHOPEDIC,"06/03/2010, 15:39:49",84.4,100.3
"""

EVENT_CFGS_YAML = """
subjects:
  subject_id_col: MRN
  eye_color:
    code:
      - EYE_COLOR
      - col(eye_color)
    time: null
    _metadata:
      demo_metadata:
        description: description
  height:
    code: HEIGHT
    time: null
    numeric_value: height
  dob:
    code: DOB
    time: col(dob)
    time_format: "%m/%d/%Y"
admit_vitals:
  admissions:
    code:
      - ADMISSION
      - col(department)
    time: col(admit_date)
    time_format: "%m/%d/%Y, %H:%M:%S"
  discharge:
    code: DISCHARGE
    time: col(disch_date)
    time_format: "%m/%d/%Y, %H:%M:%S"
  HR:
    code: HR
    time: col(vitals_date)
    time_format: "%m/%d/%Y, %H:%M:%S"
    numeric_value: HR
    _metadata:
      input_metadata:
        description: {"title": {"lab_code": "HR"}}
        parent_codes: {"LOINC/{loinc}": {"lab_code": "HR"}}
  temp:
    code: TEMP
    time: col(vitals_date)
    time_format: "%m/%d/%Y, %H:%M:%S"
    numeric_value: temp
    _metadata:
      input_metadata:
        description: {"title": {"lab_code": "temp"}}
        parent_codes: {"LOINC/{loinc}": {"lab_code": "temp"}}
"""


def test_shard_events():
    single_stage_tester(
        script=SHARD_EVENTS_SCRIPT,
        stage_name="shard_events",
        stage_kwargs={"row_chunksize": 10},
        input_files={
            "subjects.csv": SUBJECTS_CSV,
            "admit_vitals.csv": ADMIT_VITALS_CSV,
            "admit_vitals.parquet": pl.read_csv(StringIO(ADMIT_VITALS_CSV)),
            "event_cfgs.yaml": EVENT_CFGS_YAML,
        },
        event_conversion_config_fp="{input_dir}/event_cfgs.yaml",
        want_outputs={
            "data/subjects/[0-6).parquet": pl.read_csv(StringIO(SUBJECTS_CSV)),
            "data/admit_vitals/[0-10).parquet": pl.read_csv(StringIO(ADMIT_VITALS_CSV))[:10],
            "data/admit_vitals/[10-16).parquet": pl.read_csv(StringIO(ADMIT_VITALS_CSV))[10:],
        },
        df_check_kwargs={"check_column_order": False},
        test_name="Shard events should preferentially use .parquet files over .csv files.",
    )

    single_stage_tester(
        script=SHARD_EVENTS_SCRIPT,
        stage_name="shard_events",
        stage_kwargs={"row_chunksize": 10},
        input_files={
            "subjects.csv": SUBJECTS_CSV,
            "admit_vitals.par": pl.read_csv(StringIO(ADMIT_VITALS_CSV)),
            "event_cfgs.yaml": EVENT_CFGS_YAML,
        },
        event_conversion_config_fp="{input_dir}/event_cfgs.yaml",
        want_outputs={
            "data/subjects/[0-6).parquet": pl.read_csv(StringIO(SUBJECTS_CSV)),
            "data/admit_vitals/[0-10).parquet": pl.read_csv(StringIO(ADMIT_VITALS_CSV))[:10],
            "data/admit_vitals/[10-16).parquet": pl.read_csv(StringIO(ADMIT_VITALS_CSV))[10:],
        },
        df_check_kwargs={"check_column_order": False},
        test_name="Shard events should accept .par files as parquet files.",
    )

    single_stage_tester(
        script=SHARD_EVENTS_SCRIPT,
        stage_name="shard_events",
        stage_kwargs={"row_chunksize": 10},
        input_files={
            "subjects.csv": SUBJECTS_CSV,
            "admit_vitals.csv": ADMIT_VITALS_CSV,
        },
        event_conversion_config_fp="{input_dir}/event_cfgs.yaml",
        should_error=True,
        test_name="Shard events should error without event conversion config",
    )

    single_stage_tester(
        script=SHARD_EVENTS_SCRIPT,
        stage_name="shard_events",
        stage_kwargs={"row_chunksize": 10},
        input_files={"event_cfgs.yaml": EVENT_CFGS_YAML},
        event_conversion_config_fp="{input_dir}/event_cfgs.yaml",
        should_error=True,
        test_name="Shard events should error when missing all input files",
    )

    single_stage_tester(
        script=SHARD_EVENTS_SCRIPT,
        stage_name="shard_events",
        stage_kwargs={"row_chunksize": 10},
        input_files={
            "subjects.csv": EMPTY_SUBJECTS_CSV,
            "event_cfgs.yaml": EVENT_CFGS_YAML,
        },
        event_conversion_config_fp="{input_dir}/event_cfgs.yaml",
        should_error=True,
        test_name="Shard events should error when an input file is empty",
    )


def test_retrieve_columns_join():
    cfg = OmegaConf.create(load_yaml(EVENT_CFG_JOIN_YAML, Loader=Loader))
    cols = retrieve_columns(cfg)
    assert set(cols["vitals"]) == {"HR", "charttime", "stay_id"}
    assert set(cols["stays"]) == {"stay_id", "subject_id"}


def test_shard_events_join():
    single_stage_tester(
        script=SHARD_EVENTS_SCRIPT,
        stage_name="shard_events",
        stage_kwargs={"row_chunksize": 2},
        input_files={
            "vitals.csv": VITALS_JOIN_CSV,
            "stays.csv": STAYS_JOIN_CSV,
            "event_cfg.yaml": EVENT_CFG_JOIN_YAML,
        },
        event_conversion_config_fp="{input_dir}/event_cfg.yaml",
        want_outputs={
            "data/vitals/[0-2).parquet": pl.read_csv(StringIO(VITALS_JOIN_CSV))[:2],
            "data/vitals/[2-3).parquet": pl.read_csv(StringIO(VITALS_JOIN_CSV))[2:],
            "data/stays/[0-2).parquet": pl.read_csv(StringIO(STAYS_JOIN_CSV)),
        },
        df_check_kwargs={"check_column_order": False},
        test_name="Shard events with join config should sub-shard all tables",
    )
