"""Tests the convert to sharded events process.

Set the bash env variable `DO_USE_LOCAL_SCRIPTS=1` to use the local py files, rather than the installed
scripts.
"""

from io import StringIO

import polars as pl

from tests import CONVERT_TO_SUBJECT_SHARDED_SCRIPT
from tests.utils import Loader, load_yaml, single_stage_tester

SUBJECTS_CSV = """
MRN,dob,eye_color,height
1195293,06/20/1978,BLUE,164.6868838269085
239684,12/28/1980,BROWN,175.271115221764
1500733,07/20/1986,BROWN,158.60131573580904
814703,03/28/1976,HAZEL,156.48559093209357
754281,12/19/1988,BROWN,166.22261567137025
68729,03/09/1978,HAZEL,160.3953106166676
"""

ADMIT_VITALS_0_10_CSV = """
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
"""

ADMIT_VITALS_10_16_CSV = """
subject_id,admit_date,disch_date,department,vitals_date,HR,temp
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
    text_value: temp
    _metadata:
      input_metadata:
        description: {"title": {"lab_code": "temp"}}
        parent_codes: {"LOINC/{loinc}": {"lab_code": "temp"}}
"""

SHARDS_JSON = {
    "train/0": [239684, 1195293],
    "train/1": [68729, 814703],
    "tuning/0": [754281],
    "held_out/0": [1500733],
}


WANT_OUTPUTS_YAML = """
data/train/0/subjects.parquet: |-2
  MRN,dob,eye_color,height
  1195293,06/20/1978,BLUE,164.6868838269085
  239684,12/28/1980,BROWN,175.271115221764

data/train/1/subjects.parquet: |-2
  MRN,dob,eye_color,height
  814703,03/28/1976,HAZEL,156.48559093209357
  68729,03/09/1978,HAZEL,160.3953106166676

data/tuning/0/subjects.parquet: |-2
  MRN,dob,eye_color,height
  754281,12/19/1988,BROWN,166.22261567137025

data/held_out/0/subjects.parquet: |-2
  MRN,dob,eye_color,height
  1500733,07/20/1986,BROWN,158.60131573580904

data/train/0/admit_vitals.parquet: |-2
  subject_id,admit_date,disch_date,department,vitals_date,HR,temp
  239684,"05/11/2010, 17:41:51","05/11/2010, 19:27:19",CARDIAC,"05/11/2010, 18:57:18",112.6,95.5
  239684,"05/11/2010, 17:41:51","05/11/2010, 19:27:19",CARDIAC,"05/11/2010, 18:25:35",113.4,95.8
  1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 20:12:31",112.5,99.8
  239684,"05/11/2010, 17:41:51","05/11/2010, 19:27:19",CARDIAC,"05/11/2010, 17:48:48",105.1,96.2
  239684,"05/11/2010, 17:41:51","05/11/2010, 19:27:19",CARDIAC,"05/11/2010, 17:41:51",102.6,96.0
  1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 19:25:32",114.1,100.0
  1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 20:41:33",107.5,100.4
  1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 20:24:44",107.7,100.0
  1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 19:45:19",119.8,99.9
  1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 19:23:52",109.0,100.0

data/train/1/admit_vitals.parquet: |-2
  subject_id,admit_date,disch_date,department,vitals_date,HR,temp
  814703,"02/05/2010, 05:55:39","02/05/2010, 07:02:30",ORTHOPEDIC,"02/05/2010, 05:55:39",170.2,100.1
  68729,"05/26/2010, 02:30:56","05/26/2010, 04:51:52",PULMONARY,"05/26/2010, 02:30:56",86.0,97.8

data/tuning/0/admit_vitals.parquet: |-2
  subject_id,admit_date,disch_date,department,vitals_date,HR,temp
  754281,"01/03/2010, 06:27:59","01/03/2010, 08:22:13",PULMONARY,"01/03/2010, 06:27:59",142.0,99.8

data/held_out/0/admit_vitals.parquet: |-2
  subject_id,admit_date,disch_date,department,vitals_date,HR,temp
  1500733,"06/03/2010, 14:54:38","06/03/2010, 16:44:26",ORTHOPEDIC,"06/03/2010, 16:20:49",90.1,100.1
  1500733,"06/03/2010, 14:54:38","06/03/2010, 16:44:26",ORTHOPEDIC,"06/03/2010, 14:54:38",91.4,100.0
  1500733,"06/03/2010, 14:54:38","06/03/2010, 16:44:26",ORTHOPEDIC,"06/03/2010, 15:39:49",84.4,100.3
    """

WANT_OUTPUTS = {}
for k, v in load_yaml(WANT_OUTPUTS_YAML.strip(), Loader=Loader).items():
    v = pl.read_csv(StringIO(v))
    WANT_OUTPUTS[k] = v


def test_convert_to_subject_sharded():
    single_stage_tester(
        script=CONVERT_TO_SUBJECT_SHARDED_SCRIPT,
        stage_name="convert_to_subject_sharded",
        stage_kwargs={},
        input_files={
            "data/subjects/[0-6).parquet": pl.read_csv(StringIO(SUBJECTS_CSV)),
            "data/admit_vitals/[0-10).parquet": pl.read_csv(StringIO(ADMIT_VITALS_0_10_CSV)),
            "data/admit_vitals/[10-16).parquet": pl.read_csv(StringIO(ADMIT_VITALS_10_16_CSV)),
            "event_cfgs.yaml": EVENT_CFGS_YAML,
            "metadata/.shards.json": SHARDS_JSON,
        },
        event_conversion_config_fp="{input_dir}/event_cfgs.yaml",
        shards_map_fp="{input_dir}/metadata/.shards.json",
        want_outputs=WANT_OUTPUTS,
        test_name="Stage tester: convert_to_subject_sharded",
        df_check_kwargs={
            "check_row_order": False,
            "check_column_order": False,
            "check_dtypes": False,
        },
    )

    # If we don't provide the event_cfgs.yaml file, the script should error.
    single_stage_tester(
        script=CONVERT_TO_SUBJECT_SHARDED_SCRIPT,
        stage_name="convert_to_subject_sharded",
        stage_kwargs={},
        input_files={
            "data/subjects/[0-6).parquet": pl.read_csv(StringIO(SUBJECTS_CSV)),
            "data/admit_vitals/[0-10).parquet": pl.read_csv(StringIO(ADMIT_VITALS_0_10_CSV)),
            "data/admit_vitals/[10-16).parquet": pl.read_csv(StringIO(ADMIT_VITALS_10_16_CSV)),
            "metadata/.shards.json": SHARDS_JSON,
        },
        event_conversion_config_fp="{input_dir}/event_cfgs.yaml",
        shards_map_fp="{input_dir}/metadata/.shards.json",
        test_name="Stage tester: convert_to_subject_sharded",
        should_error=True,
    )
