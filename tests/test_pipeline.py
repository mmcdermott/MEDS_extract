"""End-to-end pipeline test via ``MEDS_transforms.pytest_plugin.pipeline_tester``.

Runs all eight MEDS_extract stages in sequence and validates each stage's output against its
registered ``default`` scenario. The first stage's ``in.yaml`` bootstraps the pipeline's input
(raw CSVs + event_cfg.yaml); each subsequent stage reads the prior stage's output.
"""

from MEDS_transforms.pytest_plugin import pipeline_tester

PIPELINE_YAML = """
input_dir: "{input_dir}"
output_dir: "{output_dir}"
event_conversion_config_fp: "{input_dir}/event_cfg.yaml"
shards_map_fp: "{output_dir}/metadata/.shards.json"
etl_metadata:
  dataset_name: TEST
  dataset_version: "0.1"
seed: 1
stages:
  - shard_events:
      row_chunksize: 2
      infer_schema_length: 10000
  - split_and_shard_subjects:
      split_fracs:
        train: 0.5
        tuning: 0.25
        held_out: 0.25
      n_subjects_per_shard: 10
  - convert_to_subject_sharded
  - convert_to_MEDS_events:
      do_dedup_text_and_numeric: false
  - merge_to_MEDS_cohort
  - extract_code_metadata
  - finalize_MEDS_metadata
  - finalize_MEDS_data
"""

STAGE_SCENARIOS = [
    "shard_events/default",
    "split_and_shard_subjects/default",
    "convert_to_subject_sharded/default",
    "convert_to_MEDS_events/default",
    "merge_to_MEDS_cohort/default",
    "extract_code_metadata/default",
    "finalize_MEDS_metadata/default",
    "finalize_MEDS_data/default",
]


def test_pipeline():
    pipeline_tester(
        pipeline_yaml=PIPELINE_YAML,
        stage_runner_yaml=None,
        stage_scenario_sequence=STAGE_SCENARIOS,
    )
