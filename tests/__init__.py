import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

extraction_root = root / "src" / "MEDS_extract"

SHARD_EVENTS_SCRIPT = "MEDS_extract-shard_events"
SPLIT_AND_SHARD_SCRIPT = "MEDS_extract-split_and_shard_subjects"
CONVERT_TO_SUBJECT_SHARDED_SCRIPT = "MEDS_extract-convert_to_subject_sharded"
CONVERT_TO_MEDS_EVENTS_SCRIPT = "MEDS_extract-convert_to_MEDS_events"
MERGE_TO_MEDS_COHORT_SCRIPT = "MEDS_extract-merge_to_MEDS_cohort"
EXTRACT_CODE_METADATA_SCRIPT = "MEDS_extract-extract_code_metadata"
FINALIZE_DATA_SCRIPT = "MEDS_extract-finalize_MEDS_data"
FINALIZE_METADATA_SCRIPT = "MEDS_extract-finalize_MEDS_metadata"
