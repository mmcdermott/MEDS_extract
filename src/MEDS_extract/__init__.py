from importlib.metadata import PackageNotFoundError, version

from .convert_to_MEDS_events import stage as convert_to_MEDS_events
from .convert_to_subject_sharded import stage as convert_to_subject_sharded
from .extract_code_metadata import stage as extract_code_metadata
from .finalize_MEDS_data import stage as finalize_MEDS_data
from .finalize_MEDS_metadata import stage as finalize_MEDS_metadata
from .merge_to_MEDS_cohort import stage as merge_to_MEDS_cohort
from .shard_events import stage as shard_events
from .split_and_shard_subjects import stage as split_and_shard_subjects

__package_name__ = "MEDS_extract"
try:
    __version__ = version(__package_name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
