from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files

import polars as pl
from meds import code_field, subject_id_field, time_field

__package_name__ = "MEDS_extract"
try:
    __version__ = version(__package_name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

CONFIG_YAML = files(__package_name__).joinpath("configs/_extract.yaml")

MANDATORY_COLUMNS = [subject_id_field, time_field, code_field, "numeric_value"]

MANDATORY_TYPES = {
    subject_id_field: pl.Int64,
    time_field: pl.Datetime("us"),
    code_field: pl.String,
    "numeric_value": pl.Float32,
    "categorical_value": pl.String,
    "text_value": pl.String,
}

DEPRECATED_NAMES = {
    "numerical_value": "numeric_value",
    "categoric_value": "categoric_value",
    "category_value": "categoric_value",
    "textual_value": "text_value",
    "timestamp": "time",
    "patient_id": subject_id_field,
}

INFERRED_STAGE_KEYS = {
    "is_metadata",
    "data_input_dir",
    "metadata_input_dir",
    "output_dir",
    "reducer_output_dir",
}

# TODO(mmd): This should really somehow be pulled from MEDS.
MEDS_METADATA_MANDATORY_TYPES = {
    "code": pl.String,
    "description": pl.String,
    "parent_codes": pl.List(pl.String),
}

MEDS_DATA_MANDATORY_TYPES = {c: MANDATORY_TYPES[c] for c in MANDATORY_COLUMNS}
