import polars as pl
from meds import code_field, subject_id_field, time_field

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
