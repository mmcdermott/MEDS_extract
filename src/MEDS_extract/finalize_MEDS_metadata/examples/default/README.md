Takes the cohort's aggregated `metadata/codes.parquet` plus the `metadata/.shards.json` split
map and produces three schema-compliant MEDS metadata files:

- `metadata/codes.parquet` — validated against the MEDS code metadata schema (required `code`
    column; `description` / `parent_codes` typed to the schema when present, not added if absent).
- `metadata/subject_splits.parquet` — derived from the shards map (`Int64 subject_id`, `str split`).
- `metadata/dataset.json` — dataset-level info (name, version, ETL version, MEDS version,
    and a `created_at` timestamp; not diffed by the stage test since it changes each run). It also
    records the data's extension columns: those assigned in the stage config to the MEDS categories
    (`raw_source_id_columns`, `code_modifier_columns`, `additional_value_modality_columns`,
    `site_id_columns`), and all others under `other_extension_columns`. The optional `license`,
    `location_uri` and `description_uri` are included when set.

This stage should almost always be the last metadata stage in an extraction pipeline.
