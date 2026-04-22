Takes the cohort's aggregated `metadata/codes.parquet` plus the `metadata/.shards.json` split
map and produces three schema-compliant MEDS metadata files:

- `metadata/codes.parquet` — validated against the MEDS code metadata schema (mandatory `code`,
    `description`, `parent_codes` columns).
- `metadata/subject_splits.parquet` — derived from the shards map (`Int64 subject_id`, `str split`).
- `metadata/dataset.json` — dataset-level info (name, version, ETL version, MEDS version,
    and a `created_at` timestamp; not diffed by the stage test since it changes each run).

This stage should almost always be the last metadata stage in an extraction pipeline.
