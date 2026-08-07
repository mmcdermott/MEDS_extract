Applies the MESSY event config to each subject-sharded source table, emitting one row per
configured `(table, event)` definition per source row. Each row gets:

- `subject_id` — resolved via the table's `_defaults.subject_id` expression (`$MRN`, `$patient_id`)
- `code` — from the event's `code` dftly expression (e.g. `f"EYE_COLOR//{$eye_color}"`, `$test_name`)
- `code_components` — per-row struct preserving the raw source columns referenced by `code`
- `time` — from the event's `time` expression (null for static events like eye_color)
- `numeric_value` (labs only) — from `$result`
- `source_block` — `"{table}/{event}"` provenance string

Writes one parquet per `(split, table)` pair at `data/<split>/<shard>/<table>.parquet`.
