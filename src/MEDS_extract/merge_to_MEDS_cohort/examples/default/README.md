Concatenates the per-table event parquets (`convert_to_MEDS_events` output) within each
`(split, shard)` into a single file at `data/<split>/<shard>.parquet`. Rows are sorted
so null `time` values (static events like `EYE_COLOR`) precede real timestamps per subject,
and the resulting schema unions all `code_components` struct fields across tables.
