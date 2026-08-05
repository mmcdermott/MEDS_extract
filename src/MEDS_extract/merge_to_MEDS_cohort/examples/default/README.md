Concatenates the per-table event parquets (`convert_to_MEDS_events` output) within each
`(split, shard)` into a single file at `data/<split>/<shard>.parquet`. Rows are sorted
so null `time` values (static events like `EYE_COLOR`) precede real timestamps per subject.
The internal `code_components` struct is dropped during the merge (#254): unifying the
per-table structs into one field-union superstruct is catastrophically memory-expensive,
and metadata extraction reads the pre-merge per-table events, so nothing downstream of
the merge consumes the column.
