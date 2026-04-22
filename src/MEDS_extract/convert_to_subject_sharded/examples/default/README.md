Regroups the row-chunked sub-shards (from `shard_events`) by subject, using the
`metadata/.shards.json` partition written by `split_and_shard_subjects`. For each
`(split, table)` pair, every matching sub-shard is scanned, filtered to the split's subject
list (through the table's `subject_id` expression), joined to any referenced table via
`_table.join`, and written to `data/<split>/<shard>/<table>.parquet`. Rows are preserved
unchanged — only re-organized — so this stage outputs raw source columns, not MEDS events.
