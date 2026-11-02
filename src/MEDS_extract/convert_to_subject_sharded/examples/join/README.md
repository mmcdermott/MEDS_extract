Demonstrates `_table.joins` — the `vitals` table has no `subject_id` column of its own, so
`_table.joins.stays.key: stay_id, cols: [subject_id]` brings in the `subject_id` from the
`stays` table via a left join on `stay_id`. The stage materializes the joined frame, filters
to the split's subjects, and writes per-subject shards.
