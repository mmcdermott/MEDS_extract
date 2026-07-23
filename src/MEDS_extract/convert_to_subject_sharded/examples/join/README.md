Demonstrates `_table.join` — the `vitals` table has no `subject_id` column of its own, so
`_table.join.stays.key: stay_id, cols: [subject_id]` brings in the `subject_id` from the
`stays` table via an inner join on `stay_id`. The stage materializes the joined frame, filters
to the split's subjects, and writes per-subject shards. Note the provenance anchor columns
(`__row_idx__` / `__source_file__`): the main (`vitals`) side's pass through, while the join
target's are dropped — join-side provenance is out of scope for the base provenance
implementation (issue #132).
