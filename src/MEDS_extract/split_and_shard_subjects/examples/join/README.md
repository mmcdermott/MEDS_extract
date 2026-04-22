Demonstrates split-and-shard over a joined table: the `vitals` table exposes `subject_id` only
via a `_table.join` on `stays`. The split machinery follows that join to discover unique
subject IDs before assigning splits.
