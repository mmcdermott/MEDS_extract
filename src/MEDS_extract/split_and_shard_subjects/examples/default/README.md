Scans unique `subject_id` values across the sub-sharded input tables and partitions them into
`train`/`tuning`/`held_out` splits, then into shards of at most `n_subjects_per_shard` subjects
per shard. The resulting split/shard map is written as `metadata/.shards.json`, which all
downstream stages consume.

Since the dataset here has 4 subjects and the `50/25/25` split assigns subjects
deterministically under `seed: 1`, each split gets at most one shard.
