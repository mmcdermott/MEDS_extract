Row-chunks each raw source table (`patients.csv`, `labs.csv`) into fixed-size parquet sub-shards
named `[start-end).parquet`. With `row_chunksize: 2`, two 4-row input tables become four
2-row output chunks. Column names are preserved from the source — no MEDS mapping happens
here; that's the job of `convert_to_MEDS_events`.

The `event_conversion_config_fp` (wired via `pipeline_cfg.yaml`) tells the stage which input
prefixes to sub-shard.
