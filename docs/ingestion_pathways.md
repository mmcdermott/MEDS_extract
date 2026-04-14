# Ingestion Pathways & File Layouts

This page documents how data flows through the MEDS-Extract pipeline —
specifically **what file layout each stage reads** and **what layout each
stage writes**. It's intended as a design reference: before changing how
stages consume or produce files, update this page so everyone's on the
same page, then change the code.

Most production users will run the stages in order (top to bottom in the
table below) and never need to think about the intermediate layouts. But
the pipeline supports several **entry points** — places where a user can
plug in pre-processed data and skip earlier stages — and those are only
safe if we're precise about the file-layout contracts.

## Stages at a glance

| Stage                                           | Consumes                                             | Produces                                                                            | File reader                                                                                                                                          |
| ----------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `shard_events`                                  | raw user data (`parquet` / `par` / `csv` / `csv.gz`) | row-subsharded parquet                                                              | `MEDS_extract.io.resolve_source_files` + `scan_source` (via `messy_cfg.needed_source_columns()`)                                                     |
| `split_and_shard_subjects`                      | row-subsharded parquet                               | JSON shards map (no parquet output)                                                 | `TableConfig.scan(dir)` — uses `resolve_source_files` under the hood, applies joins                                                                  |
| `convert_to_subject_sharded`                    | row-subsharded parquet + shards map                  | *see [Known issue](#known-issue-convert_to_subject_sharded-collapses-chunks) below* | `TableConfig.source_files(dir)` + `scan_source`, applies joins via `JoinConfig.apply`                                                                |
| `convert_to_MEDS_events`                        | subject-filtered parquet (output of previous stage)  | per-`(split, prefix)` MEDS-format parquet                                           | `TableConfig.source_files(dir / split)` + `scan_source`; **no join** (already materialized upstream)                                                 |
| `extract_code_metadata`                         | raw metadata files + MEDS events                     | `codes.parquet` per worker, then a merged `codes.parquet`                           | `resolve_source_files` + `scan_source` for metadata files; direct `rglob("*.parquet")` for event files (internal layout only)                        |
| `merge_to_MEDS_cohort`                          | per-`(split, prefix)` MEDS events                    | one MEDS parquet per split                                                          | its own helper `merge_subdirs_and_sort` — reads `{sp_dir}/{prefix}.parquet` directly by prefix list (does **not** go through `resolve_source_files`) |
| `finalize_MEDS_data` / `finalize_MEDS_metadata` | per-split MEDS parquet                               | MEDS-schema-validated parquet                                                       | delegates to `MEDS_transforms`, no in-repo file I/O                                                                                                  |

## Layout conventions

Two layouts are supported by `resolve_source_files`:

### Bare file

```text
{dir}/{prefix}.{parquet,par,csv.gz,csv}
```

One file per table prefix, any of the four supported formats. Typical for:

- Raw user data entering `shard_events`.
- Subject-sharded-per-split output (one file per `{split}/{prefix}`).

### Sub-sharded directory

```text
{dir}/{prefix}/*.{parquet,par,csv.gz,csv}
```

Multiple files under a `{prefix}/` directory, any of the supported formats.
`resolve_source_files` globs one level deep and does **not** recurse. This is
what `shard_events` produces as output (many row-chunk files per prefix) and
what downstream stages consume.

The two layouts are **mutually exclusive** — if both match simultaneously
(e.g., both `labs.parquet` and `labs/shard_0.parquet` exist under the same
directory), `resolve_source_files` raises `ValueError`. Ambiguity is never
intentional.

### Nested prefixes

`prefix` values may contain slashes (e.g. `hosp/patients`). That's just a
path component — `{dir}/hosp/patients.parquet` works, no recursive walking
happens. A nested prefix under the sub-sharded layout looks like
`{dir}/hosp/patients/shard_0.parquet`.

## End-to-end layout walk-through

Using the `example/raw_data` dataset with `row_chunksize=20`:

### Raw input (source for `shard_events`)

```text
raw_data/
  patients.csv
  labs_vitals.csv
  medications.csv
  diagnoses.csv
  stays.csv               ← join target referenced by labs_vitals._table.join
  lab_descriptions.csv    ← metadata file, consumed by extract_code_metadata
  medication_classes.csv
```

Each top-level prefix in `event_cfg.yaml` (and each join target) resolves to
a single bare file here. `stays` is a join target (not a top-level table),
but it still appears in `needed_source_columns()` because `labs_vitals` pulls
columns from it.

### After `shard_events`

```text
data/
  patients/[0-10).parquet
  labs_vitals/[0-20).parquet
  labs_vitals/[20-40).parquet
  labs_vitals/[40-60).parquet
  labs_vitals/[60-80).parquet
  labs_vitals/[80-86).parquet
  medications/[0-12).parquet
  diagnoses/[0-12).parquet
  stays/[0-10).parquet    ← join target, still present
```

Each raw file is row-chunked into the sub-sharded directory layout.
**Multi-file output per prefix is the normal case**, not an edge case.
Downstream stages globbing `{prefix}/*.parquet` pick up all chunks.

If the user pre-sharded their raw input — for example, supplying
`patients/shard_a.parquet` and `patients/shard_b.parquet` rather than a single
`patients.csv` — `shard_events` row-chunks each input file independently and
prefixes the output chunk names with the source stem to avoid collisions
(`patients/shard_a_[0-3).parquet`, `patients/shard_b_[0-3).parquet`).

### After `split_and_shard_subjects`

```text
metadata/.shards.json       ← JSON: {split_name/shard_idx: [subject_ids]}
```

No parquet output — this is a metadata-only stage. It reads every prefix's
sub-sharded parquet (applying joins as needed), collects unique subject IDs
via each table's `subject_id_polars_expr`, and writes the final shards map.

### After `convert_to_subject_sharded`

> **⚠️ Known issue** — see [below](#known-issue-convert_to_subject_sharded-collapses-chunks).
> The current behavior collapses all row-chunks into one output file per
> `(split, prefix)`, which is a regression from the pre-2025-05-08 design.

Currently produces:

```text
data/
  train/0/patients.parquet
  train/0/labs_vitals.parquet
  train/0/medications.parquet
  train/0/diagnoses.parquet
  train/1/...
  tuning/0/...
  held_out/0/...
```

One file per `(split, prefix)`. Note that:

- `stays` is NOT in the output. Only tables with events produce output — the
    stays columns needed by `labs_vitals` are **materialized into
    `labs_vitals.parquet`** via the join at read time.
- The sub-sharding from `shard_events` is lost. All row-chunks for a prefix
    are concatenated into a single output file.

**Historical (pre-2025-05-08) behavior** — the original design produced:

```text
data/
  train/0/patients/[0-10).parquet           ← row-chunk preserved
  train/0/labs_vitals/[0-20).parquet
  train/0/labs_vitals/[20-40).parquet
  ...
```

One output file **per input row-chunk**, preserving chunk boundaries. This is
what we should restore — see the fix discussion below.

### After `convert_to_MEDS_events`

```text
data/
  train/0/patients.parquet                  ← MEDS-schema: subject_id, code, time, ...
  train/0/labs_vitals.parquet
  train/0/medications.parquet
  train/0/diagnoses.parquet
  train/1/...
  tuning/0/...
  held_out/0/...
  event_conversion_config.yaml              ← verbatim copy of source config
```

Same layout as the previous stage, but now in MEDS schema (subject_id, code,
time, numeric_value, text_value, source_block, code_components).

### After `merge_to_MEDS_cohort`

```text
data/
  train/0.parquet         ← all prefixes merged, sorted by (subject_id, time)
  train/1.parquet
  tuning/0.parquet
  held_out/0.parquet
```

One MEDS parquet file per split. All tables' events are vertically concatenated
(diagonally — different events have different extra columns) and sorted.

## Entry points (where you can plug in pre-processed data)

The stages form a chain, but you can enter at any of these points if your
data is already in the right format:

### 1. `shard_events` (normal entry)

You have raw tables as csv/parquet and want the full pipeline.

**Skip this stage when**: your data is already row-sharded parquet. Enter
at stage 2.

### 2. `split_and_shard_subjects`

You have row-sharded parquet at `{data}/{prefix}/*.parquet` and want MEDS-Extract
to determine the subject splits.

**Skip this stage when**: you have a pre-computed `.shards.json` from another
source (e.g., a sibling run or external train/test split tool). Enter at
stage 3.

### 3. `convert_to_subject_sharded`

You have row-sharded parquet AND a shards map, and want subject-filtered
output organized by split.

**Skip this stage when**: you've already done subject sharding elsewhere
(unusual). Enter at stage 4.

### 4. `convert_to_MEDS_events`

You have subject-sharded parquet (one file per split, per prefix, **with joins
already materialized**) and just want the event-schema transformation.

**Constraint**: because the stage doesn't re-apply joins, any columns
referenced by event expressions that come from a join target must already be
present in the input parquet.

### 5. `merge_to_MEDS_cohort`

You have per-`(split, prefix)` MEDS-format parquet and want them merged per
split.

## `extract_code_metadata` is mostly independent

`extract_code_metadata` reads raw metadata files (via `resolve_source_files`,
supporting the same layouts as the main data path) and the event output from
`convert_to_MEDS_events`. It's orthogonal to the main ingestion flow —
parallelizable, and typically run alongside the main stages rather than in
strict sequence.

## Known issue: `convert_to_subject_sharded` collapses chunks

**Status**: Behavior regression introduced in commit `ab5ffcf` on 2025-05-08.

**What the current behavior does**: For each `(split, prefix)` pair, reads
every row-chunk file under `{input_dir}/{prefix}/`, concatenates them, applies
joins and the subject filter, and writes one output file at
`{output_dir}/{split}/{prefix}.parquet`.

**Why it's a problem**:

1. **Lost memory bound.** Row-chunking exists so the pipeline can process
    arbitrarily large tables with bounded memory per worker. Concatenating all
    chunks in one polars lazy query *usually* streams through memory
    correctly — polars pushes the subject filter into each scan — but this
    guarantee is **not** preserved when a join is involved. Verified query
    plan for `labs_vitals ⋈ stays`, filter on subject_id:

    ```text
    INNER JOIN:
    LEFT PLAN ON: [col("stay_id")]
      UNION
        Parquet SCAN [vitals_0.parquet]   PROJECT */2 COLUMNS   ← no filter
        Parquet SCAN [vitals_1.parquet]   PROJECT */2 COLUMNS   ← no filter
        Parquet SCAN [vitals_2.parquet]   PROJECT */2 COLUMNS   ← no filter
      END UNION
    RIGHT PLAN ON: [col("stay_id")]
      Parquet SCAN [stays.parquet]
      SELECTION: col("subject_id").is_in(...)                   ← pushed down ✓
    END INNER JOIN
    ```

    The filter pushes to the `stays` side (good) but **not** the `vitals`
    side (bad). All vitals shards are read in full before joining and
    filtering. Polars' streaming execution engine typically handles this, but
    "relies on polars doing the right thing" is exactly the guarantee
    row-chunking was supposed to make unnecessary.

2. **Lost parallelism.** The pre-regression design let workers pick off
    `(split, prefix, shard)` tuples independently. The current design is
    `(split, prefix)` — coarser, less parallel.

3. **It's a regression.** The pre-2025-05-08 version of this stage (then
    named `convert_to_sharded_events.py`) processed per-shard with output
    layout `{split}/{prefix}/{shard_name}.parquet`. The 2025-05-08 restructure
    for the MEDS-transforms upgrade lost this without discussion or a note in
    the commit message.

**Proposed fix**:

Restore per-shard iteration with per-shard output:

```text
convert_to_subject_sharded output (proposed):
  train/0/patients/[0-10).parquet
  train/0/labs_vitals/[0-20).parquet
  train/0/labs_vitals/[20-40).parquet
  ...
```

This cascades to `convert_to_MEDS_events` (which would also process
per-shard and produce sub-sharded output) and finally `merge_to_MEDS_cohort`
(which already aggregates across all shards when building the per-split
output). `resolve_source_files` already supports the sub-sharded directory
layout so the mid-pipeline reads are fine; `merge_to_MEDS_cohort` is the
only stage that hard-codes the bare-file layout and would need updating.

**Scope**: deferred to a follow-up. The regression is old enough (~11
months) that the loss of per-shard processing isn't a recent surprise, and
the larger question of how ingestion pathways should work across stages
(entry points, layout contracts, per-shard vs. collapsed output) warrants
its own discussion. Tracked in
[#76 — Redesign ingestion pathways](https://github.com/mmcdermott/MEDS_extract/issues/76).
