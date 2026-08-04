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

| Stage                                           | Consumes                                                 | Produces                                                                                                            | File reader                                                                                                                   |
| ----------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `convert_to_parquet`                            | raw user data (`parquet` / `par` / `csv` / `csv.gz`)     | one projected parquet per input file, same relative layout                                                          | `MEDS_extract.io.resolve_source_files` + `convert_csv_to_parquet` via `messy_cfg.needed_source_columns()`                     |
| `split_and_shard_subjects`                      | normalized parquet                                       | JSON shards map (no parquet output)                                                                                 | `TableConfig.scan(dir)` — uses `resolve_source_files`, applies joins                                                          |
| `convert_to_subject_sharded`                    | normalized parquet + shards map                          | per-`shard` subject-sharded copy of the same tables (same schema as input, just filtered by shard)                  | `TableConfig.source_files(dir)` + `scan_source`, applies joins via `JoinConfig.apply`                                         |
| `convert_to_MEDS_events`                        | subject-sharded source tables (output of previous stage) | same layout, same paths — just the row *schema* changes to MEDS events                                              | `TableConfig.source_files(dir / shard)` + `scan_source`; **no join** (already materialized upstream)                          |
| `extract_code_metadata`                         | raw metadata files + MEDS events                         | one partial `{prefix}_{entry}.parquet` per `(metadata prefix, config entry)` pair, then one reduced `codes.parquet` | `resolve_source_files` + `scan_source` for metadata files; direct `rglob("*.parquet")` for event files (internal layout only) |
| `merge_to_MEDS_cohort`                          | per-`(shard, prefix)` MEDS events                        | one MEDS parquet **per shard** (e.g. `train/0.parquet`), across all table sources                                   | its own `merge_subdirs_and_sort` helper — reads `{sp_dir}/{prefix}.parquet` by explicit prefix list                           |
| `finalize_MEDS_data` / `finalize_MEDS_metadata` | per-shard MEDS parquet                                   | MEDS-schema-validated parquet                                                                                       | delegates to `MEDS_transforms`, no in-repo file I/O                                                                           |

Three things worth calling out, because they're easy to get wrong:

- **"Shard" here is the unit the whole pipeline parallelizes over.** The
    shards map file (`.shards.json`) has keys like `train/0`, `tuning/0`,
    `held_out/0` — each key is a single shard, and the slash-separated
    first component happens to be the split name. No stage ever aggregates
    at the split level; aggregation is at the shard level (across tables
    within a shard).
- **`convert_to_MEDS_events` is the only stage that preserves exact
    relative paths.** Its input and output live at the same relative
    paths (under different roots); it just rewrites the row contents
    from "source columns" to "MEDS events". Every other data stage
    reshapes the layout somehow (normalizing format, adding a shard dimension,
    aggregating across prefixes).
- **No stage overwrites its input files.** By the MEDS_transforms
    pipeline convention, every stage gets a distinct `output_dir` from
    its `data_input_dir` — see the per-stage table above. Reads and
    writes never collide, even when relative paths match exactly (as in
    `convert_to_MEDS_events`), because the roots are always different.

## Layout conventions

Two layouts are supported by `resolve_source_files`:

### Bare file

```text
{dir}/{prefix}.{parquet,par,csv.gz,csv}
```

One file per table prefix, any of the four supported formats. Typical for:

- Raw user data entering `convert_to_parquet`.
- `convert_to_subject_sharded` / `convert_to_MEDS_events` output, one
    file per `(shard, prefix)` pair.

### Sub-sharded directory

```text
{dir}/{prefix}/*.{parquet,par,csv.gz,csv}
```

Multiple files under a `{prefix}/` directory, any of the supported formats.
`resolve_source_files` globs one level deep and does **not** recurse. This is
what a pre-sharded raw dataset looks like, and what `convert_to_parquet` preserves
(one output per input file) when it normalizes such a table.

The two layouts are **mutually exclusive** — if both match simultaneously
(e.g., both `labs.parquet` and `labs/shard_0.parquet` exist under the same
directory), `resolve_source_files` raises `ValueError`. Ambiguity is never
intentional.

### Nested prefixes

`prefix` values may contain slashes (e.g. `hosp/patients`). That's just a
path component — `{dir}/hosp/patients.parquet` works, no recursive walking
happens. A nested prefix under the sub-sharded layout looks like
`{dir}/hosp/patients/shard_0.parquet`.

## Raw input layouts users actually have

In the wild, raw source data shows up in several layouts. Only the first
is supported end-to-end by the current implementation; the rest are real
requirements that #76 needs to address.

### (a) Flat, nested by table path — supported

```text
hosp/patients.csv
hosp/events.csv
icu/stays.csv
transfers.csv
```

Each table is a single file. The prefix is the path with the extension
stripped — `hosp/patients`, `hosp/events`, `icu/stays`, `transfers`.
`resolve_source_files(raw, "hosp/patients")` finds this naturally because
the prefix string is just a path component.

**This is the only raw layout `convert_to_parquet` currently handles.**

### (b) Pre-subject-sharded, same table structure per shard — NOT supported

```text
0/hosp/patients.csv
0/hosp/events.csv
0/icu/stays.csv
1/hosp/patients.csv
1/hosp/events.csv
...
```

The table hierarchy is identical to (a), but there's a shard-index
directory *above* it. The prefixes from the config are still
`hosp/patients`, `hosp/events`, etc. Each shard directory is the same
table layout.

**Current state**: `resolve_source_files(raw, "hosp/patients")` fails —
it looks at `{raw}/hosp/patients/*.parquet` or `{raw}/hosp/patients.csv`,
neither of which exists. This form is only reachable if the user enters
at `split_and_shard_subjects` **and** does file-layout massaging
themselves, which isn't documented anywhere.

### (c) Pre-sharded with shard index redundantly in both directory and filename — NOT supported

```text
0/transfers_0.csv
1/transfers_1.csv
```

The prefix is `transfers`, but each shard directory contains exactly
one file whose name redundantly encodes the same shard index. This is
an unfortunate layout that still shows up in real datasets.

**Current state**: not supported by `resolve_source_files` under any
interpretation. There's no way to express "prefix `transfers`, files
found at `*/transfers_*.csv`" with the current layout model.

### What this means for the current implementation

Of the three layouts above, only (a) works with `convert_to_parquet` today.
Layouts (b) and (c) need one of:

1. Extending `resolve_source_files` with a layout hint that says "look
    for shard-prefixed subdirectories", or
2. Declaring the shard axis explicitly in the MESSY config and using
    it to drive layout resolution, or
3. Preprocessing raw data into layout (a) before running the pipeline
    (the user's responsibility, documented but not automated).

Design discussion and implementation for (b) and (c) is tracked in
[#76](https://github.com/mmcdermott/MEDS_extract/issues/76).

## End-to-end layout walk-through

Using the `example/raw_data` dataset:

### Raw input (source for `convert_to_parquet`)

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

Each top-level prefix in `example/messy.yaml` (and each join target) resolves to
a single bare file — layout (a) above. `stays` is a join target (not a
top-level table), but it still appears in `needed_source_columns()` because
`labs_vitals` pulls columns from it.

### After `convert_to_parquet`

```text
{output_dir}/convert_to_parquet/
  patients.parquet
  labs_vitals.parquet
  medications.parquet
  diagnoses.parquet
  stays.parquet           ← join target, still present
```

One output per **input file**, at the same relative position, with the columns
the MESSY config references projected in. A single raw `labs_vitals.csv` becomes
a single `labs_vitals.parquet` — there is no row-range chunking, so downstream
globbing `{prefix}/*.parquet` simply finds one file.

A source that is *already* parquet is hardlinked rather than rewritten: downstream
scans push projection into the parquet reader themselves, so a rewrite would cost a
full pass to save nothing.

If the user pre-sharded their raw input — for example, supplying
`patients/shard_a.parquet` and `patients/shard_b.parquet` rather than a single
`patients.csv` — the sharding is preserved exactly, one output per input file
(`patients/shard_a.parquet`, `patients/shard_b.parquet`).

### After `split_and_shard_subjects`

```text
metadata/.shards.json       ← JSON: {shard_id: [subject_ids]}
```

No parquet output — this is a metadata-only stage. It reads every prefix's
sub-sharded parquet (applying joins as needed), collects unique subject IDs
via each table's `subject_id_polars_expr`, and writes the final shards map.

Shard IDs in the JSON look like `train/0`, `train/1`, `tuning/0`,
`held_out/0`. They're used as paths under the data directories by all
subsequent stages.

### After `convert_to_subject_sharded`

**Intent** (per the stage's design): produce the same set of source tables,
each filtered down to a single shard's subjects. The output is a
subject-sharded *copy* of the input — same prefixes, same source columns,
just filtered by shard:

```text
data/
  train/0/patients.parquet       ← same schema as raw patients, filtered to shard train/0
  train/0/labs_vitals.parquet    ← same schema as raw labs_vitals, PLUS joined cols from stays
  train/0/medications.parquet
  train/0/diagnoses.parquet
  train/1/...
  tuning/0/...
  held_out/0/...
```

Each file is the source schema (not yet MEDS events), filtered to the
subjects in that shard. Where a table has a `_table.join` configured
(`labs_vitals ⋈ stays` in the example), the joined columns are
materialized into the output so downstream stages don't need to re-apply
the join.

Note: `stays` is NOT in the output because it's only a join target, not a
table with events. It doesn't exist on its own in the MESSY config's
top-level `tables` list.

**Known inefficiency**: the stage reads every file for a prefix once per
subject shard, so its cost is O(n_shards x table_size) regardless of how the
table is laid out on disk. Row-chunking never helped here either — it read all
chunks for every shard. Fixing it means a single pass that partitions by subject
shard; tracked in #76.

### After `convert_to_MEDS_events`

```text
data/
  train/0/patients.parquet                  ← MEDS schema: subject_id, code, time, ...
  train/0/labs_vitals.parquet
  train/0/medications.parquet
  train/0/diagnoses.parquet
  train/1/...
  tuning/0/...
  held_out/0/...
  messy.yaml                                ← copy of source config (reserved `sources:` blocks redacted)
```

**Same paths as the previous stage.** This stage is a schema-only
transformation — it rewrites each file in place (from the perspective of
the directory tree) by reading the source-schema rows and emitting MEDS
event rows (`subject_id`, `code`, `time`, `numeric_value`, `text_value`,
`source_block`, `code_components`).

### After `merge_to_MEDS_cohort`

```text
data/
  train/0.parquet         ← all prefixes merged for shard train/0, sorted by (subject_id, time)
  train/1.parquet         ← same for shard train/1
  tuning/0.parquet
  held_out/0.parquet
```

**One MEDS parquet file per shard**, not per split. `merge_to_MEDS_cohort`
uses `shard_iterator_by_shard_map` to iterate each entry in the shards
map file and merges the per-prefix files under `{shard_dir}/` into a
single `{shard_id}.parquet`. Splits are implicit in the shard IDs
(`train/0`, `train/1` both live under `train/`), but there's no
merge step at the split level — downstream consumers that want
split-level views open the per-shard files as a collection.

## Entry points (where you can plug in pre-processed data)

The stages form a chain, but you can enter at any of these points if your
data is already in the right format:

### 1. `convert_to_parquet` (normal entry)

You have raw tables as csv/parquet in layout (a) above and want the full
pipeline.

**Skip this stage when**: your data is already row-sharded parquet. Enter
at stage 2.

### 2. `split_and_shard_subjects`

You have row-sharded parquet at `{data}/{prefix}/*.parquet` and want
MEDS-Extract to determine the subject shards.

**Skip this stage when**: you have a pre-computed `.shards.json` from
another source. Enter at stage 3.

### 3. `convert_to_subject_sharded`

You have row-sharded parquet AND a shards map, and want subject-filtered
output organized by shard.

**Skip this stage when**: you already have per-shard source tables
(layout (b) above, or the output of this stage from a sibling run).
Enter at stage 4.

### 4. `convert_to_MEDS_events`

You have per-`(shard, prefix)` source parquet (with joins already
materialized) and just want the event-schema transformation.

**Constraint**: the stage doesn't re-apply joins, so any columns
referenced by event expressions that come from a join target must already
be present in the input parquet.

### 5. `merge_to_MEDS_cohort`

You have per-`(shard, prefix)` MEDS-format parquet and want them merged
per shard.

## `extract_code_metadata` is mostly independent

`extract_code_metadata` reads raw metadata files (via `resolve_source_files`,
supporting the same layouts as the main data path) and the event output from
`convert_to_MEDS_events`. It's orthogonal to the main ingestion flow —
parallelizable, and typically run alongside the main stages rather than in
strict sequence.

## Open design questions (tracked in #76)

1. **`convert_to_subject_sharded` re-reads each table per subject shard.**
    For every `(shard, prefix)` pair it reads all of the prefix's files,
    concatenates, filters to the shard's subjects, and writes one file — so its
    cost is O(n_shards x table_size). Query plan for `labs_vitals ⋈ stays` with a
    `subject_id` filter:

    ```text
    INNER JOIN:
    LEFT PLAN ON: [col("stay_id")]
      Parquet SCAN [labs_vitals.parquet]   PROJECT */2 COLUMNS   ← no filter
    RIGHT PLAN ON: [col("stay_id")]
      Parquet SCAN [stays.parquet]
      SELECTION: col("subject_id").is_in(...)                   ← pushed down ✓
    END INNER JOIN
    ```

    The filter pushes through to `stays` but not to `labs_vitals`, because the
    subject column arrives via the join. Polars' streaming engine copes in
    practice; the structural fix is a single pass that partitions by subject
    shard rather than one pass per shard.

2. **Layouts (b) and (c) are unsupported.** Pre-subject-sharded and
    shard-index-in-filename raw layouts (see above) can't currently be
    consumed by `convert_to_parquet` or `split_and_shard_subjects`.

3. **`merge_to_MEDS_cohort` bypasses the unified reader.** If upstream
    stages ever produce multi-file output per prefix, this stage breaks
    silently because it hard-codes `{sp_dir}/{prefix}.parquet`.

4. **`convert_to_MEDS_events` assumes pre-materialized joins** without
    validating. If a user enters at this stage with un-joined data, event
    extraction fails with a column-not-found error at collect time
    instead of a clear "you need to run convert_to_subject_sharded first"
    message.

5. **`convert_to_parquet` silently accepts sub-sharded input** via the
    `{stem}_` disambiguation. This is technically a corner but users with
    pre-sharded data should probably be told to enter at
    `split_and_shard_subjects` instead.
