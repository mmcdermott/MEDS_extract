# Migrating from MEDS_extract 0.6.x to 0.7.0

The 0.7.0 release is a deliberate breaking cut. Four areas change: the MESSY config key layout, null
handling in composite codes and event times, metadata extraction / `codes.parquet`, and raw-data
fetching (a first-class download layer replaces per-ETL `download.py` scripts). This guide walks every breaking change with
before/after snippets you can copy.

> **Scope**: 0.6.x (any of 0.6.0–0.6.2) → 0.7.0. If you're on 0.5.x or earlier, land the 0.6.0 migration
> first (notebook-driven `event_cfg.yaml` → dftly-native MESSY + Hydra stage DAG); that's orthogonal.

## At a glance

| Area                                 | Before (0.6.x)                                                                               | After (0.7.0)                                                                                                                                                                                       |
| ------------------------------------ | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `subject_id_col` / `subject_id_expr` | top-level table keys                                                                         | `_defaults.subject_id` (a dftly expression)                                                                                                                                                         |
| `transforms`                         | top-level table key                                                                          | `_table.cols`                                                                                                                                                                                       |
| `join`                               | top-level table key with `columns_from_right`                                                | `_table.join: {prefix: {key, cols}}`                                                                                                                                                                |
| `schema`                             | top-level table key (parsed, never used)                                                     | **removed**                                                                                                                                                                                         |
| Null component in a composite `code` | auto-filled with `"UNK"`, row kept                                                           | code is null → **row dropped**; opt back in per component with `?? 'UNK'`                                                                                                                           |
| Unparsable `time` values             | strict-cast `""` silently dropped; lenient junk kept with null time                          | strict cast **errors**; lenient junk **drops the row**, with a per-event WARNING                                                                                                                    |
| `_metadata` blocks                   | raw-column shorthand; implicit all-component join; `_match_on`; `parent_codes` matcher lists | a dftly program over the metadata table (bare strings are **literals** — write `$col`); join keys = produced component-named columns; `_match_on` **removed**; `parent_codes` = a dftly conditional |
| `_match_on` metadata joins           | joined against **all** events' codes; dtype-fragile                                          | replaced by producing key columns (below); joins scoped to the declaring event; keys dtype-normalized                                                                                               |
| `codes.parquet`                      | run-order-dependent schema/values; `*_right` merge forks                                     | deterministic byte-identical output; deduplicated values; stable schema                                                                                                                             |
| Multi-file source prefixes           | csv + parquet chunks silently unified                                                        | mixed csv/parquet chunks are an error                                                                                                                                                               |
| `shard_events.infer_schema_length`   | stage option (default: 10000 rows)                                                           | **removed** — CSV schemas are always inferred from the full file (the knob's only realistic use was "big enough to avoid mid-file type flips"; full-file inference makes that the only behavior)    |
| Raw-data fetching                    | hand-rolled `download.py` per ETL                                                            | MESSY `sources:` block + `meds-extract-download`                                                                                                                                                    |
| `cloud_io_storage_options`           | pipeline-level polars `storage_options` passthrough for cloud reads                          | **removed** — pipeline directories are local; fetch remote raw data first via `meds-extract-download`                                                                                               |
| Python floor                         | 3.12                                                                                         | **3.11** (relaxed, not raised)                                                                                                                                                                      |
| Dependency pins                      | `MEDS-transforms~=0.6.0`, `dftly>=0.1.2,<0.2`                                                | `MEDS-transforms>=0.6.7,<0.7`, `dftly>=0.5.0`                                                                                                                                                       |

## 1. MESSY config redesign

0.6.x accepted five ad-hoc top-level keys per table: `subject_id_col`, `subject_id_expr`, `transforms`,
`join`, `schema`. 0.7.0 unifies them under two clearly-prefixed structural keys — `_defaults` for
inherited fields, `_table` for whole-table modifications — and every other non-underscored key is an
event name.

### 1a. `subject_id_col` / `subject_id_expr` → `_defaults.subject_id`

**Before:**

```yaml
subject_id_col: patient_id

patients:
  dob:
    code: MEDS_BIRTH
    time: $dob::"%Y-%m-%dT%H:%M:%S"
```

**After:**

```yaml
_defaults:
  subject_id: $patient_id

patients:
  dob:
    code: MEDS_BIRTH
    time: $dob::"%Y-%m-%dT%H:%M:%S"
```

The value is now a dftly expression, so a hash is `$MRN` → `hash($MRN)`:

```yaml
# Before: subject_id_expr: "hash($MRN)"
_defaults:
  subject_id: hash($MRN)
```

Per-table overrides use a local `_defaults` block instead of a top-level key inside the table:

```yaml
# Before
labs_vitals:
  subject_id_col: patient_id
  lab: {code: ..., time: ...}

# After
labs_vitals:
  _defaults:
    subject_id: $patient_id
  lab: {code: ..., time: ...}
```

**What you must change:** rename the keys as above. If the column is literally named `subject_id`, you
can drop the field entirely — it defaults to reading the `subject_id` column.

### 1b. `transforms` → `_table.cols`

**Before:**

```yaml
hosp/patients:
  transforms:
    year_of_birth: $anchor_year - $anchor_age
  dob:
    code: MEDS_BIRTH
    time: $year_of_birth::year
```

**After:**

```yaml
hosp/patients:
  _table:
    cols:
      year_of_birth: $anchor_year - $anchor_age
  dob:
    code: MEDS_BIRTH
    time: $year_of_birth::year
```

**What you must change:** rename the key. New (additive): later `_table.cols` entries can reference
earlier ones, so chained derived-column idioms (pseudo-timestamps built in steps) no longer need
inlining or repetition.

### 1c. `join` → `_table.join` (and new syntax)

The old `join:` block took a flat dict with `input_prefix`, `left_on`, `right_on`, `columns_from_right`:

**Before:**

```yaml
labs_vitals:
  join:
    input_prefix: stays
    left_on: stay_id
    right_on: stay_id
    columns_from_right:
      - patient_id
  lab: {code: '...', time: '...'}
```

**After** — the joined table's prefix is the outer mapping key, and the inner block takes either `key:`
(same column on both sides) or `left_on:` + `right_on:`:

```yaml
labs_vitals:
  _table:
    join:
      stays:
        key: stay_id
        cols: [patient_id]
  lab: {code: '...', time: '...'}
```

Long form when the keys differ:

```yaml
_table:
  join:
    admissions:
      left_on: hadm_id
      right_on: admission_id
      cols: [dischtime]
```

**What you must change:** restructure each `join:` block by hand (it's too structurally different for a
mechanical rewrite). `cols` is required — a join that pulls in no columns is now a config error.

**New capability (not a migration requirement):** `cols` also accepts a `{name: aggregation}` mapping
(`min`/`max`/`sum`/`mean`/`count`), which reduces the joined table per key *before* the join — e.g.
`cols: {deathtime: min}` for earliest-death-time-per-subject, previously only expressible in pre-MEDS
Python. Aggregated joins log a use-with-care warning (conflicting values are resolved silently and
row-level provenance does not trace through the aggregation); see the README's *Aggregated joins*
section for a runnable example and the String-column caveats.

### 1d. `schema:` key removed entirely

The old top-level `schema:` key was dead code in 0.6.x — parsed but never consulted. Delete it. To cast
a column, use a `_table.cols` expression:

```yaml
# Before:
# schema:
#   result: Float64

# After:
_table:
  cols:
    result: $result::float64
```

### 1e. Mechanical transformation

For larger files, this sed pattern covers the common case:

```bash
sed -i '
  s/^\(\s*\)subject_id_col: *\(\S\+\)/\1_defaults:\n\1  subject_id: $\2/
  s/^\(\s*\)subject_id_expr: *\(.\+\)/\1_defaults:\n\1  subject_id: \2/
  s/^\(\s*\)transforms:/\1_table:\n\1  cols:/
' event_cfg.yaml
```

The `join:` block must be edited by hand.

## 2. Null handling: composite codes and event times

### Null components in composite codes

0.6.x silently rendered any null component of an interpolated `code` as the literal `"UNK"` and kept the
row. 0.7.0 makes that an explicit author choice: the composite `code` expression null-propagates — if
any component is null, the whole code is null, and since a MEDS `code` may never be null, the **row is
dropped**. To keep such rows, coalesce the specific components you want filled, using dftly 0.5's `??`
operator:

| MESSY `code`                                  | null component →                         |
| --------------------------------------------- | ---------------------------------------- |
| `f"{$itemid}//{$valueuom}"`                   | row dropped                              |
| `f"{$itemid ?? 'UNK'}//{$valueuom ?? 'UNK'}"` | filled with `UNK`, row kept              |
| `f"{$itemid}//{$valueuom ?? 'UNK'}"`          | unit filled; a null `itemid` still drops |

Single-quote the literal fallback *inside* the double-quoted f-string — `?? "UNK"` with double quotes
clashes with the f-string delimiter.

**What you must change:** audit every interpolated `code` in your MESSY file. For each component,
decide: should a null value drop the row (leave it bare) or be filled (add `?? 'UNK'` or another
literal)? To reproduce 0.6.x output exactly, coalesce every component with `?? 'UNK'`.

### Null and unparsable `time` values

0.7.0 drops null-time rows by filtering the **computed** `time` column, not by pre-filtering the raw
source columns it references. Three visible consequences:

- **Strict casts (`::"fmt"`) now error on unparsable non-null values.** Previously an empty-string
    `""` in a String time column was silently pre-filtered before the cast ever saw it; now a strict
    cast raises a polars `InvalidOperationError` naming the offending value(s). Strict means strict:
    if your data contains `""` (or other junk) time values you consider droppable, switch to a
    lenient cast (`::?"fmt"`) — or clean the data in pre-MEDS.

- **Lenient casts (`::?"fmt"`) now drop rows whose value parses under no format.** Previously such
    rows slipped past the raw-column pre-filter and leaked downstream with `time=null` — invalid
    MEDS output. Dropped rows are no longer silent: each event logs one WARNING with counts, so a
    mis-specified format that wipes out a whole table is immediately visible (null-code drops are
    counted in the same message):

    ```text
    `admissions/visit`: dropped 12/5000 rows with null time (unparsable or missing under the configured formats)
    ```

- **Coalesced multi-format times work as written**, including across columns. The idiom for a single
    column mixing several formats is a coalesce of lenient parses — each row takes the first format
    that matches, and rows matching none are dropped (and counted):

```yaml
visit:
  code: VISIT
  time: coalesce($ts::?"%m/%d/%y %H:%M:%S", $ts::?"%m/%d/%y")
```

**What you must change:** audit every strict (`::"fmt"`) time cast — if the column can contain `""`
or unparsable junk you want dropped rather than erroring, make it lenient (`::?"fmt"`). Then watch
the extraction logs: new WARNING lines are the rows 0.6.x was dropping (or leaking) silently.

## 3. Metadata extraction and `codes.parquet`

The `extract_code_metadata` stage was substantially reworked for correctness and determinism. Most of
this is transparent, but the output shape changes and two config patterns now error.

### 3a. Re-extract event shards before running metadata extraction

The metadata stage now requires every extracted event row carrying `code_components` to also carry the
`source_block` tag that 0.7.0's event extraction stamps unconditionally. Event shards produced by a
0.6.x run may lack it, and the stage refuses them with a `ValueError`.

**What you must change:** re-run event extraction under 0.7.0 before running `extract_code_metadata`;
don't point the 0.7.0 metadata stage at shards extracted by 0.6.x.

### 3b. `_metadata` is now a dftly program; `_match_on` is removed

A `_metadata` entry is now a mapping of **output column name → dftly expression**, evaluated over the
raw metadata table. The contract is *name matching*: produced columns whose names match the code
expression's component columns are the **join keys**; every other produced column is metadata output
attached to the matched codes. Three consequences drive the migration:

- **Join keys must be produced explicitly.** 0.6.x implicitly joined on all code-referenced columns;
    0.7.0 requires the block to produce each key column (a block producing no component-named column
    is a config error listing the components the event offers).
- **`_match_on` is gone.** Partial matching is now simply *which* component-named columns the block
    produces — producing a subset broadcasts the metadata to every code sharing those key values. A
    leftover `_match_on` key raises a `ValueError` pointing here.
- **Values have exactly dftly's semantics — a bare string is a LITERAL.** In 0.6.x,
    `description: label` meant "take raw column `label`". In 0.7.0 it means the constant *text*
    `"label"`. See the loud warning below: this is the one part of the migration that does **not**
    fail with an error if you skip it.

> [!WARNING]
> **An unmigrated bare-string `_metadata` value does not error — it silently produces the wrong
> data.** `description: label` stamps the literal string `"label"` as the description of every
> matched code, and a *join key* written as `itemid: itemid` becomes the constant `"itemid"`, which
> matches (at most) codes whose component value is literally the text `itemid` — i.e., usually
> nothing, with only a zero-match WARNING in the logs. Audit **every** value in every `_metadata`
> block and prefix column reads with `$` (`description: $label`, `itemid: $itemid`).

**Before (0.6.x):**

```yaml
labs:
  lab:
    code: f"LAB//{$test_name}//{$units}"
    _metadata:
      lab_dictionary:
        description: label # bare string = raw metadata column
medications:
  med:
    code: f"{$medication_name}//{$dose}"
    _metadata:
      med_classes:
        _match_on: medication_name # narrow the join
        description: drug_class
```

**After (0.7.0):**

```yaml
labs:
  lab:
    code: f"LAB//{$test_name}//{$units}"
    _metadata:
      lab_dictionary:
        test_name: $test_name # join keys: names match the code's components
        units: $units
        description: $label
medications:
  med:
    code: f"{$medication_name}//{$dose}"
    _metadata:
      med_classes:
        medication_name: $medication_name # partial match: produce only this key
        description: $drug_class
```

**What you must change**, per `_metadata` entry:

1. Add one line per join key: `component: $component` for a full match, or only the components you
    want to key on for a partial match (the old `_match_on` list, one line each).

2. Rewrite every value as a dftly expression:

    - Bare column names get a `$` prefix: `description: label` → `description: $label` (see the
        warning above — this one is silent if missed).
    - The list-coalesce form `description: [special_title, title]` is removed — write
        `description: coalesce($special_title, $title)` (or `$special_title ?? $title`). Lists were
        never valid dftly; the config error names this rewrite.
    - The 0.6.x `{col}` interpolation form (e.g. `prefix: "LOINC/{code}"`) must become a dftly
        f-string: `prefix: 'f"LOINC/{$code}"'`.
    - String literals are quoted: `vocab: '"MIMIC-IV"'`.

3. Rewrite `parent_codes` matcher/template machinery as dftly. A single template becomes an
    f-string; a matcher list becomes a chained conditional (`<then> if <condition> else ...`), and
    omitting the final `else` yields a real null for rows matching no case (do **not** write a bare
    `else null` — that is the *string* `"null"`):

    ```yaml
    # Before (0.6.x)
    parent_codes:
      - "ICD{icd_version}CM/{code}": {icd_version: 9}
      - "ICD{icd_version}CM/{code}": {icd_version: 10}

    # After (0.7.0)
    parent_codes: >-
      f"ICD{$icd_version}CM/{$code}" if $icd_version == "9"
      else f"ICD{$icd_version}CM/{$code}" if $icd_version == "10"
    ```

    One expression yields at most **one** parent per metadata row — a single row needing several
    simultaneous parents is no longer expressible in one entry (the reducer still unions parents
    across rows and across `_metadata` sources per code, which covers the vocabulary shapes we know
    of; declare a second `_metadata` entry over the same table if you truly need two parents from
    one row).

4. Config mistakes now fail at MESSY load time, in every stage — not mid-way through
    `extract_code_metadata`.

**New capability (not a migration requirement):** because keys are expressions, sourcing a key from a
differently-named metadata column is just a rename (`itemid: $omop_source_code`) — the capability the
accidental 0.6.x `_match_on` shadowing provided, now first-class — and key normalization is explicit
dftly (`itemid: $itemid::str`, `valueuom: $unit ?? $unit_alt`).

Join semantics are unchanged from the 0.7.0 component-join redesign, and still differ from 0.6.x:

- **Scoped to the declaring event.** A `_metadata` block joins only against codes from the event that
    declares it. In 0.6.x the join ran against all events' component columns, so same-named components
    with colliding values on *other* events (e.g. `CHART//{$itemid}` and `LAB//{$itemid}` sharing
    itemid values) wrongly received the metadata and a false `code_template`. If you relied on one
    `_metadata` block fanning out across events, declare it on each event.
- **Dtypes are normalized at the join.** Typed integer components now join all-String CSV metadata keys
    correctly, and integer-valued float components render as `220045`, not `220045.0`. A join that
    matches zero codes emits a WARNING instead of passing silently.
- A source column literally named `code` no longer causes a `DuplicateError` (or silent
    misclassification) in the metadata stage — and producing a *key* named `code` is allowed exactly
    when the code expression references a source column named `code` (the ICD/OMOP vocabulary shape);
    `code`/`code_template` remain reserved as metadata *output* names.

### 3c. `codes.parquet` has a deterministic, data-independent shape

The reduced `metadata/codes.parquet` is now byte-identical across runs (canonical config-order
reduction, sorted by code) with a stable schema:

- `description`: String — **distinct** values joined with `description_separator` in config order
    (repeated identical descriptions no longer duplicate).
- `parent_codes`: `List(String)`, deduplicated in first-seen order.
- `code_template`: a plain String. One code must map to exactly one template — distinct templates
    colliding on one code is now a config error naming the offenders.
- Every other extracted metadata column: always `List(String)` of distinct values, sorted — even when a
    code has a single source.
- Missing values are null, never `""` or `[]`.
- `code` and `code_template` are **reserved** output names — a `_metadata` block may not define them.

**What you must change:** downstream consumers of `codes.parquet` should expect list-typed extra
metadata columns and the schema above; byte-level diffs against 0.6.x outputs are expected. Remove any
`_metadata` entries named `code` / `code_template`.

### 3d. Merging into a pre-existing `codes.parquet` coalesces

When the stage merges extracted metadata into a pre-existing `codes.parquet`, same-named columns are now
coalesced — freshly extracted values win, pre-existing values fill the gaps — instead of silently
forking into `*_right` duplicate columns. Dtype conflicts between the two sides raise an error naming
the column.

### 3e. Mixed-format source prefixes are an error

A multi-file source prefix (metadata *or* event table) mixing csv-family and parquet-family chunks now
raises a `ValueError` instead of silently unifying typed parquet with all-String csv. Convert the chunks
to a single format.

## 4. Raw-data fetching: `sources:` + `meds-extract-download`

0.6.x left raw-data fetching entirely to each ETL. 0.7.0 adds a download layer: you declare where raw
files live in a `sources:` block of the same MESSY file, and the new `meds-extract-download` CLI (or the
`Source.download_all` Python API) stages them — with SHA-256 verification, `.part` staging + atomic
renames, resumable HTTP transfers, and a strict overwrite policy. This is additive, but it's the reason
you can delete your `download.py`.

### 4a. Combined MESSY file

```yaml
# messy.yaml
sources:
  dataset: # bucket selected by key= (default "dataset")
    - type: physionet
      base_url: https://physionet.org/files/mimiciv/3.1
      username: ${oc.env:PHYSIONET_USER}
      password: ${oc.env:PHYSIONET_PASS}
      include: # optional fnmatch globs — stage only what the ETL reads
        - hosp/*.csv.gz
        - icu/*.csv.gz
  common: # always appended, regardless of key=
    - type: http
      urls:
        - https://raw.githubusercontent.com/.../concept_map.csv

_defaults:
  subject_id: $subject_id

hosp/patients:
  dob:
    code: MEDS_BIRTH
    time: $anchor_year::year
```

```bash
meds-extract-download spec=messy.yaml raw_input_dir=/tmp/raw
MEDS_transform-pipeline pipeline.yaml \
	--overrides input_dir=/tmp/raw output_dir=/tmp/out
```

The pipeline's `event_conversion_config_fp` points at the **same** file. The event-conversion stages
ignore `sources:` — and treat it as sensitive:

- `sources:` is stripped from the config dump the pipeline logs and from the config copy written into
    the output tree, so literal credentials or API keys in the block never land in logs or shared output
    directories (symbolic `${oc.env:...}` interpolations are left unresolved either way, and the pipeline
    never requires those env vars to be set).
- A MESSY file with **only** a `sources:` block (no event tables) is now rejected at config load with a
    clear error — previously it silently no-op'd and crashed stages later.

> [!IMPORTANT]
> **`etl:` is also a reserved top-level key now, and `dataset_version` is reserved inside `sources:`.**
> Like `sources:`, a top-level `etl:` block is stripped before event-table parsing — it is consumed
> only by the `meds-extract-run` generic runner (see the README's *Running a packaged dataset ETL*).
> Its schema is a small, flat, all-optional set: `dataset_name` / `raw_dataset_version` fallbacks plus
> the curated stage options (`row_chunksize`, `n_subjects_per_shard`, `split_fracs`,
> `external_splits_json_fp`, `do_dedup_text_and_numeric`, `description_separator`); anything else is
> rejected at config load. If a 0.6.x MESSY file used `etl` as a *table* prefix (a raw file literally
> named `etl.{csv,parquet}`), rename the file/prefix — the block no longer parses as an event table.
> Unlike `sources:`, `etl:` carries no credentials, so it is **not** redacted from logs or from the
> config copy written into the output tree. Inside `sources:`, the key `dataset_version` (scalar
> version string or `{bucket: version}` mapping) is version metadata, never a bucket —
> `meds-extract-download` won't select it via `key=`, bucket entries may interpolate it
> (`${sources.dataset_version}`), and `meds-extract-run` stamps it into the output metadata.

### 4b. The CLI

`meds-extract-download` takes Hydra dotlist overrides:

- `spec=` / `raw_input_dir=` — required.
- `key=` — which `sources:` bucket to pull (`dataset` default, `demo`, ...); `common` is always
    appended. A `key` naming no declared bucket is an error, not a silent no-op. A spec with no
    `sources:` block warns and exits 0.
- `concurrency=` — one thread pool shared across all sources.
- `continue_on_error=` — collect per-file failures and keep going; default stops at the first failing
    source.
- `do_overwrite=` — re-fetch even verified local copies.

Cross-source destination collisions are rejected up front, before any fetch. The process exits `0` only
on full success — wire it into scripts accordingly.

### 4c. Backends

| Backend           | `type:`     | Use case                                                                  |
| ----------------- | ----------- | ------------------------------------------------------------------------- |
| `HTTPSource`      | `http`      | explicit URL list (concept maps, public mirrors)                          |
| `PhysioNetSource` | `physionet` | any PhysioNet release (MIMIC, eICU, ...) — driven by its `SHA256SUMS.txt` |
| `FsspecSource`    | `fsspec`    | local re-runs / S3 / GCS mirrors of pre-downloaded data                   |

Every backend accepts `include:` / `exclude:` fnmatch globs over destination paths. `http` URL entries
are plain strings or dicts with `url`, `rel_path` (defaults to the URL basename), and `sha256`;
`HTTPSource` also takes custom request `headers:` for API-key-auth services (e.g. DANS DataVerse for
AUMCdb):

```
sources:
  dataset:
    - type: http
      headers:
        X-Dataverse-key: ${oc.env:AUMCDB_API_KEY}
      urls:
        - url: https://lifesciences.datastations.nl/api/access/datafile/:persistentId?persistentId=doi:...
          rel_path: AUMCdb.zip
```

### 4d. Overwrite policy

A pre-existing destination file that verifies against the manifest's SHA-256 is skipped; one that
**can't** be verified (checksum mismatch, or no manifest checksum) is a hard `FileExistsError` — never a
silent overwrite or a silent skip. Only `do_overwrite=true` clears and re-fetches.

### 4e. Install footprint

The base install runs the `fsspec` backend and the CLI. The `[download]` extra (`httpx`, `tenacity`) is
only needed for `http` / `physionet` sources:

```bash
pip install "MEDS_extract[download]"
```

**What you must change:** nothing is required — but you can delete your ETL's `download.py` and replace
it with a `sources:` block, and you should move any credentials in the block to `${oc.env:...}`
interpolations.

## 5. Python and dependency floors

- **Python**: 0.6.x required ≥ 3.12; 0.7.0 *relaxes* the floor to ≥ 3.11. Nothing to change; 3.11
    environments now work.
- **dftly**: `>=0.1.2,<0.2` → `>=0.5.0` (0.5 introduces the `??` operator from section 2).
- **MEDS-transforms**: `~=0.6.0` → `>=0.6.7,<0.7` (0.6.7 added the StageExample / pipeline-tester APIs
    MEDS_extract's stages now register against).

Update your downstream ETL's `pyproject.toml`:

```toml
# Before
"MEDS-transforms~=0.6.0",
"MEDS_extract>=0.6.0,<0.7",

# After
"MEDS-transforms>=0.6.7,<0.7",
"MEDS_extract>=0.7.0,<0.8",
```

## 6. Example / tutorial restructure

The 0.6.x tutorial notebook (`example/example.ipynb`) is replaced by a regression-tested
`example/README.md` run end-to-end in CI. If you had tooling or docs pointing at the notebook, redirect
them — the new layout uses a combined `messy.yaml` (sources + event conversion) and a `pipeline.yaml`
for the stage DAG.

## Recommended migration order

1. **Rewrite your MESSY file** per section 1 (key renames), then audit composite codes and time
    casts per section 2 (`??` coalescing where you want rows kept; `::?` where unparsable times
    should drop instead of error), and rewrite every `_metadata` block per section 3b (produce the
    join-key columns explicitly; delete `_match_on`; prefix every column read with `$` — bare
    strings are literals now; rewrite `parent_codes` matchers as conditionals).
2. **Add a `sources:` block** to the same file (renaming it to `messy.yaml` is conventional, not
    required) and delete your `download.py`. Move credentials to `${oc.env:...}`.
3. **Bump the dependency pins** per section 5.
4. **Re-run the pipeline end-to-end from extraction** — don't reuse 0.6.x event shards (section 3a) —
    and expect `codes.parquet` to differ byte-wise from 0.6.x outputs (section 3c).
5. **Run `meds-extract-download spec=messy.yaml raw_input_dir=...`** to confirm the download leg.

If any migration step isn't obvious from the above, file an issue — the `help wanted` label tracks
migration friction that warrants additional doc.
