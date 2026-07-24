# Migrating from MEDS_extract 0.6.x to 0.7.0

The 0.7.0 release is a deliberate breaking cut. Four areas change: the MESSY config key layout, null
handling in composite codes, metadata extraction / `codes.parquet`, and raw-data fetching (a first-class
download layer replaces per-ETL `download.py` scripts). This guide walks every breaking change with
before/after snippets you can copy.

> **Scope**: 0.6.x (any of 0.6.0–0.6.2) → 0.7.0. If you're on 0.5.x or earlier, land the 0.6.0 migration
> first (notebook-driven `event_cfg.yaml` → dftly-native MESSY + Hydra stage DAG); that's orthogonal.

## At a glance

| Area                                 | Before (0.6.x)                                           | After (0.7.0)                                                                    |
| ------------------------------------ | -------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `subject_id_col` / `subject_id_expr` | top-level table keys                                     | `_defaults.subject_id` (a dftly expression)                                      |
| `transforms`                         | top-level table key                                      | `_table.cols`                                                                    |
| `join`                               | top-level table key with `columns_from_right`            | `_table.join: {prefix: {key, cols}}`                                             |
| `schema`                             | top-level table key (parsed, never used)                 | **removed**                                                                      |
| Null component in a composite `code` | auto-filled with `"UNK"`, row kept                       | code is null → **row dropped**; opt back in per component with `?? 'UNK'`        |
| `_match_on` metadata joins           | joined against **all** events' codes; dtype-fragile      | scoped to the declaring event; join keys dtype-normalized; key-rename now errors |
| `codes.parquet`                      | run-order-dependent schema/values; `*_right` merge forks | deterministic byte-identical output; deduplicated values; stable schema          |
| Multi-file source prefixes           | csv + parquet chunks silently unified                    | mixed csv/parquet chunks are an error                                            |
| Raw-data fetching                    | hand-rolled `download.py` per ETL                        | MESSY `sources:` block + `meds-extract-download`                                 |
| Python floor                         | 3.12                                                     | **3.11** (relaxed, not raised)                                                   |
| Dependency pins                      | `MEDS-transforms~=0.6.0`, `dftly>=0.1.2,<0.2`            | `MEDS-transforms>=0.6.7,<0.7`, `dftly>=0.5.0`                                    |

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

## 2. Null components in composite codes

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

## 3. Metadata extraction and `codes.parquet`

The `extract_code_metadata` stage was substantially reworked for correctness and determinism. Most of
this is transparent, but the output shape changes and two config patterns now error.

### 3a. Re-extract event shards before running metadata extraction

The metadata stage now requires every extracted event row carrying `code_components` to also carry the
`source_block` tag that 0.7.0's event extraction stamps unconditionally. Event shards produced by a
0.6.x run may lack it, and the stage refuses them with a `ValueError`.

**What you must change:** re-run event extraction under 0.7.0 before running `extract_code_metadata`;
don't point the 0.7.0 metadata stage at shards extracted by 0.6.x.

### 3b. `_match_on` joins are scoped and normalized

- **Scoped to the declaring event.** A `_metadata` block with `_match_on` now joins only against codes
    from the event that declares it. In 0.6.x the join ran against all events' component columns, so
    same-named components with colliding values on *other* events (e.g. `CHART//{$itemid}` and
    `LAB//{$itemid}` sharing itemid values) wrongly received the metadata and a false `code_template`.
    If you relied on one `_metadata` block fanning out across events, declare it on each event.
- **Join keys must be raw metadata columns.** The accidental key-rename capability — declaring a
    `_match_on` column as a `_metadata` output expression to source the key from a differently-named
    column — is removed and now raises a `ValueError`. Rename the column in the metadata file itself
    instead.
- **Dtypes are normalized at the join.** Typed integer components now join all-String CSV metadata keys
    correctly, and integer-valued float components render as `220045`, not `220045.0`. A partial-match
    join that matches zero codes emits a WARNING instead of passing silently.
- A source column literally named `code` no longer causes a `DuplicateError` (or silent
    misclassification) in the metadata stage.

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

1. **Rewrite your MESSY file** per section 1 (key renames), then audit composite codes per section 2
    (`??` coalescing where you want rows kept).
2. **Add a `sources:` block** to the same file (renaming it to `messy.yaml` is conventional, not
    required) and delete your `download.py`. Move credentials to `${oc.env:...}`.
3. **Bump the dependency pins** per section 5.
4. **Re-run the pipeline end-to-end from extraction** — don't reuse 0.6.x event shards (section 3a) —
    and expect `codes.parquet` to differ byte-wise from 0.6.x outputs (section 3c).
5. **Run `meds-extract-download spec=messy.yaml raw_input_dir=...`** to confirm the download leg.

If any migration step isn't obvious from the above, file an issue — the `help wanted` label tracks
migration friction that warrants additional doc.
