# Migrating from MEDS_extract 0.6.x to 0.7.0

The 0.7.0 release is a deliberate breaking cut: legacy MESSY keys are gone, Python 3.10 support is dropped, and a first-class download layer lands alongside the event-conversion pipeline. The upside for downstream ETLs is significant — if you run MIMIC-IV, eICU, HIRID, AUMCdb, or any other "raw files → MEDS" pipeline, 0.7.0 lets you delete most of your bespoke `download.py` / `pre_MEDS.py` boilerplate in favor of declarative MESSY blocks. This guide walks every breaking change end-to-end, with before/after snippets you can copy.

> **Scope**: 0.6.1 → 0.7.0. If you're on 0.5.x or earlier, land the 0.6.0 migration first (notebook-driven `event_cfg.yaml` → Hydra stage DAG); that's orthogonal.

## At a glance

| Area                                 | Before (0.6.x)                                | After (0.7.0)                                    | Why                                                            |
| ------------------------------------ | --------------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------- |
| Python floor                         | 3.10                                          | **3.11**                                         | `match`/`PEP 604`/dataclass slots lean, dftly 0.3 dropped 3.10 |
| `subject_id_col` / `subject_id_expr` | top-level table key                           | `_defaults.subject_id`                           | unified inheritance; #71                                       |
| `transforms`                         | top-level table key                           | `_table.cols`                                    | underscored-structural key convention                          |
| `join`                               | top-level table key with `columns_from_right` | `_table.join: {prefix: {key, cols}}`             | single source of join syntax; #71                              |
| `schema`                             | top-level table key                           | **removed** (use `_table.cols` casts)            | was dead code                                                  |
| Raw-data fetching                    | hand-rolled `download.py` per ETL             | `meds-extract-download` + MESSY `sources:` block | #82                                                            |
| MESSY file                           | event-conversion only                         | combined: `sources:` + event-conversion          | same file drives both stages; #86                              |
| `MEDS-transforms` pin                | `>=0.6.0,<0.7`                                | `>=0.6.7,<0.7`                                   | StageExample + pipeline_tester APIs; #80                       |

None of the new MESSY features introduced in the 0.7.0 cycle (chained `_table.cols` in #93, `HTTPSource` `headers:` in #91, Fetcher unarchive in #92, aggregated joins in #65) are breaking. They're additive — existing configs keep working; new ones opt in.

## 1. MESSY config redesign (breaking)

0.6.x accepted five ad-hoc top-level keys per table: `subject_id_col`, `subject_id_expr`, `transforms`, `join`, `schema`. 0.7.0 unifies them under two clearly-prefixed structural keys (`_defaults` for inherited fields, `_table` for whole-table modifications) — every other non-underscored key is an event name.

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

If the column is literally named `subject_id`, you can drop the field entirely — `_defaults.subject_id` defaults to `$subject_id`.

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

**New in 0.7.0 (PR #94, not breaking):** later `_table.cols` entries can reference earlier ones — so the MIMIC-IV / eICU / HIRID chained-pseudotime idiom collapses from ~60 YAML lines into ~15:

```yaml
_table:
  cols:
    hospital_discharge_ts: set_time(date_from_year($hospdischargeyear, 12, 31), 
      strptime($hospdischargetime24, "%H:%M:%S"))
    unit_admit_ts: $hospital_discharge_ts - $hospdischargeoffset::minutes
    unit_discharge_ts: $unit_admit_ts + $unitdischargeoffset::minutes
```

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

**After** — the joined table's prefix is the outer mapping key, and the inner block takes either `key:` (same column on both sides) or `left_on:` + `right_on:`:

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

**New in 0.7.0 (PR #98, not breaking):** aggregated joins eliminate custom pre-MEDS aggregation code. The motivating case is MIMIC-IV's `fix_static_data`, which computes earliest death-time per subject before joining into patients:

```yaml
# Was: custom pre_MEDS.py code
# death_times = admissions.group_by("subject_id").agg(pl.col("deathtime").min())
# patients.join(death_times, on="subject_id", how="left")

# Now: declarative MESSY block
hosp/patients:
  _table:
    join:
      hosp/admissions:
        key: subject_id
        cols:
          deathtime: min   # ← min/max/first/last/sum/mean/count
```

### 1d. `schema:` key removed entirely

The old top-level `schema:` key was dead code in 0.6.x — it was parsed but never consulted. If you were setting it: delete it. To cast a column, use a `_table.cols` expression:

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

If your MESSY file is small, copy-edit by hand using the table above. For larger files, this sed pattern covers the common case:

```bash
# ~90% of real-world migrations:
sed -i '
  s/^\(\s*\)subject_id_col: *\(\S\+\)/\1_defaults:\n\1  subject_id: $\2/
  s/^\(\s*\)subject_id_expr: *\(.\+\)/\1_defaults:\n\1  subject_id: \2/
  s/^\(\s*\)transforms:/\1_table:\n\1  cols:/
' event_cfg.yaml
```

The `join:` block is structurally too different for a sed pattern — edit those by hand.

## 2. Python 3.11 floor (PR #77)

0.7.0 requires Python ≥ 3.11 (0.6.x supported 3.10+). If you're pinning:

```toml
# Before
requires-python = ">=3.10"

# After
requires-python = ">=3.11"
```

The practical effect for downstream ETLs: you can use `match` statements, `X | Y` union syntax, and dataclass `slots=True` everywhere in your own code without conditional imports.

## 3. `meds-extract-download` + MESSY `sources:` block (new layer, not breaking)

0.6.x left raw-data fetching entirely to each ETL — every downstream ETL had its own `download.py` with hand-rolled `requests`/`curl`/BeautifulSoup code plus bespoke `zipfile.extractall` / `tarfile.extractall` for archives. 0.7.0 adds a first-class `Source` ABC + `Fetcher` orchestrator + the `meds-extract-download` console script; you declare your data sources in the MESSY file alongside your event-conversion config, and `meds-extract-download` drives them.

### 3a. Combined MESSY format

The MESSY spec now carries both `sources:` (for the download stage) and event-tables (for the event-conversion stages). Example — delete old `download.py`, point pipelines at the same file for both stages:

```yaml
# messy.yaml
sources:
  dataset:
    - type: physionet
      base_url: https://physionet.org/files/mimiciv/3.1/
      username: ${oc.env:PHYSIONET_USER}
      password: ${oc.env:PHYSIONET_PASS}
  common:
    - type: http
      urls:
        - https://raw.githubusercontent.com/.../concept_map.csv

_defaults:
  subject_id: $subject_id

hosp/patients:
  dob:
    code: MEDS_BIRTH
    time: $anchor_year::year
  # ... more events
```

Run it:

```bash
meds-extract-download spec=messy.yaml raw_input_dir=/tmp/raw
MEDS_transform-pipeline pipeline.yaml \
  --overrides input_dir=/tmp/raw output_dir=/tmp/out
```

The pipeline's `event_conversion_config_fp` points at the **same** `messy.yaml`. `MessyConfig.parse` silently ignores the `sources:` block, so the event-conversion stages see only the table entries.

### 3b. Supported backends

| Backend           | YAML `type:` | Use case                                                                        |
| ----------------- | ------------ | ------------------------------------------------------------------------------- |
| `HTTPSource`      | `http`       | explicit URL list (concept maps, public mirrors)                                |
| `PhysioNetSource` | `physionet`  | any PhysioNet release (MIMIC, eICU, MIMIC-IV demo) — driven by `SHA256SUMS.txt` |
| `FsspecSource`    | `fsspec`     | local re-runs / S3 / GCS mirrors (re-runs via a cached copy)                    |

Custom HTTP headers (PR #95) unblock DANS DataVerse (AUMCdb) and any other API-key-auth service:

```yaml
sources:
  dataset:
    - type: http
      headers:
        X-Dataverse-key: ${oc.env:AUMCDB_API_KEY}
      urls:
        - url: 
            https://lifesciences.datastations.nl/api/access/datafile/:persistentId?persistentId=doi:10.17026/dans-22u-f8vd
          rel_path: AUMCdb.zip
```

Post-fetch archive unpack (PR #96) — with `unarchive: auto` the Fetcher unpacks `.zip`, `.tar.gz`, `.tgz`, `.tar` after fetch, with zip-slip/tar-slip guards:

```yaml
sources:
  dataset:
    - type: http
      urls:
        - url: https://example.com/AUMCdb.zip
          unarchive: zip
          cleanup_archive: true   # remove the archive after extraction
```

### 3c. Deleting your `download.py`

If your downstream ETL has `src/<dataset>_MEDS/download.py` with `def download_dataset(...)`, you can usually delete it entirely in 0.7.0. The replacement is a `sources:` block in your `messy.yaml`. Three common patterns:

**PhysioNet (MIMIC-IV, eICU, MIMIC-IV-demo):**

```yaml
sources:
  dataset:
    - type: physionet
      base_url: https://physionet.org/files/mimiciv/3.1/
      username: ${oc.env:PHYSIONET_USER}
      password: ${oc.env:PHYSIONET_PASS}
```

**DANS DataVerse (AUMCdb):**

```yaml
sources:
  dataset:
    - type: http
      headers:
        X-Dataverse-key: ${oc.env:AUMCDB_API_KEY}
      urls:
        - url: 
            https://lifesciences.datastations.nl/api/access/datafile/:persistentId?persistentId=doi:...
          rel_path: AUMCdb.zip
          unarchive: zip
          cleanup_archive: true
```

**GitHub-hosted concept maps (bundled with any dataset):**

```yaml
sources:
  common:  # ``common:`` is always appended — shared across dataset/demo buckets
    - type: http
      urls:
        - https://raw.githubusercontent.com/.../concept_map.csv
```

## 4. `MEDS-transforms` pin bump (PR #79)

0.7.0 requires `MEDS-transforms >=0.6.7,<0.7` (was `>=0.6.0,<0.7`). The 0.6.7 release added the StageExample + pipeline_tester APIs that MEDS_extract's stages now register against.

Update your downstream ETL's `pyproject.toml`:

```toml
# Before
"MEDS-transforms>=0.6.0,<0.7",
"MEDS_extract>=0.6.0,<0.7",

# After
"MEDS-transforms>=0.6.7,<0.7",
"MEDS_extract>=0.7.0,<0.8",
```

## 5. Example / tutorial restructure (PR #87, not breaking but notable)

The 0.6.x tutorial was a Jupyter notebook (`example/example.ipynb`) that required manual runtime setup. 0.7.0 replaces it with a regression-tested `example/README.md` that's run in CI end-to-end (`tests/test_example.py` under the `integration` marker). If you had tooling or docs pointing at `example.ipynb`, redirect them at `example/README.md` — the new layout uses `messy.yaml` (combined sources + event-conversion) and a `pipeline.yaml` for the stage DAG.

## Recommended migration order

1. **Read your existing `event_cfg.yaml`** and apply section 1's edits (the MESSY redesign). Most ETLs take ~10-20 minutes.
2. **Add a `sources:` block** at the top of the same file (renaming to `messy.yaml` is conventional but not required). Delete your `download.py`.
3. **Bump Python floor** in `pyproject.toml` to 3.11.
4. **Bump the dependency pin** for `MEDS_extract` and `MEDS-transforms`.
5. **Run `pytest`** — the StageExample machinery validates your scenarios without stage-specific glue code.
6. **Run `meds-extract-download spec=messy.yaml raw_input_dir=...`** end-to-end to confirm the download leg works.

If any migration step isn't obvious from the above, file an issue — the `help wanted` label tracks migration friction that warrants additional doc.
