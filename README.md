<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="static/logo_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="static/logo_light.svg">
    <img width="520" height="200" alt="MEDS Logot" src="static/logo_light.svg">
  </picture>
</p>

# MEDS-Extract

[![PyPI - Version](https://img.shields.io/pypi/v/MEDS-extract)](https://pypi.org/project/MEDS-extract/)
![python](https://img.shields.io/badge/-Python_3.11+-blue?logo=python&logoColor=white)
[![MEDS v0.4](https://img.shields.io/badge/MEDS-0.4-blue)](https://medical-event-data-standard.github.io/)
[![Documentation Status](https://readthedocs.org/projects/meds-extract/badge/?version=latest)](https://meds-extract.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/mmcdermott/MEDS_extract/graph/badge.svg?token=5RORKQOZF9)](https://codecov.io/gh/mmcdermott/MEDS_extract)
[![tests](https://github.com/mmcdermott/MEDS_extract/actions/workflows/tests.yaml/badge.svg)](https://github.com/mmcdermott/MEDS_extract/actions/workflows/tests.yaml)
[![code-quality](https://github.com/mmcdermott/MEDS_extract/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/mmcdermott/MEDS_extract/actions/workflows/code-quality-main.yaml)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/mmcdermott/MEDS_extract#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/mmcdermott/MEDS_extract/pulls)
[![contributors](https://img.shields.io/github/contributors/mmcdermott/MEDS_extract.svg)](https://github.com/mmcdermott/MEDS_extract/graphs/contributors)
[![DOI](https://zenodo.org/badge/954891070.svg)](https://doi.org/10.5281/zenodo.17535804)

MEDS Extract is a Python package that leverages the MEDS-Transforms framework to build efficient, reproducible
ETL (Extract, Transform, Load) pipelines for converting raw electronic health record (EHR) data into the
standardized [MEDS format](https://medical-event-data-standard.github.io/). If your dataset consists of files
containing patient observations with timestamps, codes, and values, MEDS Extract can automatically convert
your raw data into a compliant MEDS dataset in an efficient, scalable, and communicable way.

> **Migrating from 0.6.x?** The 0.7.0 release is a breaking cut: MESSY config key names changed (unifying under `_defaults:` and `_table:`), null components in composite codes now drop rows unless you coalesce them, `codes.parquet` gained a deterministic stable schema, and `meds-extract-download` now handles raw-data fetching declaratively. See [**MIGRATION.md**](MIGRATION.md) for copy-pastable before/after snippets per change.

## 🚀 Quick Start

### 1. Install via `pip`:

```bash
pip install MEDS-extract
```

> [!NOTE]
> The development line (towards **0.7.0**) pins `meds ~=0.4.0`, `MEDS-transforms >=0.6.7,<0.7`, and
> `dftly >=0.3.0`, and supports Python ≥ 3.11. The MESSY config schema changed for 0.7.0 — subject IDs are
> set in a `_defaults` block and table joins under `_table.join` — and the examples below use that new
> syntax. Each `code`/`time`/property value is a [dftly](https://github.com/mmcdermott/dftly) expression
> (see [Event Configuration Deep Dive](#-event-configuration-deep-dive)).

> [!WARNING]
> **Breaking change in v0.6.0**: The MESSY event configuration syntax has changed significantly. Event
> field expressions (e.g., `code` and `time`) are now parsed by
> [dftly](https://github.com/mmcdermott/dftly), a lightweight declarative expression language. The old
> `col()` function syntax and list-based code construction are no longer supported. The `time_format` key
> has been replaced by inline type casting with the `as` operator (e.g., `$timestamp as "%Y-%m-%d"`).
> See the [Event Configuration Deep Dive](#-event-configuration-deep-dive) for the updated syntax.

### 2. Prepare your raw data

Ensure your data meets these requirements:

- **File-based**: Data stored in `.csv`, `.csv.gz`, or `.parquet` files. These may be stored locally or in the
    cloud, though intermediate processing currently must be done locally.
- **Comprehensive Rows**: Each file contains a dataframe structure where each row contains all required
    information to produce one or more MEDS events at full temporal granularity, without additional joining or
    merging.
- **Integer subject IDs**: The `subject_id` column must contain integer values (`int64`). You can set
    `_defaults: {subject_id: hash($string_col)}` in your MESSY file to automatically convert string IDs to
    integers.

If these requirements are not met, you may need to perform some pre-processing steps to convert your raw data
into an accepted format, though typically these are very minor (e.g., joining across a join key, converting
time deltas into timestamps, etc.).

### 3. Create a MESSY file for your messy data!

The secret sauce of MEDS-Extract is how you configure it to identify events within your raw data files. This
is done by virtue of the "MEDS-Extract Specification Syntax YAML" (MESSY) file. Event field values like
`code` and `time` are written as [dftly](https://github.com/mmcdermott/dftly) expressions -- a small
declarative language for column references, string interpolation, type casting, and arithmetic. See the
[dftly documentation](https://github.com/mmcdermott/dftly) for the full expression syntax.

Let's see an example of this event configuration file in action:

```yaml
# Global default subject ID (a dftly expression; can be overridden per file)
_defaults:
  subject_id: $patient_id

# File-level configurations
patients:
  _defaults:
    subject_id: $MRN # This file has a different subject ID column
  demographics: # One kind of event in this file.
    code: f"DEMOGRAPHIC//{$gender}"
    time:       # Static event
    race: $race           # Column references are `$`-prefixed
    ethnicity: $ethnicity

admissions:
  admission: # One kind of event in this file.
    code: f"HOSPITAL_ADMISSION//{$admission_type}"
    time: $admit_datetime as "%Y-%m-%d %H:%M:%S"
    department: $department # Extra columns get tracked
    insurance: $insurance

  discharge: # A different kind of event in this file.
    code: f"HOSPITAL_DISCHARGE//{$discharge_location}"
    time: $discharge_datetime as "%Y-%m-%d %H:%M:%S"

lab_results:
  lab:
    code: f"LAB//{$test_name}//{$units}"
    time: $result_datetime as "%Y-%m-%d %H:%M:%S"
    numeric_value: $result_value # This will get converted to a numeric
    text_value: $result_text # This will get converted to a string
```

This file is also called the "Event conversion configuration file" and is the heart of the MEDS Extract
system.

> [!IMPORTANT]
> Every `code`, `time`, and property value is a [dftly](https://github.com/mmcdermott/dftly) expression, so
> the syntax matters:
>
> - A **column reference** must be `$`-prefixed: `$gender`. A bare, unquoted word (e.g. `gender`) is a
>     **string literal**, not a column — a common gotcha.
> - **String interpolation** uses an f-string: `f"DEMOGRAPHIC//{$gender}"`.
> - **Type casts** use `as` (or the equivalent `::`): `$admit_datetime as "%Y-%m-%d %H:%M:%S"`.
>
> Subject IDs are set once per table in a `_defaults` block (not per event), and table joins go under
> `_table.join`. See the [Event Configuration Deep Dive](#-event-configuration-deep-dive) for the full syntax
> (including when YAML quoting is required).

### 4. Assemble your pipeline configuration

Beyond your extraction event configuration file, you also need to specify what pipeline stages you want to
run. You do this through a typical [MEDS-Transforms](https://meds-transforms.readthedocs.io/en/latest/)
pipeline configuration file. Here is a typical pipeline configuration file example.
Values like `$RAW_INPUT_DIR` are placeholders for your own paths or environment
variables and should be replaced with real values:

```yaml
input_dir: $RAW_INPUT_DIR
output_dir: $PIPELINE_OUTPUT

description: This pipeline extracts a dataset to MEDS format.

etl_metadata:
  dataset_name: $DATASET_NAME
  dataset_version: $DATASET_VERSION

# Points to the event conversion (MESSY) yaml file defined above. Replace with a real path.
event_conversion_config_fp: $EVENT_CONVERSION_CONFIG
# The shards mapping is stored in the root of the final output directory.
shards_map_fp: ${output_dir}/metadata/.shards.json

# Used if you need to load input files from cloud storage.
cloud_io_storage_options: {}

stages:
  - shard_events
  - split_and_shard_subjects
  - convert_to_subject_sharded
  - convert_to_MEDS_events
  - merge_to_MEDS_cohort
  - extract_code_metadata
  - finalize_MEDS_metadata
  - finalize_MEDS_data
```

Save it on disk to `$PIPELINE_YAML` (e.g., `pipeline_config.yaml`).

> [!NOTE]
> A pipeline with these defaults is provided at `MEDS_extract.configs._extract.yaml`. Instead of writing
> your own pipeline file you can reference the packaged one directly with the `pkg://` prefix (note the
> `.yaml` suffix is required) and supply the per-run values as overrides. The packaged config ships no
> dataset name/version, so pass those too:
>
> ```bash
> MEDS_transform-pipeline pkg://MEDS_extract.configs._extract.yaml \
> 	--overrides \
> 	input_dir="$RAW_INPUT_DIR" \
> 	output_dir="$PIPELINE_OUTPUT" \
> 	event_conversion_config_fp="$EVENT_CONVERSION_CONFIG" \
> 	dataset.name="$DATASET_NAME" \
> 	dataset.version="$DATASET_VERSION"
> ```

### 5. Run the extraction pipeline

MEDS-Extract does not have a stand-alone CLI runner; instead, you run it via the default MEDS-Transforms
pipeline runner, passing your pipeline configuration file as the first **positional** argument (there is no
`pipeline_config_fp=` flag):

```bash
MEDS_transform-pipeline "$PIPELINE_YAML"
```

Any field in the pipeline file can be overridden on the command line after `--overrides`, e.g.
`MEDS_transform-pipeline "$PIPELINE_YAML" --overrides event_conversion_config_fp=/path/to/messy.yaml`.

The result of this will be an extracted MEDS dataset in the specified output directory!

## 📊 End-to-End Example

MEDS Extract ships with a small synthetic dataset in the `example/` directory. Here we run
the full pipeline and inspect the output. This section also serves as an automated test —
it is executed by pytest via `--doctest-glob`.

```python
>>> import subprocess, tempfile, shutil, json
>>> from pathlib import Path
>>> import polars as pl
>>> from pretty_print_directory import print_directory, PrintConfig

```

First, copy the example data into a temporary directory and run the pipeline:

```python
>>> tmpdir = tempfile.mkdtemp()
>>> _ = shutil.copytree("example/raw_data", f"{tmpdir}/raw_data")
>>> _ = shutil.copy("example/messy.yaml", tmpdir)
>>> result = subprocess.run(
...     f"MEDS_transform-pipeline "
...     f"pkg://MEDS_extract.configs._extract.yaml "
...     f"--overrides "
...     f"input_dir={tmpdir}/raw_data "
...     f"output_dir={tmpdir}/output "
...     f"event_conversion_config_fp={tmpdir}/messy.yaml "
...     f"dataset.name=EXAMPLE "
...     f"dataset.version=1.0",
...     shell=True, capture_output=True,
... )
>>> assert result.returncode == 0, result.stderr.decode()[-500:]

```

The pipeline produces MEDS-format parquet shards split into train/tuning/held_out:

```python
>>> output = Path(f"{tmpdir}/output")
>>> print_directory(output / "data", PrintConfig(ignore_regex=r"\.logs"))
├── held_out
│   └── 0.parquet
├── train
│   └── 0.parquet
└── tuning
    └── 0.parquet

```

Each shard contains the standard MEDS columns:

```python
>>> df = pl.read_parquet(output / "data" / "train" / "0.parquet")
>>> sorted(df.columns)
['code', 'code_components', 'numeric_value', 'source_block', 'subject_id', 'time']
>>> df.schema["subject_id"]
Int64
>>> df.schema["code"]
String

```

MEDS-Extract also adds provenance and structure columns to help trace and query events.
The `source_block` column tracks which MESSY config block produced each event:

```python
>>> df.group_by("source_block").len().sort("source_block")
shape: (7, 2)
┌─────────────────────┬─────┐
│ source_block        ┆ len │
│ ---                 ┆ --- │
│ str                 ┆ u32 │
╞═════════════════════╪═════╡
│ diagnoses/dx        ┆ 10  │
│ labs_vitals/lab     ┆ 70  │
│ medications/med     ┆ 10  │
│ patients/dob        ┆ 8   │
│ patients/dod        ┆ 1   │
│ patients/eye_color  ┆ 8   │
│ patients/hair_color ┆ 8   │
└─────────────────────┴─────┘

```

The `code_components` struct column preserves the individual column values that were
combined to form the code. This enables queries on code components without parsing the
code string — for example, finding all Glucose readings regardless of units:

```python
>>> glucose = df.filter(
...     pl.col("code_components").struct.field("test_name") == "Glucose (mg/dL)"
... )
>>> glucose.select("subject_id", "time", "numeric_value").sort("subject_id", "time").head(3)
shape: (3, 3)
┌────────────┬─────────────────────┬───────────────┐
│ subject_id ┆ time                ┆ numeric_value │
│ ---        ┆ ---                 ┆ ---           │
│ i64        ┆ datetime[μs]        ┆ f32           │
╞════════════╪═════════════════════╪═══════════════╡
│ 1          ┆ 2025-03-09 15:18:00 ┆ 122.290001    │
│ 1          ┆ 2025-06-05 17:02:00 ┆ 185.919998    │
│ 2          ┆ 2024-08-12 20:57:00 ┆ 157.539993    │
└────────────┴─────────────────────┴───────────────┘

```

The metadata directory contains a dataset descriptor, code metadata, and subject splits:

```python
>>> print_directory(output / "metadata", PrintConfig(ignore_regex=r"\.shards|\.logs"))
├── codes.parquet
├── dataset.json
└── subject_splits.parquet
>>> meta = json.loads((output / "metadata" / "dataset.json").read_text())
>>> meta["dataset_name"]
'EXAMPLE'
>>> splits = pl.read_parquet(output / "metadata" / "subject_splits.parquet")
>>> sorted(splits["split"].unique().to_list())
['held_out', 'train', 'tuning']
>>> len(splits)
10

```

The event config includes `_metadata` blocks that link events to description files.
Metadata joins the extracted codes on their raw code components: lab descriptions match
on `test_name` (the code's only component), while medication descriptions use
`_match_on` to narrow the join — the code is `f"{$medication_name}//{$dose}"` but the
metadata only has `medication_name` (see
[Metadata linking, in depth](#metadata-linking-in-depth) for a full walkthrough):

```python
>>> codes = pl.read_parquet(output / "metadata" / "codes.parquet")
>>> codes.filter(pl.col("code").str.starts_with("Metformin") | (pl.col("code") == "Glucose (mg/dL)")).sort("code")
shape: (2, 3)
┌───────────────────┬─────────────────────┬────────────────────────────────┐
│ code              ┆ description         ┆ code_template                  │
│ ---               ┆ ---                 ┆ ---                            │
│ str               ┆ str                 ┆ str                            │
╞═══════════════════╪═════════════════════╪════════════════════════════════╡
│ Glucose (mg/dL)   ┆ Blood glucose level ┆ $test_name                     │
│ Metformin//500 mg ┆ Antidiabetic        ┆ f"{$medication_name}//{$dose}" │
└───────────────────┴─────────────────────┴────────────────────────────────┘
>>> _ = shutil.rmtree(tmpdir)

```

### Real-World Datasets

MEDS Extract has been successfully used to convert several major EHR datasets, including
[MIMIC-IV](https://github.com/Medical-Event-Data-Standard/MIMIC_IV_MEDS).

## 📖 Event Configuration Deep Dive

The event configuration file is the heart of MEDS Extract. Here's how it works:

### Basic Structure

```yaml
relative_table_file_stem:
  event_name:
    code: [required] How to construct the event code (dftly expression)
    time: [required] Timestamp expression (set to null for static events)
    property_name: $column_name  # Additional properties (also dftly expressions)
    _metadata:                  # Optional: link to external metadata tables
      metadata_file_prefix:
        output_column: source_column
```

All `code` and `time` values are parsed as [dftly](https://github.com/mmcdermott/dftly) expressions.
dftly is a lightweight declarative expression language for data transformations. The key syntax elements
are:

- **Column references**: `$`-prefixed names (e.g., `$test_name`). A bare, unquoted word (e.g. `test_name`) is
    a **string literal**, not a column reference.
- **String literals**: a quoted value (e.g., `"ADMISSION"`) or a single bare token (e.g., `MEDS_BIRTH`)
- **String interpolation**: an f-string with `$`-prefixed columns in braces (e.g.,
    `f"LAB//{$test_name}//{$units}"`)
- **Type casting**: the `as` operator, or the equivalent `::` (e.g., `$timestamp as "%Y-%m-%d"` /
    `$timestamp::"%Y-%m-%d"`, to parse a datetime)
- **Arithmetic**: `$a + $b`, `$val * 2`
- **Hashing**: `hash($mrn)` for converting string IDs to integers

> [!NOTE]
> Quoting these expressions in YAML is optional for the forms shown here (the Quick Start above leaves them
> unquoted and they parse fine); YAML only *requires* quoting when a value would otherwise be misread — e.g.
> one beginning with `{`, `[`, or `*`. As a safe default, the shipped
> [`example/messy.yaml`](./example/messy.yaml) single-quotes the f-strings and the `::`/`as` casts
> (e.g. `code: 'f"EYE_COLOR//{$eye_color}"'`, `time: '$dob::"%Y-%m-%dT%H:%M:%S"'`) while leaving bare
> literals (`MEDS_BIRTH`) and plain `$column` references unquoted.

### Code Construction

Event codes can be built in several ways:

```yaml
# Simple string literal
vitals:
  heart_rate:
    code: "HEART_RATE"

# Column reference (the value of the `measurement_type` column)
vitals:
  heart_rate:
    code: $measurement_type

# Composite codes with string interpolation (joined with "//")
vitals:
  heart_rate:
    code: f"VITAL_SIGN//{$measurement_type}//{$units}"
```

These rules are enforced by an executable doctest (run in CI), so the examples above cannot silently drift
from the parser:

```python
>>> import polars as pl
>>> from MEDS_extract.config import EventConfig
>>> raw_demo = pl.DataFrame({
...     "subject_id": [1],
...     "measurement_type": ["HR"],
...     "units": ["bpm"],
...     "charttime": ["03/09/2025 15:18"],
... })
>>> def code_of(expr):  # the ``code`` produced by a dftly expression
...     ev = EventConfig.parse("e", {"code": expr, "time": None})
...     return ev.extract(raw_demo.lazy(), "vitals/e").collect()["code"].to_list()
>>> code_of('"HEART_RATE"')                                  # quoted string literal
['HEART_RATE']
>>> code_of("$measurement_type")                             # column reference ($-prefixed)
['HR']
>>> code_of("measurement_type")                              # bare word -> literal, NOT the column
['measurement_type']
>>> code_of('f"VITAL_SIGN//{$measurement_type}//{$units}"')  # f-string interpolation
['VITAL_SIGN//HR//bpm']
>>> EventConfig.parse(  # type cast with `as` (or the equivalent `::`)
...     "e", {"code": "VITALS", "time": '$charttime as "%m/%d/%Y %H:%M"'}
... ).extract(raw_demo.lazy(), "vitals/e").collect()["time"].to_list()
[datetime.datetime(2025, 3, 9, 15, 18)]

```

#### Null components in composite codes

A MEDS `code` may never be null, and string interpolation **null-propagates**: if any interpolated
component is null, the whole `code` becomes null and that row is **dropped**. To keep such rows, give
the component a fallback with dftly's `??` (coalesce) operator — single-quote a literal fallback
*inside* the double-quoted f-string. Whether a missing component drops the row or is filled in is
your choice, made per component:

```python
>>> import polars as pl
>>> from MEDS_extract.config import EventConfig
>>> labs = pl.DataFrame({
...     "subject_id": [1, 2, 3, 4],
...     "itemid": ["GLU", "GLU", None, None],        # present, present, null, null
...     "valueuom": ["mg/dL", None, "mg/dL", None],  # present, null,    present, null
... })
>>> def codes(expr):  # surviving (subject_id, code) rows for a `code` expression
...     ev = EventConfig.parse("lab", {"code": expr, "time": None})
...     return ev.extract(labs.lazy(), "labs/lab").collect().sort("subject_id").select(
...         "subject_id", "code"
...     )
>>> codes('f"{$itemid}//{$valueuom}"')  # plain: any null component drops the row
shape: (1, 2)
┌────────────┬────────────┐
│ subject_id ┆ code       │
│ ---        ┆ ---        │
│ i64        ┆ str        │
╞════════════╪════════════╡
│ 1          ┆ GLU//mg/dL │
└────────────┴────────────┘
>>> codes("f\"{$itemid ?? 'UNK'}//{$valueuom ?? 'UNK'}\"")  # fill both -> keep every row
shape: (4, 2)
┌────────────┬────────────┐
│ subject_id ┆ code       │
│ ---        ┆ ---        │
│ i64        ┆ str        │
╞════════════╪════════════╡
│ 1          ┆ GLU//mg/dL │
│ 2          ┆ GLU//UNK   │
│ 3          ┆ UNK//mg/dL │
│ 4          ┆ UNK//UNK   │
└────────────┴────────────┘
>>> codes("f\"{$itemid ?? 'NO_ITEM'}//{$valueuom ?? 'NO_UNIT'}\"")  # the fallback is any literal
shape: (4, 2)
┌────────────┬──────────────────┐
│ subject_id ┆ code             │
│ ---        ┆ ---              │
│ i64        ┆ str              │
╞════════════╪══════════════════╡
│ 1          ┆ GLU//mg/dL       │
│ 2          ┆ GLU//NO_UNIT     │
│ 3          ┆ NO_ITEM//mg/dL   │
│ 4          ┆ NO_ITEM//NO_UNIT │
└────────────┴──────────────────┘
>>> codes("f\"{$itemid}//{$valueuom ?? 'UNK'}\"")  # fill only the unit: a null itemid still drops
shape: (2, 2)
┌────────────┬────────────┐
│ subject_id ┆ code       │
│ ---        ┆ ---        │
│ i64        ┆ str        │
╞════════════╪════════════╡
│ 1          ┆ GLU//mg/dL │
│ 2          ┆ GLU//UNK   │
└────────────┴────────────┘

```

### Time Handling

```yaml
# Source column that is already datetime-typed (no cast needed)
lab_results:
  lab:
    time: $result_time

# A string column: parse it with an explicit format via a type cast
lab_results:
  lab:
    time: $result_time as "%m/%d/%Y %H:%M"

# Static events (no time)
demographics:
  gender:
    time: null
```

### Subject ID Configuration

The subject ID is a table-level concept: set it once per table in a `_defaults` block (as a dftly
expression), never inside an individual event.

```yaml
# Global default, applied to every table (a dftly expression)
_defaults:
  subject_id: $patient_id

# File-specific override
admissions:
  _defaults:
    subject_id: $hadm_id
  admission:
    code: ADMISSION
    # ...

# Hash a string column into an integer subject ID
patients:
  _defaults:
    subject_id: hash($MRN)
  demographics:
    code: DEMOGRAPHIC
    time:
```

### Joining Tables

Sometimes subject identifiers are stored in a separate table from the events
you wish to extract. Specify a join under the table's `_table.join` block so the
necessary columns are merged in before extraction, then point `_defaults.subject_id`
at the joined-in column.

```yaml
vitals:
  _table:
    join:
      stays: # join the `stays` table...
        key: stay_id # ...on a shared `stay_id` column
        cols: [patient_id] # ...bringing `patient_id` across
  _defaults:
    subject_id: $patient_id # now available from the join
  HR:
    code: HR
    time: $charttime as "%m/%d/%Y %H:%M:%S"
    numeric_value: $HR
```

The join key may be a single shared column (`key: stay_id`) or asymmetric
(`left_on:`/`right_on:`), and `cols` lists the columns to pull from the right table.

### Metadata linking, in depth

Datasets usually ship dictionary tables alongside the event data — `d_items.csv`,
`d_icd_diagnoses.csv`, a LOINC map. `_metadata` blocks link those tables to your
extracted codes, producing `metadata/codes.parquet`. The mental model:

- Every extracted event row carries `code_components` — a struct of the **raw source
    values** the code was built from — and `source_block`, the MESSY block that produced
    it (see [Output Columns](#output-columns)).
- The `extract_code_metadata` stage attaches metadata by **joining those raw component
    values against raw metadata columns**, scoped to the event block that declared the
    `_metadata` entry.
- The assembled code *string* is never matched against. Your metadata tables keep their
    raw values as-is — you never mirror the code expression's prefixes, separators,
    casts, or `??` fallbacks inside a metadata table.

The walkthrough below is executable (it runs in CI). Two helpers keep each example to
its essentials: `extract_events` stands in for the `convert_to_MEDS_events` stage
(extracting every event block of a MESSY snippet from in-memory raw tables), and
`link_metadata` runs the real `extract_code_metadata` stage over those events plus
metadata files, returning the reduced `metadata/codes.parquet`:

```python
>>> import yaml
>>> from omegaconf import OmegaConf
>>> from MEDS_extract.config import EventConfig
>>> from MEDS_extract.extract_code_metadata.extract_code_metadata import (
...     extract_metadata,
...     main as ecm_stage,
... )
>>> def extract_events(messy: str, raw_tables: dict) -> pl.DataFrame:
...     """What `convert_to_MEDS_events` emits: every MESSY event block, extracted."""
...     frames = []
...     for table, events in yaml.safe_load(messy).items():
...         for name, event_cfg in events.items():
...             ev = EventConfig.parse(name, event_cfg)
...             frames.append(ev.extract(raw_tables[table].lazy(), f"{table}/{name}").collect())
...     return pl.concat(frames, how="diagonal_relaxed")
>>> def link_metadata(messy: str, events: pl.DataFrame, metadata_files: dict) -> pl.DataFrame:
...     """Run the real `extract_code_metadata` stage; return `metadata/codes.parquet`."""
...     root = Path(tempfile.mkdtemp())
...     for d in ("events", "raw", "out"):
...         (root / d).mkdir()
...     events.write_parquet(root / "events" / "0.parquet")
...     for fname, df in metadata_files.items():  # .parquet stays typed; .csv reads as String
...         fp = root / "raw" / fname
...         df.write_parquet(fp) if fname.endswith(".parquet") else df.write_csv(fp)
...     (root / "messy.yaml").write_text(messy)
...     cfg = OmegaConf.create({
...         "do_overwrite": True, "worker": 0, "polling_time": 0.1,
...         "input_dir": str(root / "raw"),
...         "event_conversion_config_fp": str(root / "messy.yaml"),
...         "stage_cfg": {
...             "data_input_dir": str(root / "events"),
...             "output_dir": str(root / "out"),
...             "metadata_input_dir": str(root / "no_prior_metadata"),
...             "reducer_output_dir": str(root / "out"),
...             "description_separator": "; ",
...         },
...     })
...     ecm_stage.main_fn(cfg)
...     return pl.read_parquet(root / "out" / "codes.parquet")

```

#### Full match: one code, one metadata row

By default a `_metadata` entry joins on **every** source column the code references.
Each block under `_metadata` names a metadata file prefix (here `lab_dictionary`
matches `lab_dictionary.csv` in your input directory) and maps output columns to
source columns:

```python
>>> messy = """
... labs:
...   lab:
...     code: f"LAB//{$test_name}//{$units}"
...     time: $ts::"%Y-%m-%d %H:%M"
...     numeric_value: $result
...     _metadata:
...       lab_dictionary:           # <input_dir>/lab_dictionary.csv
...         description: label     # output column <- metadata source column
... """

```

Extracting events from a raw `labs` table shows what the join will run against — each
row's `code_components` holds the raw `test_name` and `units` values, and
`source_block` records the declaring MESSY block:

```python
>>> labs = pl.DataFrame({
...     "subject_id": [1, 1, 2],
...     "test_name": ["GLU", "CREAT", "GLU"],
...     "units": ["mg/dL", "mg/dL", "mg/dL"],
...     "ts": ["2024-01-01 09:30", "2024-01-01 09:35", "2024-03-02 14:00"],
...     "result": [98.0, 1.1, 105.0],
... })
>>> events = extract_events(messy, {"labs": labs})
>>> events.select("code", "code_components", "source_block")
shape: (3, 3)
┌───────────────────┬───────────────────┬──────────────┐
│ code              ┆ code_components   ┆ source_block │
│ ---               ┆ ---               ┆ ---          │
│ str               ┆ struct[2]         ┆ str          │
╞═══════════════════╪═══════════════════╪══════════════╡
│ LAB//GLU//mg/dL   ┆ {"GLU","mg/dL"}   ┆ labs/lab     │
│ LAB//CREAT//mg/dL ┆ {"CREAT","mg/dL"} ┆ labs/lab     │
│ LAB//GLU//mg/dL   ┆ {"GLU","mg/dL"}   ┆ labs/lab     │
└───────────────────┴───────────────────┴──────────────┘

```

The metadata table carries the same raw columns — `test_name` and `units`, exactly as
they appear in the raw data, with no `LAB//` prefix or `//` separators:

```python
>>> lab_dictionary = pl.DataFrame({
...     "test_name": ["GLU", "CREAT", "NA"],
...     "units": ["mg/dL", "mg/dL", "mmol/L"],
...     "label": ["Serum glucose", "Serum creatinine", "Serum sodium"],
... })
>>> lab_dictionary
shape: (3, 3)
┌───────────┬────────┬──────────────────┐
│ test_name ┆ units  ┆ label            │
│ ---       ┆ ---    ┆ ---              │
│ str       ┆ str    ┆ str              │
╞═══════════╪════════╪══════════════════╡
│ GLU       ┆ mg/dL  ┆ Serum glucose    │
│ CREAT     ┆ mg/dL  ┆ Serum creatinine │
│ NA        ┆ mmol/L ┆ Serum sodium     │
└───────────┴────────┴──────────────────┘

```

Linking joins each code's components `(test_name, units)` against the same-named
metadata columns. The `NA` row matches no observed code, so it does not appear —
`codes.parquet` describes the codes your data actually contains:

```python
>>> link_metadata(messy, events, {"lab_dictionary.csv": lab_dictionary})
shape: (2, 3)
┌───────────────────┬────────────────────────────────┬──────────────────┐
│ code              ┆ code_template                  ┆ description      │
│ ---               ┆ ---                            ┆ ---              │
│ str               ┆ str                            ┆ str              │
╞═══════════════════╪════════════════════════════════╪══════════════════╡
│ LAB//CREAT//mg/dL ┆ f"LAB//{$test_name}//{$units}" ┆ Serum creatinine │
│ LAB//GLU//mg/dL   ┆ f"LAB//{$test_name}//{$units}" ┆ Serum glucose    │
└───────────────────┴────────────────────────────────┴──────────────────┘

```

#### `_match_on`: dictionaries keyed on one component

When the code is composite but the dictionary is keyed on just one of its components,
`_match_on` narrows the join to the named column(s); the metadata then broadcasts to
**every** code sharing that component value. Here both dose-variants of Metformin get
the drug class, from a dictionary that knows nothing about doses:

```python
>>> messy = """
... medications:
...   med:
...     code: f"{$medication_name}//{$dose}"
...     time:
...     _metadata:
...       med_classes:
...         _match_on: medication_name   # join on this component alone
...         description: drug_class
... """
>>> medications = pl.DataFrame({
...     "subject_id": [1, 2, 3],
...     "medication_name": ["Metformin", "Metformin", "Lisinopril"],
...     "dose": ["500 mg", "1000 mg", "10 mg"],
... })
>>> events = extract_events(messy, {"medications": medications})
>>> events.select("code", "code_components")
shape: (3, 2)
┌────────────────────┬─────────────────────────┐
│ code               ┆ code_components         │
│ ---                ┆ ---                     │
│ str                ┆ struct[2]               │
╞════════════════════╪═════════════════════════╡
│ Metformin//500 mg  ┆ {"500 mg","Metformin"}  │
│ Metformin//1000 mg ┆ {"1000 mg","Metformin"} │
│ Lisinopril//10 mg  ┆ {"10 mg","Lisinopril"}  │
└────────────────────┴─────────────────────────┘
>>> med_classes = pl.DataFrame({
...     "medication_name": ["Metformin", "Lisinopril"],
...     "drug_class": ["Antidiabetic", "ACE inhibitor"],
... })
>>> link_metadata(messy, events, {"med_classes.csv": med_classes})
shape: (3, 3)
┌────────────────────┬────────────────────────────────┬───────────────┐
│ code               ┆ code_template                  ┆ description   │
│ ---                ┆ ---                            ┆ ---           │
│ str                ┆ str                            ┆ str           │
╞════════════════════╪════════════════════════════════╪═══════════════╡
│ Lisinopril//10 mg  ┆ f"{$medication_name}//{$dose}" ┆ ACE inhibitor │
│ Metformin//1000 mg ┆ f"{$medication_name}//{$dose}" ┆ Antidiabetic  │
│ Metformin//500 mg  ┆ f"{$medication_name}//{$dose}" ┆ Antidiabetic  │
└────────────────────┴────────────────────────────────┴───────────────┘

```

Without `_match_on`, the metadata table would need both `medication_name` and `dose`
columns to match the full component set. Multiple columns also work:
`_match_on: [col_a, col_b]`.

#### Metadata is scoped to the declaring event

Two events may reference same-named components with colliding values — an `itemid` in
`vitals` and an unrelated `itemid` in `labs`. A `_metadata` block only ever attaches to
codes from the event block that declared it (that is what `source_block` is for):

```python
>>> messy = """
... vitals:
...   vital:
...     code: f"VITAL//{$itemid}"
...     time:
...     _metadata:
...       d_vitals:
...         description: label
... labs:
...   lab:
...     code: f"LAB//{$itemid}"
...     time:
... """
>>> vitals = pl.DataFrame({"subject_id": [1], "itemid": ["220045"]})
>>> labs = pl.DataFrame({"subject_id": [1], "itemid": ["220045"]})
>>> events = extract_events(messy, {"vitals": vitals, "labs": labs})
>>> events.select("code", "code_components", "source_block")
shape: (2, 3)
┌───────────────┬─────────────────┬──────────────┐
│ code          ┆ code_components ┆ source_block │
│ ---           ┆ ---             ┆ ---          │
│ str           ┆ struct[1]       ┆ str          │
╞═══════════════╪═════════════════╪══════════════╡
│ VITAL//220045 ┆ {"220045"}      ┆ vitals/vital │
│ LAB//220045   ┆ {"220045"}      ┆ labs/lab     │
└───────────────┴─────────────────┴──────────────┘
>>> d_vitals = pl.DataFrame({"itemid": ["220045"], "label": ["Heart Rate"]})
>>> link_metadata(messy, events, {"d_vitals.csv": d_vitals})
shape: (1, 3)
┌───────────────┬─────────────────────┬─────────────┐
│ code          ┆ code_template       ┆ description │
│ ---           ┆ ---                 ┆ ---         │
│ str           ┆ str                 ┆ str         │
╞═══════════════╪═════════════════════╪═════════════╡
│ VITAL//220045 ┆ f"VITAL//{$itemid}" ┆ Heart Rate  │
└───────────────┴─────────────────────┴─────────────┘

```

`LAB//220045` shares the component value but not the declaring block, so it receives
nothing — vocabulary declared for one event never leaks onto another.

#### Null components match null vocabulary keys

A `??`-coalesced component (see [Null components in composite
codes](#null-components-in-composite-codes)) keeps rows whose component is missing, and
the *raw* null is preserved in `code_components` — the fallback literal appears only in
the code string:

```python
>>> messy = """
... labs:
...   lab:
...     code: 'f"{$test_name}//{$units ?? ''UNK''}"'
...     time:
...     _metadata:
...       lab_dictionary:
...         description: label
... """
>>> labs = pl.DataFrame({
...     "subject_id": [1, 2],
...     "test_name": ["GLU", "GLU"],
...     "units": ["mg/dL", None],
... })
>>> events = extract_events(messy, {"labs": labs})
>>> events.select("code", "code_components")
shape: (2, 2)
┌────────────┬─────────────────┐
│ code       ┆ code_components │
│ ---        ┆ ---             │
│ str        ┆ struct[2]       │
╞════════════╪═════════════════╡
│ GLU//mg/dL ┆ {"GLU","mg/dL"} │
│ GLU//UNK   ┆ {"GLU",null}    │
└────────────┴─────────────────┘

```

The join treats null as an ordinary key value: a vocabulary row whose `units` cell is
empty describes *specifically* the unit-less variant. Note the dictionary says `UNK`
nowhere — it holds raw values, and the raw value here is null:

```python
>>> lab_dictionary = pl.DataFrame({
...     "test_name": ["GLU", "GLU"],
...     "units": ["mg/dL", None],
...     "label": ["Glucose (serum)", "Glucose (no unit given)"],
... })
>>> link_metadata(messy, events, {"lab_dictionary.csv": lab_dictionary}).select(
...     "code", "description"
... )
shape: (2, 2)
┌────────────┬─────────────────────────┐
│ code       ┆ description             │
│ ---        ┆ ---                     │
│ str        ┆ str                     │
╞════════════╪═════════════════════════╡
│ GLU//UNK   ┆ Glucose (no unit given) │
│ GLU//mg/dL ┆ Glucose (serum)         │
└────────────┴─────────────────────────┘

```

A null key is **not** a wildcard: the null-keyed row attached only to `GLU//UNK`, not
to `GLU//mg/dL`. If you instead want one description across *all* unit-variants of a
test, that is exactly the `_match_on` narrowing:

```python
>>> messy = """
... labs:
...   lab:
...     code: 'f"{$test_name}//{$units ?? ''UNK''}"'
...     time:
...     _metadata:
...       lab_dictionary:
...         _match_on: test_name
...         description: label
... """
>>> broadcast_dictionary = pl.DataFrame({"test_name": ["GLU"], "label": ["Blood glucose"]})
>>> link_metadata(messy, events, {"lab_dictionary.csv": broadcast_dictionary}).select(
...     "code", "description"
... )
shape: (2, 2)
┌────────────┬───────────────┐
│ code       ┆ description   │
│ ---        ┆ ---           │
│ str        ┆ str           │
╞════════════╪═══════════════╡
│ GLU//UNK   ┆ Blood glucose │
│ GLU//mg/dL ┆ Blood glucose │
└────────────┴───────────────┘

```

#### Typed metadata joins string components

Component dtypes come from your raw event files and metadata dtypes from the metadata
files, and they routinely disagree: csv-sourced events carry String components while a
parquet dictionary types `itemid` as an integer — or as a float, the classic
pandas-heritage shape where a nullable integer column became `220045.0`. Both sides of
the join are normalized through one canonical String rendering (integer-valued floats
render via `Int64`, so `220045.0` matches `"220045"`):

```python
>>> messy = """
... vitals:
...   vital:
...     code: f"VITAL//{$itemid}"
...     time:
...     _metadata:
...       d_items:
...         description: label
... """
>>> vitals = pl.DataFrame({"subject_id": [1, 2], "itemid": ["220045", "220179"]})
>>> events = extract_events(messy, {"vitals": vitals})
>>> d_items = pl.DataFrame({"itemid": [220045.0, 220179.0], "label": ["Heart Rate", "NBP systolic"]})
>>> d_items
shape: (2, 2)
┌──────────┬──────────────┐
│ itemid   ┆ label        │
│ ---      ┆ ---          │
│ f64      ┆ str          │
╞══════════╪══════════════╡
│ 220045.0 ┆ Heart Rate   │
│ 220179.0 ┆ NBP systolic │
└──────────┴──────────────┘
>>> link_metadata(messy, events, {"d_items.parquet": d_items})
shape: (2, 3)
┌───────────────┬─────────────────────┬──────────────┐
│ code          ┆ code_template       ┆ description  │
│ ---           ┆ ---                 ┆ ---          │
│ str           ┆ str                 ┆ str          │
╞═══════════════╪═════════════════════╪══════════════╡
│ VITAL//220045 ┆ f"VITAL//{$itemid}" ┆ Heart Rate   │
│ VITAL//220179 ┆ f"VITAL//{$itemid}" ┆ NBP systolic │
└───────────────┴─────────────────────┴──────────────┘

```

Non-integer floats keep their float rendering (`1.5` only matches `"1.5"`); see
`normalize_join_key` in `extract_code_metadata` for the exact rules.

#### Transforms in the code never touch the join

A code expression may transform its components — casts, `substring`, arithmetic. The
join still runs on the **raw** component values, so the vocabulary stays keyed on what
the raw data contains, not on what the code displays. Here the code keeps only the
3-character ICD-9 category, yet the full raw `icd_code` is what matches:

```python
>>> messy = """
... diagnoses:
...   dx:
...     code: f"ICD9//{substring($icd_code, 0, 3)}"
...     time:
...     _metadata:
...       d_icd:
...         description: long_title
... """
>>> diagnoses = pl.DataFrame({"subject_id": [1], "icd_code": ["25000"]})
>>> events = extract_events(messy, {"diagnoses": diagnoses})
>>> events.select("code", "code_components")
shape: (1, 2)
┌───────────┬─────────────────┐
│ code      ┆ code_components │
│ ---       ┆ ---             │
│ str       ┆ struct[1]       │
╞═══════════╪═════════════════╡
│ ICD9//250 ┆ {"25000"}       │
└───────────┴─────────────────┘
>>> d_icd = pl.DataFrame({
...     "icd_code": ["25000", "250"],
...     "long_title": ["Diabetes mellitus", "Should never match"],
... })
>>> link_metadata(messy, events, {"d_icd.csv": d_icd}).select("code", "description")
shape: (1, 2)
┌───────────┬───────────────────┐
│ code      ┆ description       │
│ ---       ┆ ---               │
│ str       ┆ str               │
╞═══════════╪═══════════════════╡
│ ICD9//250 ┆ Diabetes mellitus │
└───────────┴───────────────────┘

```

The row keyed on `"250"` — the *transformed* value that appears in the code string —
matched nothing. You never replicate a code expression's transforms in a metadata
table.

#### What errors, and why

Metadata linking is validated at configuration time, in every worker, before any data
is joined. A `_metadata` block on a **literal** code is rejected — a literal references
no source columns, so there are no components to match on:

```python
>>> extract_metadata(
...     pl.DataFrame({"label": ["Birth"]}),
...     {"code": "MEDS_BIRTH", "_metadata": {"description": "label"}},
... )
Traceback (most recent call last):
    ...
ValueError: The code expression 'MEDS_BIRTH' is a literal: it references no source columns, ...

```

A `_match_on` column must be one of the code's components:

```python
>>> extract_metadata(
...     pl.DataFrame({"medication_name": ["Metformin"], "drug_class": ["Antidiabetic"]}),
...     {"code": "$medication_name",
...      "_metadata": {"_match_on": "medication", "description": "drug_class"}},
... )
Traceback (most recent call last):
    ...
KeyError: "_match_on columns ['medication'] are not referenced by the code expression
'$medication_name'. Valid columns: ['medication_name']"

```

And a match column may not double as a `_metadata` output expression — join keys are
always raw metadata columns, never derived or renamed ones:

```python
>>> extract_metadata(
...     pl.DataFrame({"itemid_alias": ["220045"], "label": ["Heart Rate"]}),
...     {"code": 'f"CHART//{$itemid}"',
...      "_metadata": {"_match_on": "itemid", "itemid": "itemid_alias", "description": "label"}},
... )
Traceback (most recent call last):
    ...
ValueError: Match column(s) ['itemid'] may not also be declared as _metadata output ...

```

#### The shape of `codes.parquet`

The reduced output has a canonical, data-independent shape. One event may declare
several `_metadata` entries (and several codes can collapse onto one metadata row), so
every column is aggregated per code:

- **`description`**: a single String — distinct values from all sources, joined with
    the stage's `description_separator` in config order.
- **`parent_codes`**: `List(String)` of distinct `vocabulary/code` strings.
- **`code_template`**: a single String (one code, one template — see
    [Output Columns](#output-columns)).
- **any other column** (extras like `loinc` below): `List(String)` of distinct values,
    sorted. Missing values are null, never `[]` or `""`.

Here a local dictionary (two rows for `GLU`, i.e. non-unique by key) and a LOINC
ontology both describe the same code; note `parent_codes` built with the
`"LOINC/{loinc_code}"` interpolation form:

```python
>>> messy = """
... labs:
...   lab:
...     code: $test_name
...     time:
...     _metadata:
...       local_dictionary:
...         description: label
...         loinc: loinc_code
...       loinc_ontology:
...         description: long_name
...         parent_codes: LOINC/{loinc_code}
... """
>>> events = extract_events(messy, {"labs": pl.DataFrame({"subject_id": [1], "test_name": ["GLU"]})})
>>> local_dictionary = pl.DataFrame({
...     "test_name": ["GLU", "GLU"],
...     "label": ["Serum glucose", "Serum glucose"],
...     "loinc_code": ["2345-7", "2339-0"],
... })
>>> loinc_ontology = pl.DataFrame({
...     "test_name": ["GLU"],
...     "long_name": ["Glucose [Mass/vol] in Serum"],
...     "loinc_code": ["2345-7"],
... })
>>> out = link_metadata(
...     messy, events,
...     {"local_dictionary.csv": local_dictionary, "loinc_ontology.csv": loinc_ontology},
... )
>>> with pl.Config(fmt_str_lengths=60, tbl_width_chars=120):
...     print(out)
shape: (1, 5)
┌──────┬───────────────┬────────────────────────────────────────────┬──────────────────────┬──────────────────┐
│ code ┆ code_template ┆ description                                ┆ loinc                ┆ parent_codes     │
│ ---  ┆ ---           ┆ ---                                        ┆ ---                  ┆ ---              │
│ str  ┆ str           ┆ str                                        ┆ list[str]            ┆ list[str]        │
╞══════╪═══════════════╪════════════════════════════════════════════╪══════════════════════╪══════════════════╡
│ GLU  ┆ $test_name    ┆ Serum glucose; Glucose [Mass/vol] in Serum ┆ ["2339-0", "2345-7"] ┆ ["LOINC/2345-7"] │
└──────┴───────────────┴────────────────────────────────────────────┴──────────────────────┴──────────────────┘
>>> out.schema
Schema({'code': String, 'code_template': String, 'description': String,
        'loinc': List(String), 'parent_codes': List(String)})

```

If a pre-existing `metadata/codes.parquet` is present (e.g. hand-curated metadata for
literal codes), the reduced output is merged with it: freshly extracted values take
precedence per code, pre-existing values survive wherever nothing was re-extracted.

### Output Columns

In addition to the standard MEDS columns (`subject_id`, `time`, `code`, `numeric_value`),
MEDS-Extract adds these extension columns to the extracted data:

- **`code_components`**: A struct column with the individual source column values that
    were combined to form the code. For example, if `code: f"{$test_name}//{$units}"`,
    each row has `{test_name: "Glucose", units: "mg/dL"}`. Only present when the code
    expression references source columns (not for literals like `code: MEDS_BIRTH`).

- **`source_block`**: A string column tracking which MESSY config block produced each
    event, formatted as `"{file_prefix}/{event_name}"` (e.g., `"patients/eye_color"`,
    `"labs_vitals/lab"`). Useful for debugging and filtering events by origin.

The `metadata/codes.parquet` file also includes:

- **`code_template`**: The dftly expression string that produced each code (e.g.,
    `$test_name`). Every code has exactly one template — this is a pipeline-generated,
    reserved column (a `_metadata` block cannot redefine it), and distinct templates
    colliding on one code is a configuration error. Enables downstream tools to
    understand code structure without access to the original MESSY config.

## 🛠️ Troubleshooting

### Performance Optimization

- **Manually pre-shard your input data** if you have very large files. You can then configure your pipeline to
    skip the row-sharding stage and start directly with the `convert_to_subject_sharded` stage.
- **Use parallel processing** for faster extraction via the typical MEDs-Transforms parallelization
    options.

## Future Roadmap

1. Incorporating more of the pre-MEDS and joining logic that is common into this repository.
2. Automatic support for running in "demo mode" for testing and validation.
3. Better examples and documentation for common use cases, including incorporating data cleaning stages
    after the core extraction.
4. Providing a default runner or multiple default pipeline files for user convenience.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

MEDS Extract builds on the [MEDS-Transforms](https://github.com/mmcdermott/MEDS_transforms) framework and the [MEDS standard](https://medical-event-data-standard.github.io/). Special thanks to:

- The MEDS community for developing the standard
- Contributors to MEDS-Transforms for the underlying infrastructure
- Healthcare institutions sharing their data for research

## 📖 Citation

If you use MEDS Extract in your research, please cite:

```bibtex
@software{meds_extract2024,
  title={MEDS Extract: ETL Pipelines for Converting EHR Data to MEDS Format},
  author={McDermott, Matthew and contributors},
  year={2024},
  url={https://github.com/mmcdermott/MEDS_extract}
}
```

______________________________________________________________________

**Ready to standardize your EHR data?** Start with our [Quick Start](#-quick-start) guide or explore our [example](./example/) directory for a real, runnable configuration.
