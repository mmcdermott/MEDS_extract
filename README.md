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

> **Migrating from 0.6.x?** The 0.7.0 release is a breaking cut: MESSY config key names changed (unifying under `_defaults:` and `_table:`), the pipeline key naming the MESSY file is now `MESSY_config_fp` (was `event_conversion_config_fp`), null components in composite codes now drop rows unless you coalesce them, `codes.parquet` gained a deterministic stable schema, and `meds-extract-download` now handles raw-data fetching declaratively. There is no in-repo migration guide: 0.6.x configs must be ported by hand (the [Event Configuration Deep Dive](#-event-configuration-deep-dive) below covers the full 0.7 surface).

## 🚀 Quick Start

### 1. Install via `pip`:

```bash
pip install MEDS-extract
```

> [!NOTE]
> **0.7.0** pins `meds ~=0.4.0`, `MEDS-transforms >=0.6.7,<0.7`, and
> `dftly >=0.6.0`, and supports Python ≥ 3.11. The MESSY config schema changed for 0.7.0 — subject IDs are
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

- **File-based**: Data stored in `.csv`, `.csv.gz`, `.parquet`, or `.par` files. Pipeline input and
    output directories are local: if your raw data lives in the cloud, fetch it onto local disk first with
    `meds-extract-download` (see [below](#stage-your-raw-data)), whose `FsspecSource` supports any fsspec
    protocol (S3, GCS, Azure, ...) using ambient credentials.
- **Comprehensive Rows**: Each file contains a dataframe structure where each row contains all required
    information to produce one or more MEDS events at full temporal granularity, without additional joining or
    merging.
- **Integer subject IDs**: The `subject_id` column must contain integer values (`int64`). You can set
    `_defaults: {subject_id: hash($string_col)}` in your MESSY file to automatically convert string IDs to
    integers.

If these requirements are not met, you may need to perform some pre-processing steps to convert your raw data
into an accepted format, though typically these are very minor (e.g., joining across a join key, converting
time deltas into timestamps, etc.).

#### Stage your raw data

If your raw files live behind an HTTP endpoint, a PhysioNet release, a cloud bucket, or a local mirror, you
can declare them in a `sources:` block and let the bundled `meds-extract-download` CLI stage them onto local
disk (with checksum verification and resumable, rate-limit-polite transfers) instead of writing download
scripts by hand. See the
[download layer documentation](https://github.com/mmcdermott/MEDS_extract/blob/main/src/MEDS_extract/download/README.md)
for the source types and CLI usage.

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

This MESSY file is the heart of the MEDS Extract system; every stage that needs it reads it from the
pipeline's `MESSY_config_fp`.

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

Beyond your MESSY file, you also need to specify what pipeline stages you want to
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

# Points to the MESSY file defined above. Replace with a real path.
MESSY_config_fp: $MESSY_CONFIG
# The shards mapping is stored in the root of the final output directory.
shards_map_fp: ${output_dir}/metadata/.shards.json

stages:
  - convert_to_parquet
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
> 	MESSY_config_fp="$MESSY_CONFIG" \
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
`MEDS_transform-pipeline "$PIPELINE_YAML" --overrides MESSY_config_fp=/path/to/messy.yaml`.

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
...     f"MESSY_config_fp={tmpdir}/messy.yaml "
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

> [!NOTE]
> **Extraction de-duplicates.** Two source rows that produce byte-identical event rows —
> every extracted column equal — collapse into one event, so the same raw row reaching
> extraction twice (a re-run, an overlapping shard, a fan-out from a non-unique join
> target) can't inflate the cohort. Repeated measurements survive as long as *something*
> extracted distinguishes them (time, value, or a code component); a table that records
> the same value twice at the same timestamp with no distinguishing column extracted
> yields one event, not two.

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
Each block maps output column names to dftly expressions over the raw metadata table;
produced columns whose names match the code's raw components are the join keys. Lab
descriptions produce `test_name` (the code's only component — a full match), while
medication descriptions produce only `medication_name` of the code's two components —
a partial match that broadcasts the drug class to every dose-variant (see
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

## 🏃 Running a packaged dataset ETL

A dataset ETL package (e.g. `MIMIC_IV_MEDS`) can be **pure config**: a `pyproject.toml` plus one MESSY
YAML plus its test suite — zero Python — while remaining versioned and released on PyPI, CI-tested, and
CLI-runnable. The one YAML describes the entire ETL:

```yaml
sources: # where the raw data lives — including its release version
  dataset_version:
    dataset: '3.1'
    demo: '2.2'
  dataset:
    - type: physionet
      # or interpolate: .../files/mimiciv/${sources.dataset_version.dataset}
      base_url: https://physionet.org/files/mimiciv/3.1
      username: ${oc.env:PHYSIONET_USER}
      password: ${oc.env:PHYSIONET_PASSWORD}
  demo:
    - type: physionet
      base_url: https://physionet.org/files/mimic-iv-demo/2.2

hosp/admissions: # what to extract (the event-conversion tables)
  admission:
    code: f"HOSPITAL_ADMISSION//{$admission_type}"
    time: $admittime::"%Y-%m-%d %H:%M:%S"
  # ... etc
```

Note what's *absent*: no stage list, no runner config — for a registered dataset (below) the file
needs **no `etl:` block at all**. `meds-extract-run` always runs the canonical 8-stage extraction
pipeline (`convert_to_parquet` → `split_and_shard_subjects` → `convert_to_subject_sharded` →
`convert_to_MEDS_events` → `merge_to_MEDS_cohort` → `extract_code_metadata` →
`finalize_MEDS_metadata` → `finalize_MEDS_data`), the dataset name defaults to the registered
pipeline name, and the raw-data version comes from `sources.dataset_version`.

Two reserved pieces of MESSY schema make this work:

- **`sources.dataset_version`** — the raw release version is a property of the *source data* (it is
    baked into download URLs), so it lives inside `sources:`. Scalar (`dataset_version: "3.1"`) or
    per-bucket mapping (as above — demo and full releases genuinely differ). It is never treated as
    a bucket by `meds-extract-download`, it is interpolatable into source entries
    (`${sources.dataset_version}` / `${sources.dataset_version.demo}`), and `meds-extract-run` stamps
    the selected bucket's version into the output's `etl_metadata.dataset_version`.

- **`etl:`** — an optional block of identity fallbacks plus a curated, flat set of per-stage options
    (real stage-parameter names, no aliases, each mapped internally onto its stage):

    ```yaml
    etl:
      # Fallbacks — needed only when the registry / sources: can't supply them:
      dataset_name: MIMIC-IV # required for pkg://- and path-resolved specs only
      raw_dataset_version: '3.1' # required only if sources: declares no dataset_version;
      #   if both are present they must match (one source of truth)
      # Curated stage options (all optional):
      n_subjects_per_shard: 1000 # split_and_shard_subjects
      split_fracs: {train: 0.8, tuning: 0.1, held_out: 0.1}   # split_and_shard_subjects
      external_splits_json_fp: /path/to/splits.json # split_and_shard_subjects
      do_dedup_text_and_numeric: true # convert_to_MEDS_events
      description_separator: "\n" # extract_code_metadata
    ```

    Anything else under `etl:` is rejected at config load, listing the allowed keys.

`sources:` and `etl:` are **reserved top-level keys**: the event-conversion pipeline strips them before
parsing tables, `meds-extract-download` consumes only `sources:`, and `meds-extract-run` consumes both.

The package's `pyproject.toml` registers the dataset under the `MEDS_extract.pipelines` entry-point
group (the same registration pattern as `MEDS_transforms.stages`, one level up), pointing **directly at
the bundled MESSY file** in `<package.module>:<filename.yaml>` form:

```toml
[project.entry-points."MEDS_extract.pipelines"]
MIMIC-IV = "MIMIC_IV_MEDS.configs:event_configs.yaml"
```

The file resolves as `importlib.resources.files("MIMIC_IV_MEDS.configs") / "event_configs.yaml"` — the
registration names the file itself, so there is no bundled-layout convention to learn. A bare module
reference (no `:filename`) is an error. The entry point is never imported/executed — its value string
is parsed, not `load()`-ed.

With that in place, the whole ETL is one command:

```bash
meds-extract-run spec=MIMIC-IV output_dir=/data/mimic_meds                     # full dataset
meds-extract-run spec=MIMIC-IV output_dir=/tmp/demo_meds download_key=demo     # demo sources bucket
meds-extract-run spec=messy.yaml output_dir=... download_key=null input_dir=.. # unpackaged / pre-staged
```

`spec=` resolves down a three-rung ladder: a **registered name** (the entry-point group above), a
**`pkg://` reference** (`pkg://MIMIC_IV_MEDS.configs.event_configs.yaml` — the same syntax
`MEDS_transform-pipeline` uses), or a **filesystem path**. The runner is a thin orchestrator over the
two public CLIs — it shells out to each in turn (in-module invocation modes may come later, upstream):

1. spawns `meds-extract-download` to stage the selected `sources:` bucket (`download_key=` picks
    the bucket, `common` is always appended; `download_key=null` skips downloading entirely);
2. synthesizes a MEDS-transforms pipeline config — the canonical stage list plus the `etl:` block's
    curated options — with every value **inlined** (no env-var indirection), written to
    `<output_dir>/.meds_extract_run/pipeline.yaml` as self-contained provenance. Its
    `MESSY_config_fp` carries the **portable spec reference** (the `pkg://` form for
    registered/`pkg://` specs): every consumer of `MESSY_config_fp` — i.e. any stage run
    independently — accepts `pkg://` alongside filesystem paths;
3. spawns the pipeline runner on it, propagating its exit code. Both children are spawned as
    **`sys.executable -m <module>`** (`MEDS_transforms.runner` / `MEDS_extract.download.cli`), pinning
    them to this interpreter's environment with no console-script `PATH` resolution to mis-resolve —
    [MEDS_transforms#398](https://github.com/mmcdermott/MEDS_transforms/issues/398)'s failure class is
    gone by construction;
4. stamps `etl_metadata.dataset_name` and `etl_metadata.dataset_version` automatically (through the
    synthesized config): the name is `etl.dataset_name`, defaulting to the registered pipeline name for
    registry-resolved specs; the version is `{raw version}:{ETL package's installed version}`, where the
    raw version is the selected bucket's `sources.dataset_version` (or the `etl.raw_dataset_version`
    fallback) and the package version comes from the entry point's providing distribution — so version
    provenance needs zero code in the dataset package. For `pkg://`/path specs (no distribution to ask)
    the stamp is the raw version alone, or pass `dataset_version=` explicitly.

`output_dir` is where the final MEDS cohort lands (`data/`, `metadata/`). Raw data downloads into
`download_dest_dir=` (defaulting under `<output_dir>/.meds_extract_run/` — point it somewhere durable
to reuse raw data across runs) and is also the pipeline's input; download-free runs pass
`download_key=null input_dir=<pre-staged raw data>` instead. Run-internal artifacts (the synthesized
pipeline config, child logs) live under `<output_dir>/.meds_extract_run/`. Exit code is `0` on success
and non-zero on any failure (child exit codes propagate). The runnable
[`example/`](https://github.com/mmcdermott/MEDS_extract/tree/main/example) directory's `messy.yaml`
carries an `etl:` block, so you can try the runner immediately:

```bash
meds-extract-run spec=example/messy.yaml output_dir=/tmp/meds_example_meds download_key=null \
	input_dir=example/raw_data
```

#### Passing knobs through to the children

The runner is a shuttle, so each child's own options are forwarded rather than re-invented:

| Flag                          | Goes to                                     | For                                                                                                                                                   |
| ----------------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `stage_runner_fp=`            | `MEDS_transform-pipeline --stage_runner_fp` | **Parallelism.** A top-level `parallelize:` block in that file becomes every stage's default, and it can override `parallelize` or `script` per stage |
| `do_profile=true`             | `MEDS_transform-pipeline --do_profile`      | Hydra profiling of each stage                                                                                                                         |
| `overrides=[...]`             | `MEDS_transform-pipeline --overrides`       | Any pipeline-config key the synthesized config doesn't template (`seed`, pipeline-level `do_overwrite`, …)                                            |
| `download_concurrency=`       | `meds-extract-download concurrency=`        | Parallel transport streams — close to linear on PhysioNet                                                                                             |
| `download_continue_on_error=` | `meds-extract-download continue_on_error=`  | Don't let one bad file sink a multi-hour fetch                                                                                                        |

```bash
# 8 workers everywhere, a faster download, and a specific split seed
printf 'parallelize:\n  n_workers: 8\n  launcher: joblib\n' >runner.yaml
meds-extract-run spec=MIMIC-IV output_dir=/data/mimic_meds \
	stage_runner_fp=runner.yaml download_concurrency=8 "overrides=['seed=2']"
```

Parallelism is deliberately a *runner* argument rather than an `etl:` option: a worker count is a
property of the machine, not of the dataset, and a registered spec ships inside a wheel. Quote each
`overrides=` element — the values contain `=`, which Hydra's override grammar otherwise rejects.

### Custom pipeline shapes

The `etl:` block deliberately does not make the stage sequence configurable. If your ETL needs a
nonstandard shape — extra trailing stages, replacing `convert_to_parquet` for data that is already
normalized, custom stage wiring — use the standalone route, unchanged from the sections above: write a pipeline YAML (see
[`example/pipeline.yaml`](https://github.com/mmcdermott/MEDS_extract/blob/main/example/pipeline.yaml))
and run `MEDS_transform-pipeline` on it directly, with `meds-extract-download` staging the raw data
first if needed. `meds-extract-run` is sugar for the canonical case, not a replacement for that
route.

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
        component_column: $key_expr    # join key: name matches a code component
        output_column: $source_column  # metadata output (any dftly expression)
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
> [`example/messy.yaml`](https://github.com/mmcdermott/MEDS_extract/blob/main/example/messy.yaml) single-quotes the f-strings and the `::`/`as` casts
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
your choice, made per component. Drops are never silent: each event logs a WARNING summarizing how
many rows were dropped for a null `code` and how many for a null `time`:

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

#### Strict vs. lenient timestamp parsing

In a MESSY file, a `time` format cast is **strict** by default: if a value does not match the format string,
extraction **aborts** rather than silently guessing. Prefix the format with `?` (`as ?"fmt"`, or the
equivalent `::?"fmt"`) to parse **leniently** instead, so an unparsable value becomes a **null** timestamp —
and, since a non-static MEDS event may not have a null time, that row is **dropped**. Strict is the safe
default — it surfaces malformed source timestamps loudly; reach for lenient when a column is known to be
occasionally malformed and discarding those rows is acceptable:

```yaml
lab_results:
  lab:
    code: f"LAB//{$test_name}"
    # Strict (default): a malformed timestamp aborts the run.
    time: $result_time as "%Y-%m-%d %H:%M:%S"

  lab_lenient:
    code: f"LAB//{$test_name}"
    # Lenient ("?" prefix): a malformed timestamp nulls out, and the row is dropped.
    time: $result_time as ?"%Y-%m-%d %H:%M:%S"
```

The `?` is the only difference between the two `time` expressions above. Applying each event
configuration to a raw table — subject 2's timestamp is malformed:

```python
>>> import polars as pl
>>> from MEDS_extract.config import EventConfig
>>> raw = pl.DataFrame({
...     "subject_id": [1, 2],
...     "result_time": ["2021-01-01", "not-a-date"],  # subject 2's timestamp is malformed
...     "test_name": ["GLU", "HR"],
... })
>>> def times(time_expr):  # surviving rows for a `time` expression
...     ev = EventConfig.parse("lab", {"code": 'f"LAB//{$test_name}"', "time": time_expr})
...     return ev.extract(raw.lazy(), "lab_results/lab").collect().sort("subject_id").select(
...         "subject_id", "time", "code"
...     )
>>> # Lenient ("?" prefix): subject 2's timestamp nulls out, so that row is dropped.
>>> times('$result_time as ?"%Y-%m-%d"')
shape: (1, 3)
┌────────────┬────────────┬──────────┐
│ subject_id ┆ time       ┆ code     │
│ ---        ┆ ---        ┆ ---      │
│ i64        ┆ date       ┆ str      │
╞════════════╪════════════╪══════════╡
│ 1          ┆ 2021-01-01 ┆ LAB//GLU │
└────────────┴────────────┴──────────┘
>>> # Strict (the default, no "?"): the malformed timestamp aborts extraction.
>>> times('$result_time as "%Y-%m-%d"')
Traceback (most recent call last):
    ...
polars.exceptions.InvalidOperationError: conversion from `str` to `date` failed in column 'result_time' ...

```

Lenient drops are never silent: each event logs a WARNING with the counts, so a mis-specified format that
wipes out a whole table is immediately visible.

```text
`lab_results/lab`: dropped 1/2 rows with null time (unparsable or missing under the configured formats)
```

When a single column mixes several formats, don't choose between them — `coalesce` a lenient parse per
format and each row takes the first that matches (rows matching none are dropped, and counted):

```yaml
lab_results:
  lab:
    code: f"LAB//{$test_name}"
    time: coalesce($result_time::?"%m/%d/%y %H:%M", $result_time::?"%m/%d/%y")
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

#### Aggregated joins

A flat join fans out: one left row per matching right row. When you instead need a
*reduction* of the right table — the classic case is pulling the earliest `deathtime`
per subject out of an admissions table — write `cols` as a `{column: aggregation}`
mapping. The right side is grouped by the join key and each named column is reduced
before the (now one-to-at-most-one) left join:

```yaml
patients:
  _table:
    join:
      admissions:
        key: subject_id
        cols:
          deathtime: min # min per subject_id, joined as `deathtime`
```

Supported aggregations: `min`, `max`, `sum`, `mean`, `count`. All of them are
order-independent, so results don't depend on the order the right table's files are
scanned in (`first`/`last` are rejected for exactly that reason — use `min`/`max` over
an ordering column instead). A `cols` block is either all-flat (list) or
all-aggregated (mapping); mixing the two in one join is not supported.

Every aggregated join also logs a WARNING when the config is parsed: because the
aggregation folds multiple source rows into a single value, data errors (e.g.
conflicting values) are resolved silently rather than surfacing, and row-level
provenance cannot be traced through the reduction — use it knowingly.

The executable example below is the motivating MIMIC-IV shape: the earliest
per-subject `deathtime` from `admissions`, joined onto `patients` and feeding a death
event whose time coalesces the joined value with the patient table's own `dod`:

```python
>>> import polars as pl
>>> from MEDS_extract.config import TableConfig
>>> with yaml_disk('''
... patients.parquet:
...   subject_id: [1, 2, 3]
...   dod: [null, "2021-05-02", null]
... admissions.parquet:
...   subject_id: [1, 1, 2]
...   deathtime: ["2020-03-05", "2020-03-01", null]
... ''') as raw_dir:
...     tc = TableConfig.parse("patients", {
...         "_defaults": {"subject_id": "$subject_id"},
...         "_table": {
...             "join": {"admissions": {"key": "subject_id", "cols": {"deathtime": "min"}}},
...         },
...         "death": {"code": "MEDS_DEATH", "time": '($deathtime ?? $dod)::"%Y-%m-%d"'},
...     })
...     df = tc.prepare(tc.scan(raw_dir))  # scan applies the aggregated join
...     events = tc.events[0].extract(df, "patients/death").collect()
>>> events.sort("subject_id").select("subject_id", "code", "time")
shape: (2, 3)
┌────────────┬────────────┬────────────┐
│ subject_id ┆ code       ┆ time       │
│ ---        ┆ ---        ┆ ---        │
│ i64        ┆ str        ┆ date       │
╞════════════╪════════════╪════════════╡
│ 1          ┆ MEDS_DEATH ┆ 2020-03-01 │
│ 2          ┆ MEDS_DEATH ┆ 2021-05-02 │
└────────────┴────────────┴────────────┘

```

Subject 1 gets the *minimum* of their two admission death times; subject 2 has no
admission-side death time and falls back to `dod`; subject 3 has neither, so the row
is dropped (null-time accounting logs the drop).

**String-ordering caveat**: `min`/`max` on a String-typed column (which is what CSV
schema inference usually leaves datetime strings as) compares *lexicographically*.
That is correct for ISO-8601-style formats (`%Y-%m-%d ...`, as above) but silently
wrong for formats like `%m/%d/%Y` — `"03/01/2020" < "12/25/2019"` lexicographically.
The pipeline logs a warning whenever `min`/`max` aggregates a String column; make sure
the column's text ordering matches its temporal ordering, or use a typed (parquet)
source. `sum`/`mean` on a String column are rejected outright.

### Metadata linking, in depth

Datasets usually ship dictionary tables alongside the event data — `d_items.csv`,
`d_icd_diagnoses.csv`, a LOINC map. `_metadata` blocks link those tables to your
extracted codes, producing `metadata/codes.parquet`. The mental model:

- A `_metadata` entry is a small **dftly program over the raw metadata table**: a
    mapping of output column name → dftly expression, in the same expression language —
    with exactly the same semantics — as `code`/`time`. In particular, a bare, unquoted
    word is a string **literal**: `description: label` stamps the constant text
    `"label"` on every row; to read the raw `label` column write `description: $label`.
- Every extracted event row carries `code_components` — a struct of the **raw source
    values** the code was built from — and `source_block`, the MESSY block that produced
    it (see [Output Columns](#output-columns)).
- **Name matching decides the join**: produced columns whose names match the code's
    component columns are the **join keys**; every other produced column is metadata
    output attached to the matched codes. Producing every component is a full match;
    producing a subset is a partial match that broadcasts the metadata to every code
    sharing the produced keys.
- The `extract_code_metadata` stage attaches metadata by **joining those key values
    against the raw component values**, scoped to the event block that declared the
    `_metadata` entry.
- The assembled code *string* is never matched against. Your metadata tables keep their
    raw values as-is — you never mirror the code expression's prefixes, separators,
    casts, or `??` fallbacks inside a metadata table. When the raw representations
    *disagree* (a differently-named key column, a split key, a type mismatch), you
    reconcile them with an explicit dftly expression on the key (`itemid:   $omop_source_code`, `valueuom: $unit ?? $unit_alt`, `itemid: $itemid::str`).

Every example below is executable (it runs in CI): `yaml_disk` writes a small raw
dataset plus its MESSY file to disk, then the **real extraction pipeline** runs over it
— the same invocation as the [End-to-End Example](#-end-to-end-example) — and the
frames shown are read back from the files the pipeline produced:

```python
>>> def run_extraction(root: Path) -> None:
...     """Run the standard extraction pipeline over `root/raw` per `root/messy.yaml`."""
...     result = subprocess.run(
...         f"MEDS_transform-pipeline pkg://MEDS_extract.configs._extract.yaml --overrides "
...         f"input_dir={root}/raw output_dir={root}/output "
...         f"MESSY_config_fp={root}/messy.yaml dataset.name=DEMO dataset.version=1.0",
...         shell=True, capture_output=True,
...     )
...     assert result.returncode == 0, result.stderr.decode()[-1000:]

```

#### A worked dataset: full matches, partial matches, and null components

One dataset, two event tables, two dictionaries. `lab_dictionary` carries **both** of
the lab code's components (`test_name`, `units`) — including a row whose `units` cell
is null — so its entry produces both (a full match). `med_classes` is keyed on
`medication_name` alone, so its entry produces only that component (a partial match):

```python
>>> root = yaml_disk('''
... raw/:
...   labs.csv:
...     subject_id: [1, 1, 2, 3]
...     test_name: [GLU, CREAT, GLU, GLU]
...     units: [mg/dL, mg/dL, mg/dL, null]
...     ts: ["2024-01-01 09:30", "2024-01-01 09:35", "2024-03-02 14:00", "2024-04-01 08:15"]
...     result: [98.0, 1.1, 105.0, 6.1]
...   medications.csv:
...     subject_id: [1, 2, 3]
...     medication_name: [Metformin, Metformin, Lisinopril]
...     dose: [500 mg, 1000 mg, 10 mg]
...     ts: ["2024-02-01 08:00", "2024-03-05 09:00", "2024-04-02 09:00"]
...   lab_dictionary.csv:
...     test_name: [GLU, GLU, CREAT, NA]
...     units: [mg/dL, null, mg/dL, mmol/L]
...     label: [Glucose (serum), Glucose (no unit given), Creatinine (serum), Sodium (serum)]
...   med_classes.csv:
...     medication_name: [Metformin, Lisinopril]
...     drug_class: [Antidiabetic, ACE inhibitor]
... messy.yaml:
...   labs:
...     lab:
...       code: 'f"LAB//{$test_name}//{$units ?? ''UNK''}"'
...       time: '$ts::"%Y-%m-%d %H:%M"'
...       numeric_value: $result
...       _metadata:
...         lab_dictionary:
...           test_name: $test_name
...           units: $units
...           description: $label
...   medications:
...     med:
...       code: 'f"{$medication_name}//{$dose}"'
...       time: '$ts::"%Y-%m-%d %H:%M"'
...       _metadata:
...         med_classes:
...           medication_name: $medication_name
...           description: $drug_class
... ''', Path(tempfile.mkdtemp()))
>>> run_extraction(root)

```

The extracted events carry the raw component values the join will run against.
Unnesting `code_components` for the lab rows shows the `?? 'UNK'` fallback appearing
*only* in the code string — the raw null survives in the components:

```python
>>> data = pl.read_parquet(f"{root}/output/data/**/*.parquet")
>>> labs = data.filter(pl.col("source_block") == "labs/lab")
>>> labs.unnest("code_components").select("code", "test_name", "units").sort("code")
shape: (4, 3)
┌───────────────────┬───────────┬───────┐
│ code              ┆ test_name ┆ units │
│ ---               ┆ ---       ┆ ---   │
│ str               ┆ str       ┆ str   │
╞═══════════════════╪═══════════╪═══════╡
│ LAB//CREAT//mg/dL ┆ CREAT     ┆ mg/dL │
│ LAB//GLU//UNK     ┆ GLU       ┆ null  │
│ LAB//GLU//mg/dL   ┆ GLU       ┆ mg/dL │
│ LAB//GLU//mg/dL   ┆ GLU       ┆ mg/dL │
└───────────────────┴───────────┴───────┘
>>> meds = data.filter(pl.col("source_block") == "medications/med")
>>> meds.unnest("code_components").select("code", "medication_name", "dose").sort("code")
shape: (3, 3)
┌────────────────────┬─────────────────┬─────────┐
│ code               ┆ medication_name ┆ dose    │
│ ---                ┆ ---             ┆ ---     │
│ str                ┆ str             ┆ str     │
╞════════════════════╪═════════════════╪═════════╡
│ Lisinopril//10 mg  ┆ Lisinopril      ┆ 10 mg   │
│ Metformin//1000 mg ┆ Metformin       ┆ 1000 mg │
│ Metformin//500 mg  ┆ Metformin       ┆ 500 mg  │
└────────────────────┴─────────────────┴─────────┘

```

And the linked `metadata/codes.parquet`:

```python
>>> codes = pl.read_parquet(f"{root}/output/metadata/codes.parquet")
>>> codes.select("code", "description").sort("code")
shape: (6, 2)
┌────────────────────┬─────────────────────────┐
│ code               ┆ description             │
│ ---                ┆ ---                     │
│ str                ┆ str                     │
╞════════════════════╪═════════════════════════╡
│ LAB//CREAT//mg/dL  ┆ Creatinine (serum)      │
│ LAB//GLU//UNK      ┆ Glucose (no unit given) │
│ LAB//GLU//mg/dL    ┆ Glucose (serum)         │
│ Lisinopril//10 mg  ┆ ACE inhibitor           │
│ Metformin//1000 mg ┆ Antidiabetic            │
│ Metformin//500 mg  ┆ Antidiabetic            │
└────────────────────┴─────────────────────────┘

```

Everything in this frame follows from the component join:

- **Full match** (labs): the entry produced both `test_name` and `units`, so each
    code's `(test_name, units)` components matched those produced key columns. The `NA`
    dictionary row matched no observed code, so it does not appear — `codes.parquet`
    describes the codes your data actually contains.
- **Partial match** (medications): the entry produced only `medication_name`, so both
    dose-variants of Metformin got `Antidiabetic` from a dictionary that knows nothing
    about doses — the metadata broadcasts to every code sharing the produced key. Had
    the entry also produced `dose`, the join would have required `med_classes` to carry
    dose values too. Any subset of the components works — just produce the columns you
    want to key on.
- **Null components** (the `LAB//GLU//UNK` row): the join treats null as an ordinary
    key value, so the dictionary row whose `units` cell is null describes *specifically*
    the unit-less variant. Note the dictionary says `UNK` nowhere — it holds raw values,
    and the raw value here is null. A null key is **not** a wildcard: that row attached
    only to `LAB//GLU//UNK`, never to `LAB//GLU//mg/dL`. (If you instead want one
    description across *all* unit-variants of a test, produce only `test_name`.)

#### Metadata is scoped to the declaring event

Two events may reference same-named components with colliding values — an `itemid` in
`vitals` and an unrelated `itemid` in `labs`. A `_metadata` block only ever attaches to
codes from the event block that declared it (that is what `source_block` is for):

```python
>>> root = yaml_disk('''
... raw/:
...   vitals.csv:
...     subject_id: [1, 2, 3]
...     itemid: [220045, 220045, 220179]
...   labs.csv:
...     subject_id: [1, 2, 3]
...     itemid: [220045, 220045, 220045]
...   d_vitals.csv:
...     itemid: [220045, 220179]
...     label: [Heart Rate, NBP systolic]
... messy.yaml:
...   vitals:
...     vital:
...       code: 'f"VITAL//{$itemid}"'
...       time:
...       _metadata:
...         d_vitals:
...           itemid: $itemid
...           description: $label
...   labs:
...     lab:
...       code: 'f"LAB//{$itemid}"'
...       time:
... ''', Path(tempfile.mkdtemp()))
>>> run_extraction(root)
>>> data = pl.read_parquet(f"{root}/output/data/**/*.parquet")
>>> data.unnest("code_components").select("code", "itemid", "source_block").unique().sort("code")
shape: (3, 3)
┌───────────────┬────────┬──────────────┐
│ code          ┆ itemid ┆ source_block │
│ ---           ┆ ---    ┆ ---          │
│ str           ┆ i64    ┆ str          │
╞═══════════════╪════════╪══════════════╡
│ LAB//220045   ┆ 220045 ┆ labs/lab     │
│ VITAL//220045 ┆ 220045 ┆ vitals/vital │
│ VITAL//220179 ┆ 220179 ┆ vitals/vital │
└───────────────┴────────┴──────────────┘
>>> pl.read_parquet(f"{root}/output/metadata/codes.parquet").select("code", "description").sort("code")
shape: (3, 2)
┌───────────────┬──────────────┐
│ code          ┆ description  │
│ ---           ┆ ---          │
│ str           ┆ str          │
╞═══════════════╪══════════════╡
│ LAB//220045   ┆ null         │
│ VITAL//220045 ┆ Heart Rate   │
│ VITAL//220179 ┆ NBP systolic │
└───────────────┴──────────────┘

```

`LAB//220045` shares the component value but not the declaring block, so it receives
nothing — vocabulary declared for one event never leaks onto another. (The code itself
still appears — `codes.parquet` always enumerates every observed code, as MEDS
requires — it just carries no metadata.)

#### Raw values, not rendered values

Two things routinely differ between what a code *displays* and what the raw data
*contains*, and the join always sides with the raw data:

1. **Dtypes.** Component dtypes come from your raw event files and metadata dtypes
    from the metadata files. Below, the csv `itemid` infers as an integer while the
    parquet dictionary types it as a float — the classic pandas-heritage shape where a
    nullable integer column became `220045.0`. Both sides of the join are normalized
    through one canonical String rendering (integer-valued floats render via `Int64`,
    so `220045.0` matches `220045`; non-integer floats keep their float rendering —
    `1.5` only matches `"1.5"`).
2. **Transforms.** A code expression may transform its components — casts,
    `substring`, arithmetic. The join still runs on the raw component values, so the
    dictionary stays keyed on what the raw data contains, not on what the code shows.
    Below, the code keeps only the 3-character ICD-10 category, yet the full raw
    `icd_code` is what matches.

```python
>>> root = yaml_disk('''
... raw/:
...   vitals.csv:
...     subject_id: [1, 2, 3]
...     itemid: [220045, 220179, 220045]
...   diagnoses.csv:
...     subject_id: [1, 2, 3]
...     icd_code: [E119, I10, E119]
...   d_items.parquet:
...     itemid: [220045.0, 220179.0]
...     label: [Heart Rate, NBP systolic]
...   d_icd.csv:
...     icd_code: [E119, I10, E11]
...     long_title: [Type 2 diabetes, Essential hypertension, Should never match]
... messy.yaml:
...   vitals:
...     vital:
...       code: 'f"VITAL//{$itemid}"'
...       time:
...       _metadata:
...         d_items:
...           itemid: $itemid
...           description: $label
...   diagnoses:
...     dx:
...       code: 'f"DX//{substring($icd_code, 0, 3)}"'
...       time:
...       _metadata:
...         d_icd:
...           icd_code: $icd_code
...           description: $long_title
... ''', Path(tempfile.mkdtemp()))
>>> run_extraction(root)

```

The components keep their raw dtypes and raw values — `itemid` is an `Int64` and
`icd_code` holds the full, untruncated code:

```python
>>> data = pl.read_parquet(f"{root}/output/data/**/*.parquet")
>>> data.schema["code_components"]
Struct({'icd_code': String, 'itemid': Int64})
>>> data.unnest("code_components").select("code", "itemid", "icd_code").unique().sort("code")
shape: (4, 3)
┌───────────────┬────────┬──────────┐
│ code          ┆ itemid ┆ icd_code │
│ ---           ┆ ---    ┆ ---      │
│ str           ┆ i64    ┆ str      │
╞═══════════════╪════════╪══════════╡
│ DX//E11       ┆ null   ┆ E119     │
│ DX//I10       ┆ null   ┆ I10      │
│ VITAL//220045 ┆ 220045 ┆ null     │
│ VITAL//220179 ┆ 220179 ┆ null     │
└───────────────┴────────┴──────────┘
>>> pl.read_parquet(f"{root}/output/metadata/codes.parquet").select("code", "description").sort("code")
shape: (4, 2)
┌───────────────┬────────────────────────┐
│ code          ┆ description            │
│ ---           ┆ ---                    │
│ str           ┆ str                    │
╞═══════════════╪════════════════════════╡
│ DX//E11       ┆ Type 2 diabetes        │
│ DX//I10       ┆ Essential hypertension │
│ VITAL//220045 ┆ Heart Rate             │
│ VITAL//220179 ┆ NBP systolic           │
└───────────────┴────────────────────────┘

```

The float-typed `220045.0` dictionary row linked to `VITAL//220045`, and `DX//E11` got
its description from the row keyed on the raw `E119` — while the decoy row keyed on
`E11`, the *transformed* value that appears in the code string, matched nothing. You
never replicate a code expression's transforms in a metadata table.

#### Sourcing and normalizing join keys

Because keys are expressions, reconciling naming or representation differences between
your metadata table and your event data is part of the block itself. Sourcing the
`itemid` key from a dictionary column named `omop_source_code` is just a rename
expression, a literal is a quoted string, and `??`/casts normalize values the join
should agree on:

```python
>>> root = yaml_disk('''
... raw/:
...   vitals.csv:
...     subject_id: [1, 2, 3]
...     itemid: [220045, 220179, 220045]
...   d_items.csv:
...     omop_source_code: [220045, 220179]
...     label: [Heart Rate, NBP systolic]
... messy.yaml:
...   vitals:
...     vital:
...       code: 'f"VITAL//{$itemid}"'
...       time:
...       _metadata:
...         d_items:
...           itemid: $omop_source_code # key sourced from a differently-named column
...           description: $label
...           vocab: '"OMOP"' # quoted -> a string literal, not a column
... ''', Path(tempfile.mkdtemp()))
>>> run_extraction(root)
>>> pl.read_parquet(f"{root}/output/metadata/codes.parquet").select(
...     "code", "description", "vocab"
... ).sort("code")
shape: (2, 3)
┌───────────────┬──────────────┬───────────┐
│ code          ┆ description  ┆ vocab     │
│ ---           ┆ ---          ┆ ---       │
│ str           ┆ str          ┆ list[str] │
╞═══════════════╪══════════════╪═══════════╡
│ VITAL//220045 ┆ Heart Rate   ┆ ["OMOP"]  │
│ VITAL//220179 ┆ NBP systolic ┆ ["OMOP"]  │
└───────────────┴──────────────┴───────────┘

```

#### What errors, and why

Metadata linking is validated when the MESSY config is parsed — at load time, in every
stage and worker, before any data is joined. The checks live in
`compile_metadata_block`, the one function that compiles a `_metadata` block (config
parsing validates through it at construction, and the stage compiles each entry through
it exactly once). A `_metadata` block on a **literal** code is rejected — a literal
references no source columns, so there are no components to match on:

```python
>>> from MEDS_extract.config import compile_metadata_block
>>> compile_metadata_block(
...     {"description": "$label"}, set(), code_template_str="MEDS_BIRTH"
... )
Traceback (most recent call last):
    ...
ValueError: The code expression 'MEDS_BIRTH' is a literal: it references no source columns, ...

```

A block must produce at least one component-named column — with none, there is no join
key, and the error lists the components the event offers:

```python
>>> compile_metadata_block(
...     {"description": "$drug_class"},
...     {"medication_name"},
...     code_template_str="$medication_name",
... )
Traceback (most recent call last):
    ...
ValueError: _metadata block produces no join-key columns: none of its produced column names
['description'] match the code expression's component columns. At least one produced column
must be named after a component to serve as a join key. Component columns available on this
event: ['medication_name'] (from code expression '$medication_name').

```

And `code` / `code_template` are pipeline-generated output names a block may not
redefine (`code` is allowed only as a *join key*, when the code expression references
a source column literally named `code` — the ICD/OMOP vocabulary-table shape):

```python
>>> compile_metadata_block(
...     {"itemid": "$itemid", "code": "$label"},
...     {"itemid"},
...     code_template_str='f"CHART//{$itemid}"',
... )
Traceback (most recent call last):
    ...
ValueError: _metadata output column name(s) ['code'] are reserved: ...

```

#### The shape of `codes.parquet`

The reduced output has a canonical, data-independent shape. One event may declare
several `_metadata` entries (and several metadata rows can collapse onto one code), so
every column is aggregated per code:

- **`description`**: a single String — distinct values from all sources, joined with
    the stage's `description_separator` (default: newline) in config order.
- **`parent_codes`**: `List(String)` of distinct `vocabulary/code` strings, unioned
    across metadata rows and sources.
- **`code_template`**: a single String (one code, one template — see
    [Output Columns](#output-columns)).
- **any other column** (extras like `loinc` below): `List(String)` of distinct values,
    sorted. Missing values are null, never `[]` or `""`.

`parent_codes` is an ordinary dftly output expression. Each metadata *row* yields at
most one parent (a nullable String); the reducer unions parents across rows and
sources into the per-code list. Multi-case vocabulary mappings are chained
conditionals — Python-style `<then> if <condition> else <then> if <condition>` — and
omitting the final `else` yields a real null for rows matching no case (write `$col`
inside conditions and f-strings; a bare `null` would be the *string* `"null"`):

```yaml
parent_codes: >-
  f"ICD{$icd_version}CM/{$icd_code}" if $icd_version == "9"
  else f"ICD{$icd_version}CM/{$icd_code}" if $icd_version == "10"
```

Here a local dictionary (two rows for `GLU`, i.e. non-unique by key) and a LOINC
ontology both describe the same code; `parent_codes` is an unconditional f-string:

```python
>>> root = yaml_disk('''
... raw/:
...   labs.csv:
...     subject_id: [1, 2, 3]
...     test_name: [GLU, GLU, GLU]
...   local_dictionary.csv:
...     test_name: [GLU, GLU]
...     label: [Serum glucose, Serum glucose]
...     loinc_code: [2345-7, 2339-0]
...   loinc_ontology.csv:
...     test_name: [GLU]
...     long_name: [Glucose in Serum or Plasma]
...     loinc_code: [2345-7]
... messy.yaml:
...   labs:
...     lab:
...       code: $test_name
...       time:
...       _metadata:
...         local_dictionary:
...           test_name: $test_name
...           description: $label
...           loinc: $loinc_code
...         loinc_ontology:
...           test_name: $test_name
...           description: $long_name
...           parent_codes: 'f"LOINC/{$loinc_code}"'
... ''', Path(tempfile.mkdtemp()))
>>> run_extraction(root)
>>> codes = pl.read_parquet(f"{root}/output/metadata/codes.parquet")
>>> with pl.Config(fmt_str_lengths=60, tbl_width_chars=120):
...     print(codes)
shape: (1, 5)
┌──────┬────────────────────────────┬──────────────────┬───────────────┬──────────────────────┐
│ code ┆ description                ┆ parent_codes     ┆ code_template ┆ loinc                │
│ ---  ┆ ---                        ┆ ---              ┆ ---           ┆ ---                  │
│ str  ┆ str                        ┆ list[str]        ┆ str           ┆ list[str]            │
╞══════╪════════════════════════════╪══════════════════╪═══════════════╪══════════════════════╡
│ GLU  ┆ Serum glucose              ┆ ["LOINC/2345-7"] ┆ $test_name    ┆ ["2339-0", "2345-7"] │
│      ┆ Glucose in Serum or Plasma ┆                  ┆               ┆                      │
└──────┴────────────────────────────┴──────────────────┴───────────────┴──────────────────────┘
>>> dict(codes.schema)
{'code': String, 'description': String, 'parent_codes': List(String),
 'code_template': String, 'loinc': List(String)}

```

The two `loinc` values from the non-unique dictionary rows aggregated into one sorted
list, both sources' descriptions joined in config order, and the single template landed
as a plain String.

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

- **Convert very large inputs to parquet ahead of time** if you re-run the pipeline often.
    `convert_to_parquet` hardlinks a parquet source instead of rewriting it when the file carries only
    the columns your MESSY config reads — so prune pre-converted files to those columns to make the
    ingest stage effectively free (an unpruned parquet is rewritten with the projection applied). It is
    not required — the stage converts csv/csv.gz in bounded memory regardless.
- **Use parallel processing** for faster extraction via the typical MEDS-Transforms parallelization
    options.

## Future Roadmap

1. Incorporating more of the common pre-MEDS logic into this repository (table joins — including
    aggregated joins — landed in 0.7.0).
2. Automatic support for running in "demo mode" for testing and validation.
3. Better examples and documentation for common use cases, including incorporating data cleaning stages
    after the core extraction.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/mmcdermott/MEDS_extract/blob/main/CONTRIBUTING.md) for more details.

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

**Ready to standardize your EHR data?** Start with our [Quick Start](#-quick-start) guide or explore our [example](https://github.com/mmcdermott/MEDS_extract/tree/main/example) directory for a real, runnable configuration.
