<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="static/logo_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="static/logo_light.svg">
    <img width="520" height="200" alt="MEDS Logot" src="static/logo_light.svg">
  </picture>
</p>

# MEDS-Extract

[![PyPI - Version](https://img.shields.io/pypi/v/MEDS-extract)](https://pypi.org/project/MEDS-extract/)
![python](https://img.shields.io/badge/-Python_3.12-blue?logo=python&logoColor=white)
[![MEDS v0.4](https://img.shields.io/badge/MEDS-0.4-blue)](https://medical-event-data-standard.github.io/)
[![Documentation Status](https://readthedocs.org/projects/meds-extract/badge/?version=latest)](https://meds-extract.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/mmcdermott/MEDS_extract/graph/badge.svg?token=5RORKQOZF9)](https://codecov.io/gh/mmcdermott/MEDS_extract)
[![tests](https://github.com/mmcdermott/MEDS_extract/actions/workflows/tests.yaml/badge.svg)](https://github.com/mmcdermott/MEDS_extract/actions/workflows/tests.yml)
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

## 🚀 Quick Start

### 1. Install via `pip`:

```bash
pip install MEDS-extract
```

> [!NOTE]
> MEDS Extract v0.2.0 uses meds v0.3.3 and MEDS transforms v0.4.0. MEDS Extract v0.3.0 uses meds v0.4.0 and
> MEDS v0.5.0. Hotfixes will be released within those namespaces as required. Older versions may be supported
> in the v0.1.0 namespace.

> [!WARNING]
> **Breaking change in v0.6.0**: The MESSY event configuration syntax has changed significantly. Event
> field expressions (e.g., `code` and `time`) are now parsed by
> [dftly](https://github.com/mmcdermott/dftly), a lightweight declarative expression language. The old
> `col()` function syntax and list-based code construction are no longer supported. The `time_format` key
> has been replaced by inline type casting with the `as` operator (e.g., `timestamp as "%Y-%m-%d"`).
> See the [Event Configuration Deep Dive](#-event-configuration-deep-dive) for the updated syntax.

### 2. Prepare your raw data

Ensure your data meets these requirements:

- **File-based**: Data stored in `.csv`, `.csv.gz`, or `.parquet` files. These may be stored locally or in the
    cloud, though intermediate processing currently must be done locally.
- **Comprehensive Rows**: Each file contains a dataframe structure where each row contains all required
    information to produce one or more MEDS events at full temporal granularity, without additional joining or
    merging.
- **Integer subject IDs**: The `subject_id` column must contain integer values (`int64`). You can use
    `subject_id_expr: "hash($string_col)"` in your MESSY file to automatically convert string IDs to integers.

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
# Global subject ID column (can be overridden per file)
subject_id_col: patient_id

# File-level configurations
patients:
  subject_id_col: MRN # This file has a different subject ID column
  demographics: # One kind of event in this file.
    code: DEMOGRAPHIC//{$gender}
    time:       # Static event
    race: race
    ethnicity: ethnicity

admissions:
  admission: # One kind of event in this file.
    code: HOSPITAL_ADMISSION//{$admission_type}
    time: admit_datetime as "%Y-%m-%d %H:%M:%S"
    department: department # Extra columns get tracked
    insurance: insurance

  discharge: # A different kind of event in this file.
    code: HOSPITAL_DISCHARGE//{$discharge_location}
    time: discharge_datetime as "%Y-%m-%d %H:%M:%S"

lab_results:
  lab:
    code: LAB//{$test_name}//{$units}
    time: result_datetime as "%Y-%m-%d %H:%M:%S"
    numeric_value: result_value # This will get converted to a numeric
    text_value: result_text # This will get converted to a string
```

This file is also called the "Event conversion configuration file" and is the heart of the MEDS Extract
system.

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

# Points to the event conversion yaml file defined above.
event_conversion_config_fp: ???
# The shards mapping is stored in the root of the final output directory.
shards_map_fp: ${output_dir}/metadata/.shards.json

# Used if you need to load input files from cloud storage.
cloud_io_storage_options: {}

stages:
  - shard_events:
      data_input_dir: ${input_dir}
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
> A pipeline with these defaults is provided in `MEDS_extract.configs._extract`.
> You can reference it directly using the package path with the `pkg://` prefix
> in the runner command:
> `MEDS_transform-pipeline pipeline_config_fp=pkg://MEDS_extract.configs._extract`
> This avoids needing a local copy on disk.

### 5. Run the extraction pipeline

MEDS-Extract does not have a stand-alone CLI runner; instead, you run it via the default MEDS-Transforms
pipeline, but you specify your own pipeline configuration file via the package syntax.

```bash
MEDS_transform-pipeline pipeline_config_fp="$PIPELINE_YAML"
```

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
>>> _ = shutil.copy("example/event_cfg.yaml", tmpdir)
>>> result = subprocess.run(
...     f"MEDS_transform-pipeline "
...     f"pkg://MEDS_extract.configs._extract.yaml "
...     f"--overrides "
...     f"input_dir={tmpdir}/raw_data "
...     f"output_dir={tmpdir}/output "
...     f"event_conversion_config_fp={tmpdir}/event_cfg.yaml "
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

The event config includes a `_metadata` block that links lab events to a descriptions
file. The extracted metadata maps each lab code to its human-readable description:

```python
>>> codes = pl.read_parquet(output / "metadata" / "codes.parquet")
>>> codes.sort("code").head(5)
shape: (5, 2)
┌─────────────────────┬──────────────────────────┐
│ code                ┆ description              │
│ ---                 ┆ ---                      │
│ str                 ┆ str                      │
╞═════════════════════╪══════════════════════════╡
│ ALT (U/L)           ┆ Alanine aminotransferase │
│ Creatinine (mg/dL)  ┆ Serum creatinine         │
│ Diastolic BP (mmHg) ┆ Diastolic blood pressure │
│ Glucose (mg/dL)     ┆ Blood glucose level      │
│ Heart Rate (bpm)    ┆ Heart rate / pulse       │
└─────────────────────┴──────────────────────────┘
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
    property_name: column_name  # Additional properties to extract
```

All `code` and `time` values are parsed as [dftly](https://github.com/mmcdermott/dftly) expressions.
dftly is a lightweight declarative expression language for data transformations. The key syntax elements
are:

- **Column references**: bare column names (e.g., `test_name`) or `$`-prefixed names (e.g., `$test_name`)
- **String literals**: quoted values (e.g., `"ADMISSION"`)
- **String interpolation**: curly braces for column values (e.g., `"LAB//{$test_name}//{$units}"`)
- **Type casting**: the `as` operator (e.g., `timestamp as "%Y-%m-%d"` to parse a datetime)
- **Arithmetic**: `$a + $b`, `$val * 2`
- **Hashing**: `hash($mrn)` for converting string IDs to integers

### Code Construction

Event codes can be built in several ways:

```yaml
# Simple string literal
vitals:
  heart_rate:
    code: "HEART_RATE"

# Column reference
vitals:
  heart_rate:
    code: measurement_type

# Composite codes with string interpolation (joined with "//")
vitals:
  heart_rate:
    code: "VITAL_SIGN//{$measurement_type}//{$units}"
```

### Time Handling

```yaml
# Simple datetime column (auto-parsed)
lab_results:
  lab:
    time: result_time

# With explicit format via type casting
lab_results:
  lab:
    time: result_time as "%m/%d/%Y %H:%M"

# Static events (no time)
demographics:
  gender:
    time: null
```

### Subject ID Configuration

```yaml
# Global default
subject_id_col: patient_id

# File-specific override
admissions:
  subject_id_col: hadm_id
  admission:
    code: ADMISSION
    # ...

# Hash a string column into an integer subject ID (uses dftly expression)
patients:
  subject_id_expr: hash($MRN)
  demographics:
    code: DEMOGRAPHIC
    time:
```

### Joining Tables

Sometimes subject identifiers are stored in a separate table from the events
you wish to extract. You can specify a join within the event configuration so
that the necessary columns are merged before extraction.

```yaml
vitals:
  join:
    input_prefix: stays
    left_on: stay_id
    right_on: stay_id
    columns_from_right:
      - subject_id
  subject_id_col: subject_id
  HR:
    code: HR
    time: charttime as "%m/%d/%Y %H:%M:%S"
    numeric_value: HR
```

### Metadata Linking

For datasets with separate metadata tables:

```yaml
lab_results:
  lab:
    code: LAB//{$itemid}
    time: charttime
    numeric_value: valuenum
    _metadata:
      input_file: d_labitems
      code_columns:
        - itemid
      properties:
        label: label
        fluid: fluid
        category: category
```

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

**Ready to standardize your EHR data?** Start with our [Quick Start](#-quick-start) guide or explore our [examples](./examples/) directory for real-world configurations.
