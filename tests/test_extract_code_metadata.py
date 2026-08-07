"""Stage-level tests for ``extract_code_metadata`` — end-to-end runs of the real stage.

Single-function behavior (code construction, key/output classification of ``_metadata``
blocks, dftly compilation) is doctested on the helpers themselves (``extract_metadata``,
``compile_metadata_block``, ``EventConfig.extract``). This
file covers behavior that needs the full mapper/reducer machinery: the component join
that attaches extracted metadata onto codes (including null-key matching and
partial-match broadcasting), its per-event scoping, dftly key renaming/normalization,
reducer determinism and the canonical output schema, merging with a pre-existing
``codes.parquet``, and reducer/worker concurrency.

Most tests run through :func:`_run_ecm_scenario`, which lays out synthetic event shards
and raw metadata files and invokes the stage's ``main_fn`` as worker 0. Event frames
carry the ``code_components`` struct and ``source_block`` tag exactly as
``convert_to_MEDS_events`` emits them — the component join runs against those columns.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path

import polars as pl
import pytest
from omegaconf import OmegaConf

_ = pl.Config.set_tbl_width_chars(600)


def _make_cfg(overrides: dict) -> OmegaConf:
    """Minimal DictConfig mimicking what MEDS-Transforms hands a stage (worker 0)."""
    base = {
        "do_overwrite": True,
        "seed": 1,
        "worker": 0,
        "polling_time": 0.1,
        "stage": "test",
        "stage_cfg": {},
        "etl_metadata": {
            "dataset_name": "TEST",
            "dataset_version": "1.0",
            "package_name": "MEDS_extract",
            "package_version": "0.0.0",
        },
    }
    base.update(overrides)
    cfg = OmegaConf.create(base)
    OmegaConf.set_struct(cfg, False)
    return cfg


def _run_ecm_scenario(
    root: Path,
    messy_yaml: str,
    event_frames: dict[str, pl.DataFrame],
    raw_files: dict[str, str | pl.DataFrame],
    existing_codes: pl.DataFrame | None = None,
    description_separator: str = "\n",
    worker: int = 0,
) -> pl.DataFrame | None:
    """Run the extract_code_metadata stage over synthetic event shards and raw metadata files.

    ``event_frames`` maps parquet basenames to frames of extra columns (code, code_components,
    source_block, ...); the standard subject_id/time/numeric_value columns are added here.
    ``raw_files`` maps raw metadata file paths (which may include subdirectories) to either
    text content (written verbatim) or a DataFrame (written as parquet). ``existing_codes``,
    when given, is written as a pre-existing ``metadata/codes.parquet`` for the reducer to
    merge with. ``worker`` selects the MR worker id (worker 0 is the reducer; any other id
    runs the map phase only). Returns the reduced ``codes.parquet`` as a DataFrame, or
    ``None`` if the stage exited without writing one (a non-reducer worker).
    """
    from MEDS_extract.extract_code_metadata.extract_code_metadata import main as ecm_stage

    events_dir = root / "events" / "train" / "0"
    events_dir.mkdir(parents=True)
    for basename, frame in event_frames.items():
        n = len(frame)
        frame.with_columns(
            subject_id=pl.Series(range(1, n + 1), dtype=pl.Int64),
            time=pl.lit(None, dtype=pl.Datetime("us")),
            numeric_value=pl.lit(None, dtype=pl.Float32),
        ).write_parquet(events_dir / f"{basename}.parquet")

    raw_dir = root / "raw"
    raw_dir.mkdir()
    for fname, content in raw_files.items():
        fp = raw_dir / fname
        fp.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, pl.DataFrame):
            content.write_parquet(fp)
        else:
            fp.write_text(content)

    metadata_in = root / "empty_meta"
    if existing_codes is not None:
        metadata_in = root / "metadata_in" / "metadata"
        metadata_in.mkdir(parents=True)
        existing_codes.write_parquet(metadata_in / "codes.parquet", use_pyarrow=True)

    event_cfg_fp = root / "messy.yaml"
    event_cfg_fp.write_text(messy_yaml)
    shards_fp = root / "metadata" / ".shards.json"
    shards_fp.parent.mkdir(parents=True)
    shards_fp.write_text(json.dumps({"train/0": [1]}))

    out_dir = root / "metadata_out" / "metadata"
    out_dir.mkdir(parents=True)

    cfg = _make_cfg(
        {
            "worker": worker,
            "input_dir": str(raw_dir),
            "stage_cfg": {
                "data_input_dir": str(root / "events"),
                "output_dir": str(out_dir),
                "metadata_input_dir": str(metadata_in),
                "reducer_output_dir": str(out_dir),
                "description_separator": description_separator,
            },
            "MESSY_config_fp": str(event_cfg_fp),
            "shards_map_fp": str(shards_fp),
        }
    )
    ecm_stage.main_fn(cfg)
    codes_fp = out_dir / "codes.parquet"
    return pl.read_parquet(codes_fp) if codes_fp.exists() else None


def _bare_code_events(code_col: str, codes: list[str], source_block: str) -> pl.DataFrame:
    """Event frame as ``convert_to_MEDS_events`` emits for a bare-column code (``$<code_col>``).

    The code string equals the single component value, the ``code_components`` struct
    carries that value under the source column's name, and every row is stamped with the
    declaring ``source_block``.
    """
    return pl.DataFrame(
        {
            "code": codes,
            "code_components": [{code_col: c} for c in codes],
            "source_block": [source_block] * len(codes),
        }
    )


# ── Metadata joining basics: attaching metadata onto extracted codes ──


def test_extract_code_metadata_multiple_files_per_prefix():
    """Multiple CSV files matching one metadata prefix are concatenated before extraction."""
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      lab_meta:
        lab_code: $lab_code
        description: $title
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": _bare_code_events("lab_code", ["HR", "TEMP"], "data/measurement")},
            # Two CSV files in a sub-sharded `lab_meta/` directory — triggers multi-file concat.
            raw_files={
                "lab_meta/part1.csv": "lab_code,title\nHR,Heart Rate\n",
                "lab_meta/part2.csv": "lab_code,title\nTEMP,Body Temperature\n",
            },
        )

    by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
    assert by_code["HR"] == "Heart Rate"
    assert by_code["TEMP"] == "Body Temperature"


def test_extract_code_metadata_parquet_source():
    """Regression guard: a ``_metadata`` prefix resolving to a parquet file must read cleanly.

    Reader kwargs are chosen per metadata prefix: csv sources get the csv-only
    ``infer_schema=False`` (all-String), while parquet sources take no reader kwargs and
    keep their intrinsic types. An unconditional ``infer_schema=False`` used to crash
    ``pl.scan_parquet`` with ``TypeError: ... unexpected keyword argument 'infer_schema'``
    (https://github.com/mmcdermott/MEDS_extract/issues/148). Mirrors
    MIMIC-IV's pre-MEDS ``hosp/d_icd_diagnoses.parquet`` shape.
    """
    messy = """\
diagnoses_icd:
  diagnosis:
    code: 'f"ICD{$icd_version}//{$icd_code}"'
    _metadata:
      d_icd_diagnoses:
        icd_code: $icd_code
        icd_version: $icd_version
        description: $long_title
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "diagnoses_icd": pl.DataFrame(
                    {
                        "code": ["ICD9//25000", "ICD10//E119"],
                        # String components (as CSV-sourced events carry) against the
                        # typed Int64 parquet metadata below — the join's canonical
                        # String normalization must bridge the two.
                        "code_components": [
                            {"icd_code": "25000", "icd_version": "9"},
                            {"icd_code": "E119", "icd_version": "10"},
                        ],
                        "source_block": ["diagnoses_icd/diagnosis"] * 2,
                    }
                )
            },
            raw_files={
                "d_icd_diagnoses.parquet": pl.DataFrame(
                    {
                        "icd_code": ["25000", "E119"],
                        "icd_version": [9, 10],
                        "long_title": [
                            "Diabetes mellitus without mention of complication",
                            "Type 2 diabetes mellitus without complications",
                        ],
                    }
                )
            },
        )

    by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
    assert by_code["ICD9//25000"] == "Diabetes mellitus without mention of complication"
    assert by_code["ICD10//E119"] == "Type 2 diabetes mellitus without complications"


def test_extract_code_metadata_duplicate_codes_aggregation():
    """Description concatenation for duplicate codes from multiple metadata sources.

    Two different _metadata blocks (source_a, source_b) both produce a "description" column for the same code
    "HR". The reducer must join them with the configured separator, in config order (deterministically).
    """
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      source_a:
        lab_code: $lab_code
        description: $title_a
      source_b:
        lab_code: $lab_code
        description: $title_b
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": _bare_code_events("lab_code", ["HR"], "data/measurement")},
            raw_files={
                "source_a.csv": "lab_code,title_a\nHR,Heart Rate\n",
                "source_b.csv": "lab_code,title_b\nHR,Pulse Rate\n",
            },
            description_separator="; ",
        )

    hr_rows = codes_df.filter(pl.col("code") == "HR")
    assert len(hr_rows) == 1
    assert hr_rows["description"][0] == "Heart Rate; Pulse Rate"


def test_extract_code_metadata_duplicate_codes_no_description():
    """Aggregation of duplicate codes when metadata has no description column.

    Non-description metadata columns aggregate to the canonical sorted List(String) shape rather than being
    separator-joined.
    """
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      source_a:
        lab_code: $lab_code
        custom_prop: $val_a
      source_b:
        lab_code: $lab_code
        custom_prop: $val_b
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": _bare_code_events("lab_code", ["HR"], "data/measurement")},
            raw_files={
                "source_a.csv": "lab_code,val_a\nHR,value_1\n",
                "source_b.csv": "lab_code,val_b\nHR,value_2\n",
            },
        )

    assert codes_df.schema["custom_prop"] == pl.List(pl.String)
    hr_row = codes_df.filter(pl.col("code") == "HR")
    assert hr_row["custom_prop"][0].to_list() == ["value_1", "value_2"]


def test_extract_code_metadata_code_template_is_a_single_string():
    """``code_template`` is a plain String: one code has exactly one template.

    Multiple sources contributing metadata for the same code share that code's one template; after
    deduplication exactly one survives, emitted as a scalar (distinct templates colliding on one code are a
    config error, tested separately).
    """
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      source_a:
        lab_code: $lab_code
        description: $title_a
      source_b:
        lab_code: $lab_code
        description: $title_b
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": _bare_code_events("lab_code", ["HR"], "data/measurement")},
            raw_files={
                "source_a.csv": "lab_code,title_a\nHR,Heart Rate\n",
                "source_b.csv": "lab_code,title_b\nHR,Pulse Rate\n",
            },
            description_separator="; ",
        )

    assert codes_df.schema["code_template"] == pl.String
    hr_row = codes_df.filter(pl.col("code") == "HR")
    assert hr_row["code_template"][0] == "$lab_code"


def test_extract_code_metadata_handles_code_named_source_column():
    """Codes built from a source column literally named ``code`` flow through the full stage.

    The idiomatic ICD/OMOP vocabulary-table shape — ``code: f"ICD//{$code}"`` — must flow
    through the ``code_components`` map build, metadata extraction, and reduction without
    colliding with the output ``code`` column (regression for
    https://github.com/mmcdermott/MEDS_extract/issues/110). Before the fix this raised
    ``DuplicateError`` at the ``code_components`` unnest (this test carried a strict xfail
    marker). The events parquet carries a ``code_components`` struct with a field named
    ``code`` — exactly the shape ``convert_to_MEDS_events`` emits for that expression
    (proven by the struct-shape doctest on ``EventConfig.extract``).
    """
    messy = """\
diagnoses:
  dx:
    code: 'f"ICD//{$code}"'
    _metadata:
      icd_descriptions:
        code: $code
        description: $long_title
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "diagnoses": pl.DataFrame(
                    {
                        "code": ["ICD//250.00", "ICD//401.9"],
                        "code_components": [{"code": "250.00"}, {"code": "401.9"}],
                        "source_block": ["diagnoses/dx", "diagnoses/dx"],
                    }
                )
            },
            raw_files={
                "icd_descriptions.csv": (
                    "code,long_title\n250.00,Diabetes mellitus\n401.9,Essential hypertension\n"
                )
            },
            # A non-conflicting pre-existing column name: how overlapping metadata columns
            # merge is orthogonal to the code-column collision (overlap coalescing is covered above).
            existing_codes=pl.DataFrame({"code": ["EXISTING"], "old_description": ["pre-existing code"]}),
        )

    by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
    assert by_code.get("ICD//250.00") == "Diabetes mellitus"
    assert by_code.get("ICD//401.9") == "Essential hypertension"
    # The pre-existing row survives the merge with its non-conflicting column intact.
    existing = codes_df.filter(pl.col("code") == "EXISTING")
    assert existing["old_description"].to_list() == ["pre-existing code"]


# ── The observed-code universe: every observed code appears in the output ──


def test_observed_code_with_no_metadata_match_appears_with_null_metadata():
    """A code observed in the data but matching no metadata row still appears in codes.parquet.

    Recent MEDS spec versions require ``metadata/codes.parquet`` to enumerate EVERY code
    observed in the data for the dataset to be valid. ``TEMP`` is observed in the events
    but has no row in the metadata source, so it must appear with all-null metadata
    columns rather than being dropped by the metadata join.
    """
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      lab_meta:
        lab_code: $lab_code
        description: $title
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": _bare_code_events("lab_code", ["HR", "TEMP"], "data/measurement")},
            # Metadata describes HR only — TEMP is observed but unmatched.
            raw_files={"lab_meta.csv": "lab_code,title\nHR,Heart Rate\n"},
        )

    assert sorted(codes_df["code"].to_list()) == ["HR", "TEMP"], (
        f"codes.parquet must enumerate every observed code.\n{codes_df}"
    )
    temp_row = codes_df.filter(pl.col("code") == "TEMP").to_dicts()[0]
    assert temp_row["description"] is None
    assert temp_row["code_template"] is None
    hr_row = codes_df.filter(pl.col("code") == "HR").to_dicts()[0]
    assert hr_row["description"] == "Heart Rate"


def test_observed_vocabulary_exact_across_mixed_component_shapes():
    """codes.parquet's vocabulary is exactly the distinct non-null codes across all event frames.

    The reducer seeds the output with the observed code universe, and no event shape may
    fall out of it: component-bearing codes with a metadata match, component-bearing
    codes with no match, codes from a literal-code event file (no ``code_components``
    column at all — null-struct rows in the mixed-schema concat), and rows whose
    components are null while the code is not. Null codes are excluded, as always.
    """
    messy = """\
labs:
  measurement:
    code: 'f"LAB//{$lab_code}"'
    _metadata:
      lab_meta:
        lab_code: $lab_code
        description: $title
admissions:
  admit:
    code: ADMISSION
    time: null
"""
    event_frames = {
        "labs": pl.DataFrame(
            {
                # A metadata match (HR), a no-match (TEMP), a null-component row whose
                # code survives (UNK), and a null code that must NOT appear.
                "code": ["LAB//HR", "LAB//TEMP", "LAB//UNK", None],
                "code_components": [
                    {"lab_code": "HR"},
                    {"lab_code": "TEMP"},
                    {"lab_code": None},
                    {"lab_code": "X"},
                ],
                "source_block": ["labs/measurement"] * 4,
            }
        ),
        # A literal-code event file: no code_components column at all.
        "admissions": pl.DataFrame({"code": ["ADMISSION"], "source_block": ["admissions/admit"]}),
    }
    expected = sorted(
        {c for frame in event_frames.values() for c in frame["code"].to_list() if c is not None}
    )
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames=event_frames,
            raw_files={"lab_meta.csv": "lab_code,title\nHR,Heart Rate\n"},
        )

    assert codes_df["code"].to_list() == expected, (
        f"codes.parquet vocabulary must be exactly the distinct non-null observed codes "
        f"({expected}).\n{codes_df}"
    )
    by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
    assert by_code["LAB//HR"] == "Heart Rate"
    assert by_code["LAB//TEMP"] is None
    assert by_code["LAB//UNK"] is None
    assert by_code["ADMISSION"] is None


def test_no_metadata_blocks_still_writes_all_observed_codes():
    """A config with NO ``_metadata`` blocks still yields a codes.parquet of every observed code.

    MEDS validity requires the full observed code vocabulary in ``metadata/codes.parquet``
    even when no metadata sources are configured, so the stage may not exit without
    writing an output.
    """
    messy = """\
data:
  measurement:
    code: $lab_code
    time: null
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": _bare_code_events("lab_code", ["HR", "TEMP", "GLU"], "data/measurement")},
            raw_files={},
        )

    assert codes_df is not None, "No codes.parquet was written with no _metadata blocks configured."
    assert codes_df["code"].to_list() == ["GLU", "HR", "TEMP"]  # every observed code, sorted
    assert codes_df.columns == ["code"]


# ── Early returns and empty outputs ──


def test_preexisting_codes_with_no_metadata_blocks_still_merges(caplog):
    """No ``_metadata`` blocks + a pre-existing ``codes.parquet``: both survive in the output.

    The stage no longer returns early when the config carries no ``_metadata`` blocks —
    MEDS validity requires every observed code in the output — so the reducer still runs
    and the pre-existing metadata is merged onto the observed code vocabulary: the
    observed code appears (with null metadata), and the pre-existing row survives the
    full join intact.
    """
    messy = """\
data:
  measurement:
    code: $lab_code
    time: null
"""
    existing = pl.DataFrame({"code": ["EXISTING_CODE"], "description": ["An existing code"]})
    with tempfile.TemporaryDirectory() as d, caplog.at_level("INFO"):
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": _bare_code_events("lab_code", ["HR"], "data/measurement")},
            raw_files={},
            existing_codes=existing,
        )
        assert "No _metadata blocks" in caplog.text
        # The pre-existing input file is untouched where it was.
        input_codes = pl.read_parquet(Path(d) / "metadata_in" / "metadata" / "codes.parquet")
        assert input_codes.equals(existing)

    by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
    assert by_code == {"EXISTING_CODE": "An existing code", "HR": None}


def test_metadata_blocks_over_componentless_events_error():
    """Configured ``_metadata`` blocks over events with no ``code_components`` anywhere error.

    A ``_metadata`` block always attaches to a component-bearing code, so an input with no
    components in ANY event file cannot be a ``convert_to_MEDS_events`` output — in
    practice it is merged data (``merge_to_MEDS_cohort`` drops ``code_components``),
    i.e. ``extract_code_metadata`` ordered after the merge stage. The old behavior wrote a
    codes-only table with a buried warning: every metadata column silently lost while the
    per-source partials stayed correct, the row count matched the observed vocabulary, and
    the run exited 0. That degrade must be a loud error naming the likely cause.
    """
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      lab_meta:
        lab_code: $lab_code
        description: $title
"""
    with (
        tempfile.TemporaryDirectory() as d,
        pytest.raises(ValueError, match="most likely merged data"),
    ):
        _run_ecm_scenario(
            Path(d),
            messy,
            # Post-merge shape: source_block survives, code_components is dropped.
            event_frames={"data": pl.DataFrame({"code": ["HR"], "source_block": ["data/measurement"]})},
            raw_files={"lab_meta.csv": "lab_code,title\nHR,Heart Rate\n"},
        )


def test_extract_code_metadata_no_matching_codes(caplog):
    """Metadata whose keys match no observed event code attaches nothing, but codes survive.

    The zero-match condition is surfaced (the ``matched zero codes`` WARNING naming the
    declaring source block), and the reduced output still enumerates the observed codes —
    with null metadata, since nothing matched.
    """
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      lab_meta:
        lab_code: $lab_code
        description: $title
"""
    with tempfile.TemporaryDirectory() as d, caplog.at_level("WARNING"):
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": _bare_code_events("lab_code", ["HR"], "data/measurement")},
            raw_files={"lab_meta.csv": "lab_code,title\nNONEXISTENT,No Match\n"},
        )
    assert "matched zero codes" in caplog.text
    assert "'data/measurement'" in caplog.text  # the warning names the declaring source block
    # The observed code still appears, with null metadata (nothing matched it).
    assert codes_df["code"].to_list() == ["HR"]
    assert codes_df["description"].to_list() == [None]


# ── Mixed-schema and multi-config mapper/reducer regressions ──


def test_mixed_schema_parquet_scan_with_and_without_code_components():
    """Regression guard: metadata extraction must handle heterogeneous event parquet schemas.

    Some event files have a code_components column (dynamic codes like f"{$test_name}//{$units}")
    and others don't (literal codes like "ADMISSION"). The reducer must scan these mixed-schema
    files without crashing. Previously a single glob scan_parquet raised on schema mismatches.
    """
    messy = """\
labs:
  measurement:
    code: 'f"{$test_name}//{$units}"'
    _metadata:
      lab_meta:
        test_name: $test_name
        units: $units
        description: $title
admissions:
  admit:
    code: ADMISSION
    time: null
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                # Event file WITH code_components (dynamic code)...
                "labs": pl.DataFrame(
                    {
                        "code": ["Glucose//mg/dL", "BUN//mg/dL"],
                        "code_components": [
                            {"test_name": "Glucose", "units": "mg/dL"},
                            {"test_name": "BUN", "units": "mg/dL"},
                        ],
                        "source_block": ["labs/measurement", "labs/measurement"],
                    }
                ),
                # ...and one WITHOUT (literal code).
                "admissions": pl.DataFrame({"code": ["ADMISSION"], "source_block": ["admissions/admit"]}),
            },
            raw_files={"lab_meta.csv": "test_name,units,title\nGlucose,mg/dL,Blood Glucose\n"},
        )

    assert "Glucose//mg/dL" in codes_df["code"].to_list()
    assert codes_df.filter(pl.col("code") == "Glucose//mg/dL")["description"][0] == "Blood Glucose"


def test_component_named_produced_column_is_a_join_key():
    """A produced column whose name matches a code component IS a join key — by design.

    Under the dftly ``_metadata`` contract, key/output classification is purely
    name-based: producing ``item`` on an event whose code references ``$item`` makes
    ``item`` a join key sourced from the declared expression (rename-based key
    sourcing), never a metadata output. Here ``item: $item_description`` sources the
    ``item`` key from description text that matches no real item value, so the join
    (on category AND item) matches zero codes and warns — the metadata does NOT
    broadcast on ``category`` alone. To broadcast per category, don't produce ``item``
    (see ``test_mixed_full_and_partial_match_from_same_metadata_prefix``); to attach
    the text as metadata, name the output something that is not a component.
    """
    messy = """\
data:
  event:
    code: 'f"{$category}//{$item}"'
    _metadata:
      category_meta:
        category: $category
        item: $item_description
        description: $item_description
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "data": pl.DataFrame(
                    {
                        "code": ["Drug//Aspirin", "Drug//Ibuprofen"],
                        "code_components": [
                            {"category": "Drug", "item": "Aspirin"},
                            {"category": "Drug", "item": "Ibuprofen"},
                        ],
                        "source_block": ["data/event", "data/event"],
                    }
                )
            },
            raw_files={"category_meta.csv": "category,item_description\nDrug,Pharmaceutical compound\n"},
        )

    # "item" is a join key, not an output — it never appears as an output column, and
    # because its produced values are description text, nothing matched.
    assert "item" not in codes_df.columns, (
        f"'item' names a code component, so it must be consumed as a join key, "
        f"not emitted as metadata. Columns: {codes_df.columns}.\nFull output:\n{codes_df}"
    )
    assert codes_df.filter(pl.col("description").is_not_null()).height == 0, (
        f"The (category, item) join should match zero codes — 'item' was sourced from "
        f"description text.\nFull output:\n{codes_df}"
    )


def test_metadata_key_renamed_from_differently_named_metadata_column():
    """A join key sourced from a differently-named metadata column is just an expression.

    The event's code references ``$itemid``, but the metadata table keys its rows under
    ``omop_source_code``. Producing ``itemid: $omop_source_code`` sources the join key
    from that column — the rename capability that ``_match_on`` shadowing accidentally
    provided (and #145 removed) is now first-class dftly.
    """
    messy = """\
chartevents:
  chart:
    code: 'f"CHART//{$itemid}"'
    _metadata:
      d_items:
        itemid: $omop_source_code
        description: $label
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "chartevents": pl.DataFrame(
                    {
                        "code": ["CHART//220045", "CHART//220179"],
                        "code_components": [{"itemid": "220045"}, {"itemid": "220179"}],
                        "source_block": ["chartevents/chart", "chartevents/chart"],
                    }
                )
            },
            # No column named "itemid" anywhere in the metadata table.
            raw_files={"d_items.csv": "omop_source_code,label\n220045,Heart Rate\n220179,NBP systolic\n"},
        )

    by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
    assert by_code.get("CHART//220045") == "Heart Rate"
    assert by_code.get("CHART//220179") == "NBP systolic"


def test_metadata_key_normalized_with_dftly_coalesce():
    """Dftly normalization on a join key shapes the values the component join runs against.

    The metadata table splits its unit key across two columns (``unit_primary`` with
    gaps, ``unit_alt`` as fallback); the block coalesces them into the ``valueuom``
    key with ``??``. Only through that user-visible normalization does the row with a
    null ``unit_primary`` match its code.
    """
    messy = """\
labevents:
  lab:
    code: 'f"LAB//{$itemid}//{$valueuom}"'
    _metadata:
      d_labitems:
        itemid: $itemid
        valueuom: $unit_primary ?? $unit_alt
        description: $label
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "labevents": pl.DataFrame(
                    {
                        "code": ["LAB//50912//mg/dL", "LAB//51463//mEq/L"],
                        "code_components": [
                            {"itemid": "50912", "valueuom": "mg/dL"},
                            {"itemid": "51463", "valueuom": "mEq/L"},
                        ],
                        "source_block": ["labevents/lab", "labevents/lab"],
                    }
                )
            },
            # 50912 carries its unit in unit_primary; 51463 only in the fallback column.
            raw_files={
                "d_labitems.csv": (
                    "itemid,unit_primary,unit_alt,label\n"
                    "50912,mg/dL,IGNORED,Creatinine\n"
                    "51463,,mEq/L,Chloride\n"
                )
            },
        )

    by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
    assert by_code.get("LAB//50912//mg/dL") == "Creatinine"
    assert by_code.get("LAB//51463//mEq/L") == "Chloride", (
        f"The '?? $unit_alt' coalesce on the valueuom key should rescue this row.\n{codes_df}"
    )


def test_bare_string_metadata_value_is_a_dftly_string_literal():
    """A bare, unquoted ``_metadata`` value is a dftly string LITERAL — not a column reference.

    ``_metadata`` expressions carry exactly dftly's semantics (the pre-0.7 shorthand
    where ``description: title`` read the raw ``title`` column is gone; write
    ``description: $title``). This pins the accepted breaking behavior end-to-end: the
    bare word ``title`` stamps the constant text ``"title"`` on every matched code —
    the ``title`` column's values never appear.
    """
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      lab_meta:
        lab_code: $lab_code
        description: title
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": _bare_code_events("lab_code", ["HR", "TEMP"], "data/measurement")},
            raw_files={"lab_meta.csv": "lab_code,title\nHR,Heart Rate\nTEMP,Body Temperature\n"},
        )

    descriptions = codes_df["description"].drop_nulls().unique().to_list()
    assert descriptions == ["title"], (
        f"Bare 'title' must be the literal string 'title' for every matched code, "
        f"never the title column's values.\n{codes_df}"
    )
    assert "Heart Rate" not in codes_df["description"].to_list()


def test_parent_codes_chained_ternary_end_to_end():
    """``parent_codes`` as a chained dftly conditional — the ICD9/ICD10 vocabulary shape.

    One expression yields at most one parent per metadata row; the omitted final
    ``else`` yields a real null for versions matching no case. The reducer aggregates
    the per-row scalar Strings into the canonical per-code ``List(String)``, and a
    code whose only metadata row yields a null parent ends with a null (not ``[]``)
    ``parent_codes``.
    """
    messy = """\
diagnoses:
  dx:
    code: 'f"ICD{$icd_version}//{$icd_code}"'
    _metadata:
      d_icd:
        icd_code: $icd_code
        icd_version: $icd_version
        description: $long_title
        parent_codes: >-
          f"ICD{$icd_version}CM/{$icd_code}" if $icd_version == "9"
          else f"ICD{$icd_version}CM/{$icd_code}" if $icd_version == "10"
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "diagnoses": pl.DataFrame(
                    {
                        "code": ["ICD9//25000", "ICD10//E119", "ICD11//XX99"],
                        "code_components": [
                            {"icd_code": "25000", "icd_version": "9"},
                            {"icd_code": "E119", "icd_version": "10"},
                            {"icd_code": "XX99", "icd_version": "11"},
                        ],
                        "source_block": ["diagnoses/dx"] * 3,
                    }
                )
            },
            raw_files={
                "d_icd.csv": (
                    "icd_code,icd_version,long_title\n"
                    "25000,9,Diabetes mellitus\n"
                    "E119,10,Type 2 diabetes\n"
                    "XX99,11,Some ICD-11 dx\n"
                )
            },
        )

    assert codes_df.schema["parent_codes"] == pl.List(pl.String)
    by_code = {r["code"]: r["parent_codes"] for r in codes_df.iter_rows(named=True)}
    assert by_code["ICD9//25000"] == ["ICD9CM/25000"]
    assert by_code["ICD10//E119"] == ["ICD10CM/E119"]
    # No conditional case matched version 11: the row's parent is null, so the code's
    # aggregated parent_codes is null — never an empty list.
    assert by_code["ICD11//XX99"] is None, (
        f"A no-match parent_codes row must aggregate to null, got {by_code['ICD11//XX99']!r}.\n{codes_df}"
    )
    # The metadata itself (description) still attached to all three codes.
    assert codes_df.filter(pl.col("description").is_not_null()).height == 3


def test_list_typed_parent_codes_from_parquet_source():
    """A parquet metadata source storing ``parent_codes`` as ``List(String)`` — the codes.parquet shape.

    The mapper mandates a scalar String ``parent_codes`` per metadata row, so a list-typed
    source column is exploded into one row per element (sibling columns replicating
    alongside) before the reducer unions the scalars back into the canonical per-code
    ``List(String)``. An element delivered by two source rows appears once — the union
    dedups — and empty or null source lists behave like null scalars: the code's
    aggregated ``parent_codes`` is null, never ``[]``.
    """
    messy = """\
data:
  lab:
    code: 'f"LAB//{$code}"'
    _metadata:
      codes:
        code: $code
        description: $description
        parent_codes: $parent_codes
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "data": pl.DataFrame(
                    {
                        "code": ["LAB//A", "LAB//B", "LAB//C"],
                        "code_components": [{"code": "A"}, {"code": "B"}, {"code": "C"}],
                        "source_block": ["data/lab"] * 3,
                    }
                )
            },
            raw_files={
                "codes.parquet": pl.DataFrame(
                    {
                        "code": ["A", "A", "B", "C"],
                        "description": ["Alpha", "Alpha", "Beta", "Gamma"],
                        "parent_codes": [["ICD9CM/1", "ICD9CM/2"], ["ICD9CM/2", "ICD9CM/3"], [], None],
                    },
                    schema={
                        "code": pl.String,
                        "description": pl.String,
                        "parent_codes": pl.List(pl.String),
                    },
                )
            },
        )

    assert codes_df.schema["parent_codes"] == pl.List(pl.String)
    by_code = {r["code"]: r for r in codes_df.iter_rows(named=True)}
    # Union of both source rows' lists, with the shared "ICD9CM/2" element appearing once.
    assert sorted(by_code["LAB//A"]["parent_codes"]) == ["ICD9CM/1", "ICD9CM/2", "ICD9CM/3"]
    # An empty source list and a null source list both aggregate to null.
    assert by_code["LAB//B"]["parent_codes"] is None
    assert by_code["LAB//C"]["parent_codes"] is None
    # Sibling columns replicated through the explode: descriptions attach once per code.
    assert by_code["LAB//A"]["description"] == "Alpha"
    assert by_code["LAB//B"]["description"] == "Beta"
    assert by_code["LAB//C"]["description"] == "Gamma"


def test_reduced_column_set_for_aggregated_union_parent_codes_block():
    """The reduced ``codes.parquet`` COLUMN SET survives an aggregated-union metadata block.

    The eICU shape: one ``_metadata`` block whose only output is ``parent_codes``, with
    several metadata rows sharing a join key so the reducer unions them into one
    ``List(String)`` per code — alongside a literal-code event file with no
    ``code_components`` column. The assertion is on the exact column set, not row count:
    losing every metadata column preserves the row count (the output always enumerates
    the observed vocabulary), so a cardinality check alone stays green through total
    metadata loss.
    """
    messy = """\
diagnosis:
  dx:
    code: f"DIAGNOSIS//{$diagnosisstring}"
    _metadata:
      dx_icd_map:
        diagnosisstring: $diagnosisstring
        parent_codes: >-
          f"ICD9CM/{$icd_code}" if $icd_version == "9"
          else f"ICD10CM/{$icd_code}" if $icd_version == "10"
vitalPeriodic:
  hr:
    code: '"VITALS//PERIODIC//HEARTRATE"'
    time: null
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "diagnosis": pl.DataFrame(
                    {
                        "code": ["DIAGNOSIS//a|b", "DIAGNOSIS//c|d"],
                        "code_components": [{"diagnosisstring": "a|b"}, {"diagnosisstring": "c|d"}],
                        "source_block": ["diagnosis/dx"] * 2,
                    }
                ),
                # Literal-code table: no code_components column at all.
                "vitalPeriodic": pl.DataFrame(
                    {"code": ["VITALS//PERIODIC//HEARTRATE"], "source_block": ["vitalPeriodic/hr"]}
                ),
            },
            # Two metadata rows share the "a|b" join key: the reducer must union both
            # parents into one list for that code.
            raw_files={
                "dx_icd_map.csv": (
                    "diagnosisstring,icd_code,icd_version\na|b,250.00,9\na|b,E11.9,10\nc|d,401.9,9\n"
                )
            },
        )

    assert set(codes_df.columns) == {"code", "code_template", "parent_codes"}, (
        f"The aggregated-union metadata block's columns must survive reduction.\n{codes_df}"
    )
    assert codes_df.schema["parent_codes"] == pl.List(pl.String)
    by_code = {r["code"]: r["parent_codes"] for r in codes_df.iter_rows(named=True)}
    assert sorted(by_code["DIAGNOSIS//a|b"]) == ["ICD10CM/E11.9", "ICD9CM/250.00"]
    assert by_code["DIAGNOSIS//c|d"] == ["ICD9CM/401.9"]
    # The literal code appears in the vocabulary with null metadata.
    assert by_code["VITALS//PERIODIC//HEARTRATE"] is None


def test_reduced_output_invariant_to_event_file_discovery_order(monkeypatch):
    """Event-file discovery order cannot change the reduced output.

    Per-table event files have heterogeneous schemas (a literal-code table carries no
    ``code_components`` column), and ``Path.rglob`` enumerates in filesystem order — which
    varies across machines and directory histories. Two runs over identical mixed-schema
    inputs are forced through opposite enumeration orders and must produce identical,
    correct output frames.
    """
    from polars.testing import assert_frame_equal

    messy = """\
labs:
  measurement:
    code: 'f"{$test_name}//{$units}"'
    _metadata:
      lab_meta:
        test_name: $test_name
        units: $units
        description: $title
admissions:
  admit:
    code: ADMISSION
    time: null
"""
    event_frames = {
        "labs": pl.DataFrame(
            {
                "code": ["Glucose//mg/dL", "BUN//mg/dL"],
                "code_components": [
                    {"test_name": "Glucose", "units": "mg/dL"},
                    {"test_name": "BUN", "units": "mg/dL"},
                ],
                "source_block": ["labs/measurement"] * 2,
            }
        ),
        # Literal-code table (no code_components) — under the wrong enumeration handling
        # this file coming first is what degraded the run to a codes-only output.
        "admissions": pl.DataFrame({"code": ["ADMISSION"], "source_block": ["admissions/admit"]}),
    }
    raw_files = {"lab_meta.csv": "test_name,units,title\nGlucose,mg/dL,Blood Glucose\n"}

    real_rglob = Path.rglob
    frames: list[pl.DataFrame] = []
    for reorder in (lambda fps: fps, lambda fps: list(reversed(fps))):

        def forced_order_rglob(self, pattern, _reorder=reorder):
            return iter(_reorder(sorted(real_rglob(self, pattern))))

        monkeypatch.setattr(Path, "rglob", forced_order_rglob)
        with tempfile.TemporaryDirectory() as d:
            frames.append(_run_ecm_scenario(Path(d), messy, event_frames, raw_files))
    monkeypatch.undo()

    assert_frame_equal(frames[0], frames[1], check_row_order=True, check_column_order=True)
    assert set(frames[0].columns) == {"code", "code_template", "description"}
    by_code = {r["code"]: r["description"] for r in frames[0].iter_rows(named=True)}
    assert by_code["Glucose//mg/dL"] == "Blood Glucose"
    assert by_code["ADMISSION"] is None


def test_mixed_full_and_partial_match_from_same_metadata_prefix():
    """Regression guard: configs with different match-column sets sharing a metadata prefix.

    A single metadata file prefix can be referenced by multiple event configs whose joins
    use different match columns and are scoped to different declaring events. Each must be
    written to a separate intermediate shard so the reducer can expand them independently
    (historically, all configs for one prefix were concatenated into one shard and rows
    from all but one config were silently dropped).

    This test uses "shared_meta" referenced by a full-match config (code: $lab_code,
    producing its only component) and a partial-match config
    (code: f"{$category}//{$item}", producing only ``category``).
    """
    messy = """\
labs:
  measurement:
    code: $lab_code
    _metadata:
      shared_meta:
        lab_code: $lab_code
        description: $desc
products:
  product:
    code: 'f"{$category}//{$item}"'
    _metadata:
      shared_meta:
        category: $category
        description: $desc
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                # Lab events: full match on the single referenced column.
                "labs": _bare_code_events("lab_code", ["HR"], "labs/measurement"),
                # Product events: partial match narrows a composite code to one component.
                "products": pl.DataFrame(
                    {
                        "code": ["Drug//Aspirin", "Drug//Ibuprofen"],
                        "code_components": [
                            {"category": "Drug", "item": "Aspirin"},
                            {"category": "Drug", "item": "Ibuprofen"},
                        ],
                        "source_block": ["products/product", "products/product"],
                    }
                ),
            },
            # Shared metadata file: "HR" matches on lab_code (all components); "Drug"
            # matches on category (partial-match narrowing).
            raw_files={"shared_meta.csv": "lab_code,category,desc\nHR,Drug,Shared description\n"},
        )

    codes_with_desc = codes_df.filter(pl.col("description").is_not_null())
    matched_codes = set(codes_with_desc["code"].to_list())

    # Full match: HR should get description from shared_meta via lab_code.
    assert "HR" in matched_codes, f"Full-match code 'HR' missing from output.\n{codes_df}"
    # Partial match: both Drug codes should get description via category=Drug.
    assert "Drug//Aspirin" in matched_codes, (
        f"Partial-match code 'Drug//Aspirin' missing from output.\n{codes_df}"
    )
    assert "Drug//Ibuprofen" in matched_codes, (
        f"Partial-match code 'Drug//Ibuprofen' missing from output.\n{codes_df}"
    )


# ── Component-join correctness regressions ──


def test_partial_match_on_column_named_code_is_expanded_not_passed_through():
    """Regression guard: a metadata shard keyed on a column named ``code`` is still expanded.

    Producing a key named ``code`` (allowed exactly because it names a component — the
    ICD/OMOP vocabulary shape) makes the intermediate shard carry a ``code`` column of raw
    component values (e.g. ``250.00``). The reducer historically classified shards by
    sniffing for a ``code`` column in the schema and passed such raw component values
    straight through as output codes — silently wrong codes.parquet. The join must be
    driven by explicit map-time bookkeeping instead.
    """
    messy = """\
diagnoses:
  dx:
    code: 'f"ICD//{$code}"'
    _metadata:
      icd_meta:
        code: $code
        description: $long_title
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "diagnoses": pl.DataFrame(
                    {
                        "code": ["ICD//250.00", "ICD//401.9"],
                        "code_components": [{"code": "250.00"}, {"code": "401.9"}],
                        "source_block": ["diagnoses/dx", "diagnoses/dx"],
                    }
                )
            },
            raw_files={"icd_meta.csv": "code,long_title\n250.00,Diabetes mellitus\n401.9,Hypertension\n"},
        )

        by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
        # Component-keyed expansion: metadata lands on the FULL codes...
        assert by_code.get("ICD//250.00") == "Diabetes mellitus"
        assert by_code.get("ICD//401.9") == "Hypertension"
        # ...and the raw component values are NOT passed through as codes (the old
        # schema-sniffing misclassification symptom).
        assert "250.00" not in by_code
        assert "401.9" not in by_code


def test_partial_match_scoped_to_declaring_event():
    """Regression guard: partial-match expansion is scoped to the declaring event.

    Two events build codes from a same-named ``itemid`` component with colliding values, but
    only the chartevents event declares the ``d_items`` metadata. The labevents code must NOT
    receive that metadata, and no output row may carry the declaring config's code_template
    as false provenance for a labevents code.
    """
    messy = """\
chartevents:
  chart:
    code: 'f"CHART//{$itemid}"'
    _metadata:
      d_items:
        itemid: $itemid
        description: $label
labevents:
  lab:
    code: 'f"LAB//{$itemid}"'
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "chartevents": pl.DataFrame(
                    {
                        "code": ["CHART//1"],
                        "code_components": [{"itemid": "1"}],
                        "source_block": ["chartevents/chart"],
                    }
                ),
                "labevents": pl.DataFrame(
                    {
                        "code": ["LAB//1"],
                        "code_components": [{"itemid": "1"}],
                        "source_block": ["labevents/lab"],
                    }
                ),
            },
            raw_files={"d_items.csv": "itemid,label\n1,Heart Rate (chart)\n"},
        )

        chart_rows = codes_df.filter(pl.col("code") == "CHART//1")
        assert chart_rows["description"].to_list() == ["Heart Rate (chart)"]
        assert chart_rows["code_template"].to_list() == ['f"CHART//{$itemid}"']

        # The labevents code must not receive the chartevents-declared metadata.
        lab_rows = codes_df.filter(pl.col("code") == "LAB//1")
        assert lab_rows["description"].drop_nulls().to_list() == [], (
            f"LAB//1 must not receive metadata declared on the chartevents event.\n{codes_df}"
        )
        assert lab_rows["code_template"].drop_nulls().to_list() == [], (
            f"LAB//1 must not be stamped with the chartevents code_template.\n{codes_df}"
        )


def test_partial_match_typed_int_components_join_csv_metadata():
    """Regression guard: typed ``Int64`` components join against all-String CSV keys.

    CSV metadata sources are read with ``infer_schema=False`` (all-String), while code
    components keep their raw source dtypes. The reducer join used to crash with
    ``SchemaError: datatypes of join keys don't match``; join keys must be normalized to
    String on both sides.
    """
    messy = """\
chartevents:
  chart:
    code: 'f"CHART//{$itemid}"'
    _metadata:
      d_items:
        itemid: $itemid
        description: $label
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "chartevents": pl.DataFrame(
                    {
                        "code": ["CHART//220045", "CHART//220179"],
                        "code_components": [{"itemid": 220045}, {"itemid": 220179}],
                        "source_block": ["chartevents/chart", "chartevents/chart"],
                    }
                )
            },
            raw_files={"d_items.csv": "itemid,label\n220045,Heart Rate\n220179,NBP systolic\n"},
        )

        by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
        assert by_code.get("CHART//220045") == "Heart Rate"
        assert by_code.get("CHART//220179") == "NBP systolic"


def test_partial_match_integer_valued_float_components_join_csv_metadata():
    """Regression guard: integer-valued float components render as ``220045``.

    A ``Float64`` component with value ``220045.0`` must match the metadata string
    ``"220045"`` — a plain String cast would render ``"220045.0"`` and silently zero-match.
    Non-integer float values keep their float rendering.
    """
    messy = """\
chartevents:
  chart:
    code: 'f"CHART//{$itemid}"'
    _metadata:
      d_items:
        itemid: $itemid
        description: $label
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "chartevents": pl.DataFrame(
                    {
                        "code": ["CHART//220045", "CHART//1.5"],
                        "code_components": [{"itemid": 220045.0}, {"itemid": 1.5}],
                        "source_block": ["chartevents/chart", "chartevents/chart"],
                    }
                )
            },
            raw_files={"d_items.csv": "itemid,label\n220045,Heart Rate\n1.5,Half Item\n"},
        )

        by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
        assert by_code.get("CHART//220045") == "Heart Rate"
        assert by_code.get("CHART//1.5") == "Half Item"


def test_null_component_matches_null_metadata_key():
    """A null metadata join key matches the code a coalesced null component produced.

    The labevents code coalesces a null ``valueuom`` to ``UNK``, so the data row with no
    unit emits ``LAB//51463//UNK`` while its ``valueuom`` component stays null. The
    vocabulary row for itemid 51463 also has no unit (a null key). The component join
    treats null keys as equal to null components, so the label lands on the ``UNK`` code;
    the real-unit row keeps matching its real-unit metadata.
    """
    messy = """\
labevents:
  lab:
    code: |-
      f"LAB//{$itemid}//{$valueuom ?? 'UNK'}"
    _metadata:
      d_labitems:
        itemid: $itemid
        valueuom: $valueuom
        description: $label
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "labevents": pl.DataFrame(
                    {
                        "code": ["LAB//51463//UNK", "LAB//50912//mg/dL"],
                        "code_components": [
                            {"itemid": "51463", "valueuom": None},
                            {"itemid": "50912", "valueuom": "mg/dL"},
                        ],
                        "source_block": ["labevents/lab", "labevents/lab"],
                    }
                )
            },
            # The 51463 row's empty valueuom field reads as null under infer_schema=False.
            raw_files={"d_labitems.csv": "itemid,valueuom,label\n51463,,Chloride\n50912,mg/dL,Creatinine\n"},
        )

    by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
    assert by_code.get("LAB//51463//UNK") == "Chloride"
    assert by_code.get("LAB//50912//mg/dL") == "Creatinine"


def test_transformed_code_matches_raw_component_values():
    """A code transforming a component still matches metadata holding the RAW value.

    The code truncates ``icd_code`` to its first three characters, so the emitted code is
    ``DX//250`` while the component retains the raw ``250.00``. Matching happens in raw
    component space, so a vocabulary table keyed on the untransformed ``250.00`` links —
    the metadata table is never re-rendered through the code expression, and the transform
    cannot be applied twice.
    """
    messy = """\
diagnoses:
  dx:
    code: 'f"DX//{$icd_code[0:3]}"'
    _metadata:
      icd_meta:
        icd_code: $icd_code
        description: $long_title
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "diagnoses": pl.DataFrame(
                    {
                        "code": ["DX//250"],
                        "code_components": [{"icd_code": "250.00"}],
                        "source_block": ["diagnoses/dx"],
                    }
                )
            },
            raw_files={"icd_meta.csv": "icd_code,long_title\n250.00,Diabetes mellitus\n"},
        )

    by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
    assert by_code.get("DX//250") == "Diabetes mellitus"


def test_literal_code_with_metadata_block_errors():
    """A ``_metadata`` block on a literal code raises a clear config error.

    A literal code references no source columns, so there are no components to match metadata on; the stage
    rejects the config at map time rather than degenerately matching.
    """
    messy = """\
admissions:
  admit:
    code: ADMISSION
    time: null
    _metadata:
      adm_meta:
        description: title
"""
    with (
        tempfile.TemporaryDirectory() as d,
        pytest.raises(ValueError, match="literal code has no components to match metadata on"),
    ):
        _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "admissions": pl.DataFrame({"code": ["ADMISSION"], "source_block": ["admissions/admit"]})
            },
            raw_files={"adm_meta.csv": "x,title\n1,t\n"},
        )


@pytest.mark.parametrize(
    ("metadata_block", "expected"),
    [
        ({"code": "label"}, r"\['code'\]"),
        ({"code_template": "label"}, r"\['code_template'\]"),
        ({"code": "label", "code_template": "label", "description": "label"}, r"\['code', 'code_template'\]"),
    ],
    ids=["code", "code_template", "both"],
)
def test_metadata_reserved_output_column_names_error(metadata_block, expected):
    """A ``_metadata`` block may not redefine the pipeline-generated ``code``/``code_template`` columns.

    Both are stamped by the pipeline with mandated meanings; a config trying to emit its own is rejected at
    compile time with the offending name(s) listed (the multi-name listing is what the parametrization adds
    over the single-name doctests on ``compile_metadata_block``).
    """
    from MEDS_extract.config import compile_metadata_block

    with pytest.raises(ValueError, match=f"{expected} are reserved"):
        compile_metadata_block(metadata_block, {"icd"}, code_template_str='f"ICD//{$icd}"')


def test_metadata_requiring_absent_component_columns_errors():
    """A ``_metadata`` block joining on component columns no event file carries errors.

    The code expression references ``$a`` and ``$b`` (match columns ``[a, b]``), but the
    event data's ``code_components`` struct only carries ``a`` — the events were produced
    by an older configuration. Such a block can never attach, so its extracted metadata
    would be discarded wholesale; the stage must fail fast (before the map compute, in
    every worker) naming the block, the missing column(s), and the columns that exist,
    rather than degrade to a codes-only output behind a warning.
    """
    messy = """\
data:
  event:
    code: 'f"{$a}//{$b}"'
    _metadata:
      m:
        a: $a
        b: $b
        description: $title
"""
    with (
        tempfile.TemporaryDirectory() as d,
        pytest.raises(ValueError, match=r"component column\(s\) \['b'\] that no extracted-event file"),
    ):
        _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "data": pl.DataFrame(
                    {
                        "code": ["X//Y"],
                        "code_components": [{"a": "X"}],  # no "b" component
                        "source_block": ["data/event"],
                    }
                )
            },
            # The raw metadata itself has both match columns; only the component join
            # side is impossible, and that alone must already fail the stage.
            raw_files={"m.csv": "a,b,title\nX,Y,Some title\n"},
        )


def test_partial_match_zero_matches_warns(caplog):
    """A metadata join that matches zero codes emits a WARNING (minimal diagnostic).

    Richer match-coverage diagnostics are a tracked follow-up; this only guards the silent-miss case the dtype
    normalization could otherwise introduce.
    """
    messy = """\
chartevents:
  chart:
    code: 'f"CHART//{$itemid}"'
    _metadata:
      d_items:
        itemid: $itemid
        description: $label
"""
    with tempfile.TemporaryDirectory() as d, caplog.at_level("WARNING"):
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "chartevents": pl.DataFrame(
                    {
                        "code": ["CHART//1"],
                        "code_components": [{"itemid": "1"}],
                        "source_block": ["chartevents/chart"],
                    }
                )
            },
            raw_files={"d_items.csv": "itemid,label\n999,No Such Item\n"},
        )

        assert len(codes_df.filter(pl.col("description").is_not_null())) == 0
        assert "matched zero codes" in caplog.text


# ── Reducer determinism, canonical schema, coalescing merge, and crash bugs ──


_TWO_SOURCE_MESSY = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      source_a:
        lab_code: $lab_code
        description: $title_a
        vocab: $vocab_a
      source_b:
        lab_code: $lab_code
        description: $title_b
        vocab: $vocab_b
"""

_TWO_SOURCE_EVENTS = {"data": _bare_code_events("lab_code", ["HR", "TEMP"], "data/measurement")}

_TWO_SOURCE_RAW = {
    "source_a.csv": "lab_code,title_a,vocab_a\nHR,Heart Rate,LOINC\nTEMP,Temperature,LOINC\n",
    "source_b.csv": "lab_code,title_b,vocab_b\nHR,Pulse Rate,SNOMED\n",
}


def test_reduction_is_deterministic_across_config_orderings(monkeypatch):
    """Regression guard: ``codes.parquet`` is byte-identical regardless of shuffle order.

    Each worker shuffles its metadata configs (for lock-contention spreading), and the shuffle
    order used to flow straight into the reduction: description join order and code_template
    selection depended on which partial file was concatenated first. Two runs over identical
    inputs are forced through *opposite* config orderings here — a no-op shuffle vs. a
    reversing shuffle — and must produce byte-identical output files.
    """
    from MEDS_extract.extract_code_metadata import extract_code_metadata as ecm_mod

    outputs: list[bytes] = []
    frames: list[pl.DataFrame] = []
    for shuffle in (lambda x: None, lambda x: x.reverse()):
        with tempfile.TemporaryDirectory() as d:
            monkeypatch.setattr(ecm_mod.random, "shuffle", shuffle)
            _run_ecm_scenario(Path(d), _TWO_SOURCE_MESSY, _TWO_SOURCE_EVENTS, _TWO_SOURCE_RAW)
            fp = Path(d) / "metadata_out" / "metadata" / "codes.parquet"
            outputs.append(fp.read_bytes())
            frames.append(pl.read_parquet(fp))

    from polars.testing import assert_frame_equal

    # Frame-level identity first (row order, list order, dtypes all strict) for a readable
    # diff on failure; then full byte identity of the on-disk files.
    assert_frame_equal(frames[0], frames[1], check_row_order=True, check_column_order=True)
    assert outputs[0] == outputs[1], "codes.parquet bytes differ across config orderings"

    # The canonical ordering has teeth: descriptions joined in config order, values sorted.
    by_code = {r["code"]: r for r in frames[0].iter_rows(named=True)}
    assert by_code["HR"]["description"] == "Heart Rate\nPulse Rate"
    assert by_code["HR"]["vocab"] == ["LOINC", "SNOMED"]


def test_reduction_is_deterministic_across_engines():
    """Regression guard: ``codes.parquet`` is byte-identical regardless of the polars engine.

    The reducer's expansion join feeds order-sensitive aggregations — description strings are
    separator-joined and ``parent_codes`` lists are deduplicated in first-seen order — so
    within-key join order must be input-determined, not engine-scheduled. The streaming engine
    (selected here exactly as ``POLARS_ENGINE_AFFINITY=streaming`` would select it) partitions
    joins across threads and reorders within-key rows unless the join pins its order — but only
    probabilistically, so the scenario amplifies the signal: many metadata rows for each of many
    codes, and two streaming runs, so an unpinned join escapes only if every code coincidentally
    stays ordered in both. Two default-engine runs and two streaming-engine runs over identical
    inputs must all produce byte-identical output files.
    """
    n_codes, n_per_code = 8, 500
    codes = [f"C{i}" for i in range(n_codes)]
    meta_rows = "".join(
        f"{code},title {j:03d} for {code},PARENT//{code}//{j:03d}\n"
        for j in range(n_per_code)
        for code in codes
    )
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      lab_meta:
        lab_code: $lab_code
        description: $title
        parent_codes: $parent
"""
    events = {"data": _bare_code_events("lab_code", codes, "data/measurement")}
    raw = {"lab_meta.csv": f"lab_code,title,parent\n{meta_rows}"}

    engines = (None, None, "streaming", "streaming")
    outputs: list[bytes] = []
    frames: list[pl.DataFrame] = []
    for engine_affinity in engines:
        with tempfile.TemporaryDirectory() as d, pl.Config(engine_affinity=engine_affinity):
            _run_ecm_scenario(Path(d), messy, events, raw)
            fp = Path(d) / "metadata_out" / "metadata" / "codes.parquet"
            outputs.append(fp.read_bytes())
            frames.append(pl.read_parquet(fp))

    from polars.testing import assert_frame_equal

    # Frame-level identity first (row order, list order, dtypes all strict) for a readable
    # diff on failure; then full byte identity of the on-disk files.
    for i, engine_affinity in enumerate(engines[1:], start=1):
        assert_frame_equal(frames[0], frames[i], check_row_order=True, check_column_order=True)
        assert outputs[0] == outputs[i], (
            f"codes.parquet bytes differ between a default-engine run and run {i} "
            f"(engine affinity {engine_affinity or 'default'})"
        )

    # The pinned order has teeth: values aggregate in metadata-file row order, per code.
    by_code = {r["code"]: r for r in frames[0].iter_rows(named=True)}
    for code in codes:
        assert by_code[code]["description"] == "\n".join(
            f"title {j:03d} for {code}" for j in range(n_per_code)
        )
        assert by_code[code]["parent_codes"] == [f"PARENT//{code}//{j:03d}" for j in range(n_per_code)]


def test_reduced_schema_is_data_independent():
    """Regression guard: extra metadata columns are always ``List(String)``.

    The old reducer only aggregated when some code was duplicated across metadata rows, so
    the *dtype* of extra columns flipped between ``String`` and ``List(String)`` depending on
    the data. A single-source, unique-code extraction must now yield the same schema as a
    multi-source one — one-element lists — while ``description`` keeps its MEDS-mandated
    separator-joined String form.
    """
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      lab_meta:
        lab_code: $lab_code
        description: $title
        vocab: $vocab
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": _bare_code_events("lab_code", ["HR"], "data/measurement")},
            raw_files={"lab_meta.csv": "lab_code,title,vocab\nHR,Heart Rate,LOINC\n"},
        )

    assert codes_df.schema["description"] == pl.String
    assert codes_df.schema["vocab"] == pl.List(pl.String)
    assert codes_df.schema["code_template"] == pl.String
    row = codes_df.filter(pl.col("code") == "HR").to_dicts()[0]
    assert row["description"] == "Heart Rate"
    assert row["vocab"] == ["LOINC"]
    assert row["code_template"] == "$lab_code"


def test_reduced_missing_values_are_null_not_empty():
    """A code with no value for a metadata column gets null — never ``[]`` or ``""``.

    ``TEMP`` appears only in source_a, so its source_b-only column must be null, and its
    description must be exactly the single source_a value (no stray separator).
    """
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(Path(d), _TWO_SOURCE_MESSY, _TWO_SOURCE_EVENTS, _TWO_SOURCE_RAW)

    temp_row = codes_df.filter(pl.col("code") == "TEMP").to_dicts()[0]
    assert temp_row["description"] == "Temperature"
    assert temp_row["vocab"] == ["LOINC"]


def test_preexisting_codes_merge_coalesces_overlapping_columns():
    """Regression guard: merging with a pre-existing ``codes.parquet`` coalesces columns.

    The full join used to fork overlapping columns into ``description`` + ``description_right``.
    Overlaps must coalesce into a single column with extracted values taking precedence;
    pre-existing values survive wherever nothing was re-extracted. This also covers the
    basic pre-existing-codes merge path (a former standalone test): extracted metadata
    joins onto event codes while pre-existing-only rows survive intact.
    """
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      lab_meta:
        lab_code: $lab_code
        description: $title
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": _bare_code_events("lab_code", ["HR", "NEW"], "data/measurement")},
            raw_files={"lab_meta.csv": "lab_code,title\nHR,Fresh heart rate\nNEW,A new code\n"},
            existing_codes=pl.DataFrame(
                {
                    "code": ["HR", "LEGACY"],
                    "description": ["Stale heart rate", "Legacy-only code"],
                }
            ),
        )

    assert "description_right" not in codes_df.columns
    assert codes_df.columns.count("description") == 1
    by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
    # Extracted value wins on overlap.
    assert by_code["HR"] == "Fresh heart rate"
    # Pre-existing survives where not re-extracted; newly extracted codes appear.
    assert by_code["LEGACY"] == "Legacy-only code"
    assert by_code["NEW"] == "A new code"


def test_preexisting_codes_merge_dtype_conflict_names_column():
    """A dtype conflict between pre-existing and extracted same-named columns raises clearly."""
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      lab_meta:
        lab_code: $lab_code
        description: $title
        vocab: $vocab
"""
    with (
        tempfile.TemporaryDirectory() as d,
        pytest.raises(ValueError, match="column 'vocab'"),
    ):
        _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": _bare_code_events("lab_code", ["HR"], "data/measurement")},
            raw_files={"lab_meta.csv": "lab_code,title,vocab\nHR,Heart Rate,LOINC\n"},
            # Pre-existing `vocab` is String; the extracted canonical form is List(String).
            existing_codes=pl.DataFrame({"code": ["HR"], "vocab": ["OLD"]}),
        )


def test_reduced_metadata_values_are_deduplicated():
    """Identical metadata values contributed by multiple sources collapse to one.

    Both sources give HR the same description and the same vocab value; the reduced output must carry the
    description once (no doubled separator join) and a single-element vocab list — repeating identical
    metadata per code is pure waste.
    """
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      dup_a:
        lab_code: $lab_code
        description: $title_a
        vocab: $vocab_a
      dup_b:
        lab_code: $lab_code
        description: $title_b
        vocab: $vocab_b
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": _bare_code_events("lab_code", ["HR"], "data/measurement")},
            raw_files={
                "dup_a.csv": "lab_code,title_a,vocab_a\nHR,Heart Rate,LOINC\n",
                "dup_b.csv": "lab_code,title_b,vocab_b\nHR,Heart Rate,LOINC\n",
            },
        )

    hr_row = codes_df.filter(pl.col("code") == "HR").to_dicts()[0]
    assert hr_row["description"] == "Heart Rate"
    assert hr_row["vocab"] == ["LOINC"]


def test_conflicting_code_templates_error():
    """Distinct code templates colliding on one code raise a clear config error.

    Two event blocks emit the same literal code string through different code
    expressions; ``code_template`` is a single String per code, so the collision must
    surface as an error rather than an arbitrary pick or a widened schema.
    """
    messy = """\
a_tbl:
  m:
    code: $lab_code
    _metadata:
      src_a:
        lab_code: $lab_code
        description: $title_a
b_tbl:
  m:
    code: $med_code
    _metadata:
      src_b:
        med_code: $med_code
        description: $title_b
"""
    with (
        tempfile.TemporaryDirectory() as d,
        pytest.raises(ValueError, match="multiple distinct code templates"),
    ):
        _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "a_tbl": _bare_code_events("lab_code", ["HR"], "a_tbl/m"),
                "b_tbl": _bare_code_events("med_code", ["HR"], "b_tbl/m"),
            },
            raw_files={
                "src_a.csv": "lab_code,title_a\nHR,From A\n",
                "src_b.csv": "med_code,title_b\nHR,From B\n",
            },
        )


def test_mixed_format_metadata_prefix_errors_clearly():
    """A metadata prefix mixing csv and parquet chunks is a config error, not a coercion.

    Typed parquet concatenated with all-String csv would silently unify dtypes through
    ``vertical_relaxed`` (and previously crashed with a ``TypeError`` when csv reader
    options reached ``scan_parquet``); the scan now rejects the mix up front with a
    pointer at the fix.
    """
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      mixed_meta:
        lab_code: $lab_code
        description: $title
"""
    with (
        tempfile.TemporaryDirectory() as d,
        pytest.raises(ValueError, match="mix of csv- and parquet-family files"),
    ):
        _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": _bare_code_events("lab_code", ["HR"], "data/measurement")},
            raw_files={
                "mixed_meta/[0-1).csv": "lab_code,title\nHR,Heart Rate\n",
                "mixed_meta/[1-2).parquet": pl.DataFrame({"lab_code": ["TEMP"], "title": ["Body Temp"]}),
            },
        )


def _finish_parquet_after(fp: Path, delay: float, df: pl.DataFrame) -> None:
    """Sleep briefly, then flush a valid parquet over ``fp``.

    The initial zero-byte create has already happened on the main thread; this
    helper simulates the "another worker is still writing" half of the race.
    ``delay`` is chosen to comfortably exceed the polling period so the test
    stays deterministic under CI jitter.
    """
    time.sleep(delay)
    df.write_parquet(fp)


def test_reducer_does_not_crash_on_partial_parquet_from_concurrent_worker(tmp_path):
    """Reducer's polling helper waits past partial parquets — regression for #51.

    Pre-fix (polling used ``Path.exists()``) the loop returned immediately because both
    paths existed, and ``scan_parquet`` on the zero-byte file crashed with ``ComputeError``.
    Post-fix (polling uses ``is_complete_parquet_file``) the loop blocks until the writer
    thread flushes, ``scan_parquet`` succeeds, and both rows are visible. The test imports
    the production helper instead of inlining a copy of the polling logic, so a regression
    that re-introduces ``Path.exists()`` fails here in isolation rather than only
    manifesting in the multi-worker integration test.
    """
    from MEDS_extract.extract_code_metadata.extract_code_metadata import wait_for_complete_parquets

    complete_fp = tmp_path / "shard_a_0.parquet"
    pl.DataFrame({"code": ["A"], "code_template": ["x"]}).write_parquet(complete_fp)

    # Simulate the mid-write shard: another worker has just called
    # ``pl.DataFrame.write_parquet(partial_fp, ...)`` which internally creates the
    # path before flushing the footer. We model that by touching the file empty
    # and handing off to a background thread that will finish the write ~0.5s later.
    partial_fp = tmp_path / "shard_b_0.parquet"
    partial_fp.write_bytes(b"")

    writer = threading.Thread(
        target=_finish_parquet_after,
        args=(partial_fp, 0.5, pl.DataFrame({"code": ["B"], "code_template": ["y"]})),
    )
    writer.start()

    all_out_fps = [complete_fp, partial_fp]

    # Polling time matches ``_make_cfg``'s default (0.1s) so the loop's cadence is
    # representative of a real pipeline invocation. The 5s wall-clock cap is a
    # safety net so a wedged test doesn't hang CI; the writer thread should
    # finish in 0.5s, comfortably under the cap.
    poll_thread = threading.Thread(target=wait_for_complete_parquets, args=(all_out_fps, 0.1), daemon=True)
    poll_thread.start()
    poll_thread.join(timeout=5.0)
    if poll_thread.is_alive():
        writer.join()
        pytest.fail(
            "#51: wait_for_complete_parquets failed to converge within 5s. "
            "The polling helper should have detected partial_fp's flush and exited."
        )

    dfs = [pl.scan_parquet(fp, glob=False) for fp in all_out_fps]
    try:
        result = pl.concat(dfs, how="diagonal_relaxed").collect()
    except pl.exceptions.ComputeError as e:
        pytest.fail(
            "#51: reducer crashed on a partial parquet from a concurrent worker write.\n"
            f"  underlying error: {e}\n"
            "  expected: wait_for_complete_parquets should have blocked until the "
            "writer flushed the parquet footer."
        )
    finally:
        writer.join()

    # Post-fix the reducer sees both rows — the polling helper kept going past
    # the zero-byte partial_fp, the writer flushed its footer, and scan_parquet
    # reads the valid file.
    assert set(result["code"].to_list()) == {"A", "B"}


def test_atomic_write_parquet_never_exposes_partial_state(tmp_path):
    """``atomic_write_parquet`` writes via a sibling ``.tmp`` and renames into place.

    Belt-and-suspenders companion to the polling fix above: ``atomic_write_parquet``
    is what we now pass to ``rwlock_wrap`` so the destination path appears
    atomically as the fully-written parquet — no zero-byte intermediate state for
    a concurrent reader to ``stat``. This test asserts the invariant directly:
    while the writer thread is still running, ``out_fp`` either does not exist
    yet OR is already a valid parquet; it is never a half-written file.
    """
    from MEDS_extract.extract_code_metadata.extract_code_metadata import atomic_write_parquet

    out_fp = tmp_path / "shard.parquet"

    observations: list[bool] = []

    def writer():
        atomic_write_parquet(pl.DataFrame({"code": list(range(10_000))}), out_fp)

    stop = threading.Event()

    def watcher():
        # Sample the destination as fast as Python lets us. If atomic-write is
        # working, every observation where ``out_fp`` exists must be a valid
        # parquet — we never see the path with a non-parquet body.
        while not stop.is_set():
            if out_fp.exists():
                try:
                    pl.scan_parquet(out_fp, glob=False).collect()
                    observations.append(True)
                except pl.exceptions.ComputeError:
                    observations.append(False)
            else:
                observations.append(True)  # not-yet-existing is fine

    w = threading.Thread(target=writer)
    s = threading.Thread(target=watcher)
    s.start()
    w.start()
    w.join()
    # Writer is done, so signal the watcher to stop sampling and wait for it to
    # exit before asserting — every observation it took gets checked.
    stop.set()
    s.join()

    assert all(observations), (
        f"atomic_write_parquet exposed a partial-parquet state to a concurrent reader; "
        f"observations: {observations.count(True)} ok, {observations.count(False)} crashed."
    )
    # And the .tmp staging file must not leak into the final tree.
    leftover = list(tmp_path.glob("*.tmp"))
    assert not leftover, f"unexpected .tmp leftovers: {leftover}"


def test_component_map_built_only_by_the_reducer_worker(monkeypatch):
    """Non-zero workers never materialize the code-component map (map-phase cost gate).

    The component map is a full-dataset scan/unique/collect consumed exclusively by
    worker 0's reduction, so the N-1 map-only workers must not pay for building it.
    ``build_code_component_map`` is spied via monkeypatch: a worker-1 run must not call
    it (and must not write ``codes.parquet``), while the subsequent worker-0 run over
    the same layout calls it exactly once and reduces normally.
    """
    from MEDS_extract.extract_code_metadata import extract_code_metadata as ecm

    calls = []
    real_build = ecm.build_code_component_map

    def spying_build(all_data):
        calls.append(1)
        return real_build(all_data)

    monkeypatch.setattr(ecm, "build_code_component_map", spying_build)

    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      lab_meta:
        lab_code: $lab_code
        description: $title
"""
    scenario_kwargs = {
        "event_frames": {"data": _bare_code_events("lab_code", ["HR"], "data/measurement")},
        "raw_files": {"lab_meta.csv": "lab_code,title\nHR,Heart Rate\n"},
    }

    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(Path(d), messy, worker=1, **scenario_kwargs)
        partials = [
            fp.name
            for fp in (Path(d) / "metadata_out" / "metadata").glob("*.parquet")
            if fp.name != "codes.parquet"
        ]
    assert codes_df is None, "A non-reducer worker must not write codes.parquet."
    # The map phase must actually have run — otherwise the no-component-map assertion
    # below is vacuous (a worker that exits before the map loop trivially calls nothing).
    assert partials, "Worker 1 wrote no partial metadata parquet: the map phase never ran."
    assert calls == [], (
        "Worker 1 built the code-component map: the full-dataset collect must be "
        "gated behind the worker-0 reduction."
    )

    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(Path(d), messy, worker=0, **scenario_kwargs)
    assert calls == [1], f"Worker 0 should build the map exactly once; got {len(calls)} calls."
    assert codes_df.filter(pl.col("code") == "HR")["description"].to_list() == ["Heart Rate"]


def test_self_metadata_end_to_end(tmp_path):
    """A '_self' block attaches metadata sourced from the event's own extracted rows — no raw metadata table
    exists at all, so nothing raw is scanned or joined."""
    messy_yaml = """\
chartevents:
  _defaults:
    subject_id: "$subject_id"
  chart:
    code: 'f"CHART//{$itemid}"'
    time: null
    _metadata:
      _self:
        description: "$long_label"
"""
    events = pl.DataFrame(
        {
            "code": ["CHART//10", "CHART//10", "CHART//20"],
            "code_components": [{"itemid": 10}, {"itemid": 10}, {"itemid": 20}],
            "metadata_components": [
                {"description": "Heart Rate"},
                {"description": "Heart Rate"},
                {"description": "Resp Rate"},
            ],
            "source_block": ["chartevents/chart"] * 3,
        }
    )
    codes = _run_ecm_scenario(tmp_path, messy_yaml, {"chartevents": events}, raw_files={})
    got = {r["code"]: r["description"] for r in codes.iter_rows(named=True)}
    assert got == {"CHART//10": "Heart Rate", "CHART//20": "Resp Rate"}


def test_self_metadata_conflicting_values_warn_and_aggregate(tmp_path, caplog):
    """One component key carrying distinct '_self' metadata values warns (data leaking into metadata) and the
    reducer aggregates the conflict (description joined)."""
    messy_yaml = """\
chartevents:
  _defaults:
    subject_id: "$subject_id"
  chart:
    code: 'f"CHART//{$itemid}"'
    time: null
    _metadata:
      _self:
        description: "$long_label"
"""
    events = pl.DataFrame(
        {
            "code": ["CHART//10", "CHART//10"],
            "code_components": [{"itemid": 10}, {"itemid": 10}],
            "metadata_components": [{"description": "HR"}, {"description": "Pulse"}],
            "source_block": ["chartevents/chart"] * 2,
        }
    )
    with caplog.at_level("WARNING"):
        codes = _run_ecm_scenario(tmp_path, messy_yaml, {"chartevents": events}, raw_files={})
    assert any("data leaking into metadata" in r.message for r in caplog.records)
    (desc,) = codes.filter(pl.col("code") == "CHART//10")["description"].to_list()
    assert sorted(desc.split("\n")) == ["HR", "Pulse"]


def test_self_metadata_without_components_column_errors(tmp_path):
    """'_self' blocks over event files lacking 'metadata_components' (a stale extraction) error with a re-run
    remedy instead of a cryptic struct-field failure."""
    messy_yaml = """\
chartevents:
  _defaults:
    subject_id: "$subject_id"
  chart:
    code: 'f"CHART//{$itemid}"'
    time: null
    _metadata:
      _self:
        description: "$long_label"
"""
    events = pl.DataFrame(
        {
            "code": ["CHART//10"],
            "code_components": [{"itemid": 10}],
            "source_block": ["chartevents/chart"],
        }
    )
    with pytest.raises(ValueError, match="metadata_components"):
        _run_ecm_scenario(tmp_path, messy_yaml, {"chartevents": events}, raw_files={})


def test_self_metadata_sentinel_dtypes_match_external_path(tmp_path):
    """'_self' sentinel outputs get the same mandatory-type coercions as external
    blocks: a non-String 'description' is cast to String and a scalar 'parent_codes'
    lands as the mandated List(String) — instead of crashing the reducer's list.join
    or emitting a MEDS-noncompliant dtype."""
    messy_yaml = """\
chartevents:
  _defaults:
    subject_id: "$subject_id"
  chart:
    code: 'f"CHART//{$itemid}"'
    time: null
    _metadata:
      _self:
        description: "$int_label"
        parent_codes: "$parent_id"
"""
    events = pl.DataFrame(
        {
            "code": ["CHART//10", "CHART//20"],
            "code_components": [{"itemid": 10}, {"itemid": 20}],
            "metadata_components": [
                {"description": 111, "parent_codes": 999},
                {"description": 222, "parent_codes": 888},
            ],
            "source_block": ["chartevents/chart"] * 2,
        }
    )
    codes = _run_ecm_scenario(tmp_path, messy_yaml, {"chartevents": events}, raw_files={})
    assert codes.schema["description"] == pl.String
    assert codes.schema["parent_codes"] == pl.List(pl.String)
    got = {r["code"]: (r["description"], r["parent_codes"]) for r in codes.iter_rows(named=True)}
    assert got == {"CHART//10": ("111", ["999"]), "CHART//20": ("222", ["888"])}


def test_self_metadata_partially_stale_events_error(tmp_path):
    """A '_self' block whose declaring table's files predate the config errors with a re-run remedy even when
    ANOTHER table's fresh files carry a same-named metadata field — instead of silently emitting null metadata
    for the stale table's codes."""
    messy_yaml = """\
labs:
  _defaults:
    subject_id: "$subject_id"
  lab:
    code: 'f"LAB//{$test}"'
    time: null
    _metadata:
      _self:
        description: "$name"
chartevents:
  _defaults:
    subject_id: "$subject_id"
  chart:
    code: 'f"CHART//{$itemid}"'
    time: null
    _metadata:
      _self:
        description: "$long_label"
"""
    stale_labs = pl.DataFrame(
        {
            "code": ["LAB//1"],
            "code_components": [{"test": "1"}],
            "source_block": ["labs/lab"],
        }
    )
    fresh_charts = pl.DataFrame(
        {
            "code": ["CHART//10"],
            "code_components": [{"itemid": 10}],
            "metadata_components": [{"description": "HR"}],
            "source_block": ["chartevents/chart"],
        }
    )
    with pytest.raises(ValueError, match=r"labs/lab.*carry\s+no 'metadata_components'"):
        _run_ecm_scenario(
            tmp_path, messy_yaml, {"labs": stale_labs, "chartevents": fresh_charts}, raw_files={}
        )
