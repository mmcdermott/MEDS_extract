"""Stage-level tests for ``extract_code_metadata`` — end-to-end runs of the real stage.

Single-function behavior (code construction, ``_match_on`` validation, null-component
rendering) is doctested on the helpers themselves (``extract_metadata``,
``EventConfig.extract``). This file covers behavior that needs the full mapper/reducer
machinery: joining extracted metadata onto codes, partial-match (``_match_on``)
expansion and its per-event scoping, reducer determinism and the canonical output
schema, merging with a pre-existing ``codes.parquet``, and reducer/worker concurrency.

Most tests run through :func:`_run_ecm_scenario`, which lays out synthetic event shards
and raw metadata files and invokes the stage's ``main_fn`` as worker 0.
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
) -> pl.DataFrame | None:
    """Run the extract_code_metadata stage over synthetic event shards and raw metadata files.

    ``event_frames`` maps parquet basenames to frames of extra columns (code, code_components,
    source_block, ...); the standard subject_id/time/numeric_value columns are added here.
    ``raw_files`` maps raw metadata file paths (which may include subdirectories) to either
    text content (written verbatim) or a DataFrame (written as parquet). ``existing_codes``,
    when given, is written as a pre-existing ``metadata/codes.parquet`` for the reducer to
    merge with. Returns the reduced ``codes.parquet`` as a DataFrame, or ``None`` if the
    stage exited without writing one (the no-metadata-blocks early return).
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

    event_cfg_fp = root / "event_cfgs.yaml"
    event_cfg_fp.write_text(messy_yaml)
    shards_fp = root / "metadata" / ".shards.json"
    shards_fp.parent.mkdir(parents=True)
    shards_fp.write_text(json.dumps({"train/0": [1]}))

    out_dir = root / "metadata_out" / "metadata"
    out_dir.mkdir(parents=True)

    cfg = _make_cfg(
        {
            "input_dir": str(raw_dir),
            "stage_cfg": {
                "data_input_dir": str(root / "events"),
                "output_dir": str(out_dir),
                "metadata_input_dir": str(metadata_in),
                "reducer_output_dir": str(out_dir),
                "description_separator": description_separator,
            },
            "event_conversion_config_fp": str(event_cfg_fp),
            "shards_map_fp": str(shards_fp),
        }
    )
    ecm_stage.main_fn(cfg)
    codes_fp = out_dir / "codes.parquet"
    return pl.read_parquet(codes_fp) if codes_fp.exists() else None


# ── Full-match basics: joining metadata onto extracted codes ──


def test_extract_code_metadata_with_existing_codes():
    """Extracted metadata joins onto event codes and merges with a pre-existing codes.parquet."""
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      lab_meta:
        description: title
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": pl.DataFrame({"code": ["HR", "TEMP"]})},
            raw_files={
                "lab_meta.csv": "lab_code,title,loinc\nHR,Heart Rate,8867-4\nTEMP,Temperature,8310-5\n"
            },
            existing_codes=pl.DataFrame({"code": ["EXISTING_CODE"], "description": ["An existing code"]}),
        )

    # Overlapping columns are coalesced, never forked into `*_right`.
    assert "description_right" not in codes_df.columns
    by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
    assert by_code["EXISTING_CODE"] == "An existing code"
    assert by_code["HR"] == "Heart Rate"
    assert by_code["TEMP"] == "Temperature"


def test_extract_code_metadata_multiple_files_per_prefix():
    """Multiple CSV files matching one metadata prefix are concatenated before extraction."""
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      lab_meta:
        description: title
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": pl.DataFrame({"code": ["HR", "TEMP"]})},
            # Two CSV files in a sub-sharded `lab_meta/` directory — triggers multi-file concat.
            raw_files={
                "lab_meta/part1.csv": "lab_code,title\nHR,Heart Rate\n",
                "lab_meta/part2.csv": "lab_code,title\nTEMP,Body Temperature\n",
            },
        )

    by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
    assert by_code["HR"] == "Heart Rate"
    assert by_code["TEMP"] == "Body Temperature"


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
        description: title_a
      source_b:
        description: title_b
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": pl.DataFrame({"code": ["HR"]})},
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
        custom_prop: val_a
      source_b:
        custom_prop: val_b
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": pl.DataFrame({"code": ["HR"]})},
            raw_files={
                "source_a.csv": "lab_code,val_a\nHR,value_1\n",
                "source_b.csv": "lab_code,val_b\nHR,value_2\n",
            },
        )

    assert codes_df.schema["custom_prop"] == pl.List(pl.String)
    hr_row = codes_df.filter(pl.col("code") == "HR")
    assert hr_row["custom_prop"][0].to_list() == ["value_1", "value_2"]


def test_extract_code_metadata_code_template_survives_aggregation():
    """code_template aggregates to a sorted unique List(String).

    ``.first()`` used to keep only whichever source's template happened to arrive first —
    nondeterministic under worker shuffle, and silently dropping the other sources'
    provenance. The canonical shape preserves every contributing template exactly once:
    both sources here share the same code expression, so exactly one template survives.
    """
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      source_a:
        description: title_a
      source_b:
        description: title_b
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": pl.DataFrame({"code": ["HR"]})},
            raw_files={
                "source_a.csv": "lab_code,title_a\nHR,Heart Rate\n",
                "source_b.csv": "lab_code,title_b\nHR,Pulse Rate\n",
            },
            description_separator="; ",
        )

    assert codes_df.schema["code_template"] == pl.List(pl.String)
    hr_row = codes_df.filter(pl.col("code") == "HR")
    assert hr_row["code_template"][0].to_list() == ["$lab_code"]


def test_extract_code_metadata_handles_code_named_source_column():
    """Codes built from a source column literally named ``code`` flow through the full stage.

    The idiomatic ICD/OMOP vocabulary-table shape — ``code: f"ICD//{$code}"`` — must flow
    through the ``code_components`` map build, full-match extraction, and reduction without
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
        description: long_title
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


# ── Early returns and empty outputs ──


def test_extract_code_metadata_no_metadata_blocks():
    """With no _metadata blocks in the event config, the stage returns early: no codes.parquet."""
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
            event_frames={"data": pl.DataFrame({"code": ["HR"]})},
            raw_files={},
        )
    assert codes_df is None


def test_partial_match_without_code_components_in_events():
    """Partial-match metadata with no code_components in the event data yields an empty output.

    Covers the warning path: the stage completes without error, but partial-match rows
    cannot be expanded without component provenance.
    """
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      lab_meta:
        _match_on: lab_code
        description: title
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            # Event file WITHOUT code_components (literal code via $col).
            event_frames={"data": pl.DataFrame({"code": ["HR"]})},
            raw_files={"lab_meta.csv": "lab_code,title\nHR,Heart Rate\n"},
        )
    assert len(codes_df) == 0


def test_extract_code_metadata_no_matching_codes():
    """Metadata whose keys match no observed event code reduces to an empty codes.parquet."""
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      lab_meta:
        description: title
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": pl.DataFrame({"code": ["HR"]})},
            raw_files={"lab_meta.csv": "lab_code,title\nNONEXISTENT,No Match\n"},
        )
    assert len(codes_df) == 0


# ── Mixed-schema and mixed-mode mapper/reducer regressions ──


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
        description: title
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


def test_partial_match_join_key_not_inferred_from_schema_intersection():
    """Regression guard: partial-match join keys must use explicit _match_on, not schema intersection.

    If a metadata output column shares a name with a code component column, it must not be treated
    as a join key. Only the explicit _match_on columns should be used. Previously the reducer
    inferred join keys from schema intersection, which over-constrained the join and silently
    dropped matches when column names collided.

    All event files here use dynamic codes (uniform schema) to isolate this from the mixed-schema bug.
    """
    # Code is f"{$category}//{$item}" — so code_component_map has columns: code, category, item.
    # Metadata is keyed on _match_on: category, and has an output column ALSO named "item"
    # containing description text, not actual item values.
    messy = """\
data:
  event:
    code: 'f"{$category}//{$item}"'
    _metadata:
      category_meta:
        _match_on: category
        item: item_description
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

    # The "item" column should be present in the output — it's a metadata output column.
    assert "item" in codes_df.columns, (
        f"Expected 'item' column in output (metadata output from partial match), "
        f"but columns are: {codes_df.columns}.\nFull output:\n{codes_df}"
    )
    codes_with_item = codes_df.filter(pl.col("item").is_not_null())
    # Both Drug//Aspirin and Drug//Ibuprofen should get "Pharmaceutical compound"
    # because _match_on is only "category" and both share category=Drug.
    # With the bug, the join also matches on "item" column, so neither row matches
    # (because "Pharmaceutical compound" != "Aspirin" or "Ibuprofen").
    assert len(codes_with_item) == 2, (
        f"Expected 2 codes with item metadata (both Drug codes should match via category), "
        f"got {len(codes_with_item)}.\nFull output:\n{codes_df}"
    )


def test_mixed_full_and_partial_match_from_same_metadata_prefix():
    """Regression guard: mixed full-match and partial-match configs sharing a metadata prefix.

    A single metadata file prefix can be referenced by multiple event configs with different
    match modes. Each must be written to a separate intermediate shard so the reducer can
    classify and expand them independently. Previously all configs for one prefix were
    concatenated into one shard, and the reducer treated the whole shard as full-match
    (because "code" was in the schema), silently dropping partial-match rows.

    This test uses "shared_meta" referenced by a full-match config (code: $lab_code) and a
    partial-match config (code: f"{$category}//{$item}", _match_on: category).
    """
    messy = """\
labs:
  measurement:
    code: $lab_code
    _metadata:
      shared_meta:
        description: desc
products:
  product:
    code: 'f"{$category}//{$item}"'
    _metadata:
      shared_meta:
        _match_on: category
        description: desc
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                # Lab events (full match — code is a simple column ref, no code_components).
                "labs": pl.DataFrame({"code": ["HR"], "source_block": ["labs/measurement"]}),
                # Product events (partial match — dynamic code with code_components).
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
            # Shared metadata file: "HR" matches full-match via lab_code; "Drug" matches
            # partial-match via category.
            raw_files={"shared_meta.csv": "lab_code,category,desc\nHR,Drug,Shared description\n"},
        )

    codes_with_desc = codes_df.filter(pl.col("description").is_not_null())
    matched_codes = set(codes_with_desc["code"].to_list())

    # Full-match: HR should get description from shared_meta via lab_code.
    assert "HR" in matched_codes, f"Full-match code 'HR' missing from output.\n{codes_df}"
    # Partial-match: both Drug codes should get description via category=Drug.
    assert "Drug//Aspirin" in matched_codes, (
        f"Partial-match code 'Drug//Aspirin' missing from output.\n{codes_df}"
    )
    assert "Drug//Ibuprofen" in matched_codes, (
        f"Partial-match code 'Drug//Ibuprofen' missing from output.\n{codes_df}"
    )


# ── Partial-match correctness regressions ──


def test_partial_match_on_column_named_code_is_expanded_not_passed_through():
    """Regression guard: a partial output keyed on a column named ``code`` stays partial-match.

    ``_match_on: code`` makes the intermediate shard carry a ``code`` column of raw component
    values (e.g. ``250.00``). The reducer used to classify shards by sniffing for a ``code``
    column in the schema, misclassifying this shard as full-match and passing the raw
    component values through as output codes — silently wrong codes.parquet. Classification
    must come from explicit map-time bookkeeping instead.
    """
    messy = """\
diagnoses:
  dx:
    code: 'f"ICD//{$code}"'
    _metadata:
      icd_meta:
        _match_on: code
        description: long_title
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
        # Partial-match expansion: metadata lands on the FULL codes...
        assert by_code.get("ICD//250.00") == "Diabetes mellitus"
        assert by_code.get("ICD//401.9") == "Hypertension"
        # ...and the raw component values are NOT passed through as codes (the old
        # full-match misclassification symptom).
        assert "250.00" not in by_code
        assert "401.9" not in by_code


def test_partial_match_scoped_to_declaring_event():
    """Regression guard: ``_match_on`` expansion is scoped to the declaring event.

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
        _match_on: itemid
        description: label
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
        assert chart_rows["code_template"].to_list() == [['f"CHART//{$itemid}"']]

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
        _match_on: itemid
        description: label
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
        _match_on: itemid
        description: label
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


def test_partial_match_zero_matches_warns(caplog):
    """A partial-match join that matches zero codes emits a WARNING (minimal diagnostic).

    Full match-coverage diagnostics are a tracked follow-up; this only guards the silent-miss case the dtype
    normalization could otherwise introduce.
    """
    messy = """\
chartevents:
  chart:
    code: 'f"CHART//{$itemid}"'
    _metadata:
      d_items:
        _match_on: itemid
        description: label
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
        description: title_a
        vocab: vocab_a
      source_b:
        description: title_b
        vocab: vocab_b
"""

_TWO_SOURCE_EVENTS = {"data": pl.DataFrame({"code": ["HR", "TEMP"]})}

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
        description: title
        vocab: vocab
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": pl.DataFrame({"code": ["HR"]})},
            raw_files={"lab_meta.csv": "lab_code,title,vocab\nHR,Heart Rate,LOINC\n"},
        )

    assert codes_df.schema["description"] == pl.String
    assert codes_df.schema["vocab"] == pl.List(pl.String)
    assert codes_df.schema["code_template"] == pl.List(pl.String)
    row = codes_df.filter(pl.col("code") == "HR").to_dicts()[0]
    assert row["description"] == "Heart Rate"
    assert row["vocab"] == ["LOINC"]
    assert row["code_template"] == ["$lab_code"]


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
    pre-existing values survive wherever nothing was re-extracted.
    """
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      lab_meta:
        description: title
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": pl.DataFrame({"code": ["HR", "NEW"]})},
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
        description: title
        vocab: vocab
"""
    with (
        tempfile.TemporaryDirectory() as d,
        pytest.raises(ValueError, match="column 'vocab'"),
    ):
        _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": pl.DataFrame({"code": ["HR"]})},
            raw_files={"lab_meta.csv": "lab_code,title,vocab\nHR,Heart Rate,LOINC\n"},
            # Pre-existing `vocab` is String; the extracted canonical form is List(String).
            existing_codes=pl.DataFrame({"code": ["HR"], "vocab": ["OLD"]}),
        )


def test_match_on_column_that_is_also_a_metadata_output_works():
    """Regression guard: ``_match_on`` on a renamed key column must not crash.

    Declaring the join key as a ``_metadata`` output expression (``itemid: itemid_alias``) is
    the only key-rename mechanism available; it used to raise ``DuplicateError`` at the
    mapper's final select because the column was selected both as key and as output.
    """
    messy = """\
chartevents:
  chart:
    code: 'f"CHART//{$itemid}"'
    _metadata:
      d_items:
        _match_on: itemid
        itemid: itemid_alias
        description: label
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={
                "chartevents": pl.DataFrame(
                    {
                        "code": ["CHART//220045"],
                        "code_components": [{"itemid": "220045"}],
                        "source_block": ["chartevents/chart"],
                    }
                )
            },
            # The metadata table has no `itemid` column — the key is renamed from `itemid_alias`.
            raw_files={"d_items.csv": "itemid_alias,label\n220045,Heart Rate\n"},
        )

    by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
    assert by_code.get("CHART//220045") == "Heart Rate"


def test_mixed_format_metadata_prefix_chunks():
    """Regression guard: a metadata prefix mixing csv and parquet chunks must not crash.

    Read kwargs used to be chosen from the *first* resolved file only, so a csv-first prefix
    passed the csv-only ``infer_schema`` kwarg into ``scan_parquet`` → ``TypeError``. Format
    dispatch now happens per file; typed parquet keys unify with all-String csv keys.
    """
    messy = """\
data:
  measurement:
    code: $lab_code
    _metadata:
      lab_meta:
        description: title
"""
    with tempfile.TemporaryDirectory() as d:
        codes_df = _run_ecm_scenario(
            Path(d),
            messy,
            event_frames={"data": pl.DataFrame({"code": ["HR", "TEMP"]})},
            raw_files={
                # Sub-sharded prefix directory with one csv chunk and one parquet chunk;
                # `resolve_source_files` sorts by name, so the csv is scanned first.
                "lab_meta/chunk_a.csv": "lab_code,title\nHR,Heart Rate\n",
                "lab_meta/chunk_b.parquet": pl.DataFrame(
                    {"lab_code": ["TEMP"], "title": ["Body Temperature"]}
                ),
            },
        )

    by_code = {r["code"]: r["description"] for r in codes_df.iter_rows(named=True)}
    assert by_code.get("HR") == "Heart Rate"
    assert by_code.get("TEMP") == "Body Temperature"


# ── Reducer/worker concurrency (#51): partial parquets and atomic writes ──
#
# Pre-fix, the reducer's polling loop used ``Path.exists()`` to decide when all map-phase
# outputs were ready. ``pl.DataFrame.write_parquet`` opens its destination with ``O_TRUNC``
# (no temp+rename), so the path exists before the parquet footer is flushed, and the
# reducer's ``scan_parquet`` intermittently crashed with a "File out of specification"
# ComputeError under ``N_WORKERS>1`` (a real failure reported against the MIMIC-IV_MEDS
# pipeline). The fix polls with ``is_complete_parquet_file`` and writes via
# ``atomic_write_parquet``. The race only occurs with two OS-level workers on a shared
# filesystem, which can't be scheduled deterministically — so these tests drive the
# production helpers directly with background threads standing in for the racing worker.


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

    def watcher():
        # Sample the destination as fast as Python lets us. If atomic-write is
        # working, every observation where ``out_fp`` exists must be a valid
        # parquet — we never see the path with a non-parquet body.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
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
    # Stop watcher early once writer finishes — no need to keep sampling.
    s.join(timeout=0.1)

    assert all(observations), (
        f"atomic_write_parquet exposed a partial-parquet state to a concurrent reader; "
        f"observations: {observations.count(True)} ok, {observations.count(False)} crashed."
    )
    # And the .tmp staging file must not leak into the final tree.
    leftover = list(tmp_path.glob("*.tmp"))
    assert not leftover, f"unexpected .tmp leftovers: {leftover}"
