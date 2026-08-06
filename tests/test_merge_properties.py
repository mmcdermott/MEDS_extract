"""Property tests pinning the invariants of the merge stage's performance settings.

``merge_subdirs_and_sort`` runs a multithreaded stable sort and, by shipped default, skips
post-merge deduplication. Both settings are only safe because of an invariant the upstream stages
guarantee: every event block ends with a full-column ``.unique(maintain_order=True)``
(``EventConfig.extract``) and stamps its rows with a distinct per-block ``source_block`` value, so
a merged shard can never contain a full-row duplicate. The tests here generate "pipeline-shaped"
shard inputs that honor that invariant while drawing every value from its full type domain —
subject ids and struct ints from all of Int64, times from the full datetime range (plus nulls),
codes and text from arbitrary unicode, numerics from all of Float32 including NaN/inf. Ties on
``(subject_id, time)`` are forced structurally (rows sample from a small per-example pool of
full-domain values), because ties across blocks are exactly where ordering bugs hide.

Each generated example is chained through the real df->df stage helpers (``_filter_to_subjects``
then ``merge_subdirs_and_sort``) and asserts:

- EQUIVALENCE: the shipped settings reproduce the deterministic reference (diagonal concat in
  config order, full-column dedup, stable sort) exactly, including row order — under both
  ``unique_by`` modes;
- STABLE: merging the same inputs twice yields identical output, row order included;
- SORTED: the output is non-decreasing on ``(subject_id, time)``;
- UNIQUE: the output contains no full-row duplicates;
- LOSSLESS: no row is dropped relative to the (post-filter) inputs.

The same suite pins ``convert_to_subject_sharded``'s write side, which sinks its plan on
polars' streaming engine instead of collecting it eagerly. Order determinism is load-bearing
there (merge's stable sort propagates input order into final bytes for same-time events), so
the write-path properties assert the sunk output equals the eager execution of the same plan
exactly — row order included — and that repeated sinks are byte-identical.

Run with ``HYPOTHESIS_PROFILE=thorough`` for a heavier local sweep (300 examples).
"""

import os
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

from MEDS_extract.config import TableConfig
from MEDS_extract.convert_to_subject_sharded.convert_to_subject_sharded import (
    _filter_to_subjects,
    _read_and_join,
    sink_df,
)
from MEDS_extract.merge_to_MEDS_cohort.merge_to_MEDS_cohort import merge_subdirs_and_sort

settings.register_profile("default", max_examples=50, deadline=None)
settings.register_profile("thorough", max_examples=300, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))

_I64 = 2**63

# Safe-for-filenames identifiers: prefixes become ``<prefix>.parquet`` paths, and struct field
# names must be distinct from the fixed MEDS column names — everything else about them is free.
_identifiers = st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=12)

_maybe_datetime = st.one_of(st.none(), st.datetimes())
_maybe_f32 = st.one_of(st.none(), st.floats(width=32, allow_nan=True, allow_infinity=True))
_maybe_text = st.one_of(st.none(), st.text(max_size=12))
_maybe_i64 = st.one_of(st.none(), st.integers(min_value=-_I64, max_value=_I64 - 1))


@st.composite
def _component_schema(draw: st.DrawFn) -> dict[str, pl.DataType]:
    """Per-prefix ``code_components`` struct fields: 1-3 named fields of mixed dtypes."""
    names = draw(st.lists(_identifiers, min_size=1, max_size=3, unique=True))
    return {n: draw(st.sampled_from([pl.String, pl.Int64])) for n in names}


@st.composite
def shard_inputs(draw: st.DrawFn) -> tuple[list[str], dict[str, pl.DataFrame], list[int]]:
    """Generate pipeline-shaped per-prefix shard inputs plus a subject subset for the shard.

    Every value is drawn from its full type domain; ties on ``(subject_id, time)`` are forced by
    sampling rows from small per-example pools of those full-domain draws (a fresh pool each
    example — nothing is enumerable across examples).

    Returns:
        A tuple of (table prefixes in config order, prefix -> merged-blocks frame, shard subjects).
    """
    subject_pool = draw(
        st.lists(st.integers(min_value=-_I64, max_value=_I64 - 1), min_size=1, max_size=5, unique=True)
    )
    time_pool = draw(st.lists(_maybe_datetime, min_size=2, max_size=4, unique=True))
    code_pool = draw(st.lists(st.text(max_size=16), min_size=1, max_size=4, unique=True))

    prefixes = draw(st.lists(_identifiers, min_size=1, max_size=3, unique=True))

    frames: dict[str, pl.DataFrame] = {}
    for prefix in prefixes:
        comp_schema = draw(st.one_of(st.none(), _component_schema()))
        n_blocks = draw(st.integers(min_value=1, max_value=3))
        blocks = []
        for block_idx in range(n_blocks):
            n_rows = draw(st.integers(min_value=0, max_value=12))
            schema: dict[str, pl.DataType] = {
                "subject_id": pl.Int64,
                "time": pl.Datetime("us"),
                "code": pl.String,
                "numeric_value": pl.Float32,
                "text_value": pl.String,
                "source_block": pl.String,
            }
            data: dict[str, list] = {
                "subject_id": [draw(st.sampled_from(subject_pool)) for _ in range(n_rows)],
                "time": [draw(st.sampled_from(time_pool)) for _ in range(n_rows)],
                "code": [draw(st.sampled_from(code_pool)) for _ in range(n_rows)],
                "numeric_value": [draw(_maybe_f32) for _ in range(n_rows)],
                "text_value": [draw(_maybe_text) for _ in range(n_rows)],
                "source_block": [f"{prefix}/e{block_idx}"] * n_rows,
            }
            if comp_schema is not None:
                schema["code_components"] = pl.Struct(comp_schema)
                data["code_components"] = [
                    {
                        name: draw(_maybe_text if dtype == pl.String else _maybe_i64)
                        for name, dtype in comp_schema.items()
                    }
                    for _ in range(n_rows)
                ]
            # Mirror the upstream contract: each block is deduped over all columns before merge
            # ever sees it, and carries its own distinct ``source_block`` stamp.
            blocks.append(pl.DataFrame(data, schema=schema).unique(maintain_order=True))
        frames[prefix] = pl.concat(blocks, how="vertical")

    subjects = sorted(draw(st.sets(st.sampled_from(subject_pool), min_size=1)))
    return prefixes, frames, subjects


def _deterministic_merge_reference(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """The guaranteed deterministically ordered merge: what any settings must reproduce exactly.

    Diagonal-relaxed concat in prefix order, full-column ``unique(maintain_order=True)``, then a
    single-threaded stable sort — the most conservative execution of the merge semantics.
    """
    lf = pl.concat([f.lazy() for f in frames], how="diagonal_relaxed")
    lf = lf.unique(maintain_order=True)
    return lf.sort(by=["subject_id", "time"], maintain_order=True, multithreaded=False).collect()


def _check_merge_properties(prefixes: list[str], frames: dict[str, pl.DataFrame]) -> None:
    """Write the per-prefix frames to a shard dir, run the real merge, and assert all properties."""
    # A plain TemporaryDirectory rather than the ``tmp_path`` fixture: hypothesis reuses a
    # function-scoped fixture across every generated example, so a fresh directory per example
    # has to be made inline.
    with TemporaryDirectory() as tmpdir:
        sp_dir = Path(tmpdir)
        for prefix in prefixes:
            frames[prefix].write_parquet(sp_dir / f"{prefix}.parquet")

        out = merge_subdirs_and_sort(sp_dir, table_prefixes=prefixes, unique_by=None).collect()
        rerun = merge_subdirs_and_sort(sp_dir, table_prefixes=prefixes, unique_by=None).collect()
        out_star = merge_subdirs_and_sort(sp_dir, table_prefixes=prefixes, unique_by="*").collect()

    reference = _deterministic_merge_reference([frames[p] for p in prefixes])

    # EQUIVALENCE: the shipped settings reproduce the deterministic reference exactly, row order
    # included; and the "*" opt-in remains an identical no-op on pipeline-shaped inputs.
    assert_frame_equal(out, reference, check_row_order=True, check_exact=True)
    assert_frame_equal(out_star, reference, check_row_order=True, check_exact=True)

    # STABLE: merging the same on-disk inputs twice yields the identical frame, row order included.
    assert_frame_equal(out, rerun, check_row_order=True, check_exact=True)

    # SORTED: a stable re-sort of the output on the sort key is the identity, i.e. the output is
    # already non-decreasing on (subject_id, time) (with polars' nulls-first placement).
    key = out.select("subject_id", "time")
    assert_frame_equal(
        key,
        key.sort(by=["subject_id", "time"], maintain_order=True),
        check_row_order=True,
        check_exact=True,
    )

    # UNIQUE: no full-row duplicates survive the merge.
    assert out.unique().height == out.height

    # LOSSLESS: nothing is dropped — inputs are pre-deduped per block and blocks are distinct, so
    # the merge must preserve every input row.
    assert out.height == sum(frames[p].height for p in prefixes)


@given(inputs=shard_inputs())
def test_merge_properties(inputs: tuple[list[str], dict[str, pl.DataFrame], list[int]]) -> None:
    """Chain _filter_to_subjects -> merge_subdirs_and_sort on generated pipeline-shaped inputs."""
    prefixes, frames, subjects = inputs

    filtered: dict[str, pl.DataFrame] = {}
    for prefix in prefixes:
        table = TableConfig.parse(prefix, {"e": {"code": "X", "time": None}})
        filtered[prefix] = _filter_to_subjects(
            frames[prefix].lazy(), table=table, subjects=subjects
        ).collect()

    _check_merge_properties(prefixes, filtered)


_REGRESSION_SCHEMA = {
    "subject_id": pl.Int64,
    "time": pl.Datetime("us"),
    "code": pl.String,
    "numeric_value": pl.Float32,
    "text_value": pl.String,
    "source_block": pl.String,
}


def test_merge_properties_tie_heavy(tmp_path: Path) -> None:
    """Regression: identical (subject_id, time) keys across blocks and prefixes."""
    t = datetime(2020, 1, 1)

    def block(prefix: str, block_idx: int) -> pl.DataFrame:
        rows = {
            "subject_id": [1, 1, 1],
            "time": [t, t, t],
            "code": ["A", "B//1", "C"],
            "numeric_value": [1.0, None, None],
            "text_value": [None, "low", None],
            "source_block": [f"{prefix}/e{block_idx}"] * 3,
        }
        return pl.DataFrame(rows, schema=_REGRESSION_SCHEMA).unique(maintain_order=True)

    prefixes = ["labs", "vitals"]
    for prefix in prefixes:
        pl.concat([block(prefix, 0), block(prefix, 1)], how="vertical").write_parquet(
            tmp_path / f"{prefix}.parquet"
        )
    out = merge_subdirs_and_sort(tmp_path, table_prefixes=prefixes, unique_by=None).collect()
    reference = _deterministic_merge_reference([pl.read_parquet(tmp_path / f"{p}.parquet") for p in prefixes])
    assert_frame_equal(out, reference, check_row_order=True, check_exact=True)
    assert out.height == 12  # 2 prefixes x 2 blocks x 3 rows, nothing dropped


def test_merge_properties_empty_prefix_file(tmp_path: Path) -> None:
    """Regression: one prefix contributes a schema-only (zero-row) parquet file."""
    pl.DataFrame(schema=_REGRESSION_SCHEMA).write_parquet(tmp_path / "labs.parquet")
    pl.DataFrame(
        {
            "subject_id": [2],
            "time": [None],
            "code": ["C"],
            "numeric_value": [None],
            "text_value": ["high"],
            "source_block": ["patients/e0"],
        },
        schema=_REGRESSION_SCHEMA,
    ).write_parquet(tmp_path / "patients.parquet")

    out = merge_subdirs_and_sort(tmp_path, table_prefixes=["labs", "patients"], unique_by=None).collect()
    assert out.height == 1
    assert out["code"].to_list() == ["C"]


def test_merge_properties_single_subject_shard(tmp_path: Path) -> None:
    """Regression: every row belongs to one subject, with within-subject time ties."""
    t1 = datetime(2020, 1, 1)
    t2 = datetime(2021, 6, 1, 8)
    rows = {
        "subject_id": [3, 3, 3, 3],
        "time": [t1, t2, t1, None],
        "code": ["A", "B//1", "A", "C"],
        "numeric_value": [0.5, None, 0.5, None],
        "text_value": [None, None, None, "low"],
        "source_block": ["labs/e0", "labs/e0", "labs/e1", "labs/e1"],
    }
    pl.DataFrame(rows, schema=_REGRESSION_SCHEMA).unique(maintain_order=True).write_parquet(
        tmp_path / "labs.parquet"
    )

    out = merge_subdirs_and_sort(tmp_path, table_prefixes=["labs"], unique_by=None).collect()
    reference = _deterministic_merge_reference([pl.read_parquet(tmp_path / "labs.parquet")])
    assert_frame_equal(out, reference, check_row_order=True, check_exact=True)
    assert out.height == 4


def _check_write_path_properties(lf: pl.LazyFrame, tmpdir: str) -> None:
    """Assert the write-path contract for one stage plan.

    The sunk file must hold exactly what eager execution of the same plan produces — same rows,
    same row order — and sinking the same plan twice must produce byte-identical files (order
    determinism is load-bearing: merge's stable sort propagates it into final bytes).
    """
    eager = lf.collect()
    fp_a, fp_b = Path(tmpdir) / "a.parquet", Path(tmpdir) / "b.parquet"
    sink_df(lf, fp_a)
    sink_df(lf, fp_b)
    assert_frame_equal(pl.read_parquet(fp_a), eager, check_row_order=True, check_exact=True)
    assert fp_a.read_bytes() == fp_b.read_bytes(), "Repeated sinks of one plan must be byte-identical."


@given(inputs=shard_inputs())
def test_subject_sharded_sink_write_properties(
    inputs: tuple[list[str], dict[str, pl.DataFrame], list[int]],
) -> None:
    """The sink-based write reproduces eager execution exactly on generated stage plans."""
    prefixes, frames, subjects = inputs
    with TemporaryDirectory() as tmpdir:
        for prefix in prefixes:
            table = TableConfig.parse(prefix, {"e": {"code": "X", "time": None}})
            lf = _filter_to_subjects(frames[prefix].lazy(), table=table, subjects=subjects)
            _check_write_path_properties(lf, tmpdir)


def test_subject_sharded_sink_write_with_fanout_join(tmp_path: Path) -> None:
    """Regression: the full stage plan — scan, an order-pinned join whose right side has duplicate
    keys (fan-out), then the subject filter — sinks deterministically and equal to eager.

    Duplicate right keys are the case where an unordered streaming join is free to reorder; the
    order-pinned join in ``JoinConfig.apply`` is what makes the sunk output deterministic.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    pl.DataFrame(
        {
            "stay_id": [10, 10, 20, 20, 30],
            "patient_id": [1, 1, 2, 2, 3],
            "obs": ["a", "b", "c", "d", "e"],
        }
    ).write_parquet(raw / "events.parquet")
    # Duplicate join keys on the right: each left row fans out to two.
    pl.DataFrame(
        {"stay_id": [10, 10, 20, 20, 30, 30], "ward": ["W1", "W2", "W3", "W4", "W5", "W6"]}
    ).write_parquet(raw / "stays.parquet")

    table = TableConfig.parse(
        "events",
        {
            "_defaults": {"subject_id": "$patient_id"},
            "_table": {"join": {"stays": {"key": "stay_id", "cols": ["ward"]}}},
            "e": {"code": "X", "time": None},
        },
    )
    lf = _read_and_join([raw / "events.parquet"], table=table, input_dir=raw)
    lf = _filter_to_subjects(lf, table=table, subjects=[1, 2])
    _check_write_path_properties(lf, str(tmp_path))


def test_subject_sharded_sink_write_empty_shard(tmp_path: Path) -> None:
    """Regression: a shard whose subject filter matches nothing sinks a valid, zero-row,
    schema-correct parquet file."""
    frame = pl.DataFrame(
        {
            "subject_id": [1],
            "time": [None],
            "code": ["A"],
            "numeric_value": [1.0],
            "text_value": [None],
            "source_block": ["labs/e0"],
        },
        schema=_REGRESSION_SCHEMA,
    )
    table = TableConfig.parse("labs", {"e": {"code": "X", "time": None}})
    lf = _filter_to_subjects(frame.lazy(), table=table, subjects=[999])
    fp = tmp_path / "empty.parquet"
    sink_df(lf, fp)
    out = pl.read_parquet(fp)
    assert out.height == 0
    assert out.schema == frame.schema
