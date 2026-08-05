"""Property tests pinning the invariants behind the #241 merge-stage performance changes.

Issue #241 made two changes to ``merge_subdirs_and_sort``:

1. the final sort became multithreaded (keeping ``maintain_order=True``, so it stays a stable —
   and therefore deterministic — sort), and
2. the shipped default ``unique_by`` changed from ``"*"`` to ``None``, skipping the post-merge
   dedup entirely.

Both are only safe because of an invariant the upstream stages guarantee: every event block ends
with a full-column ``.unique(maintain_order=True)`` (``EventConfig.extract``) and stamps its rows
with a distinct per-block ``source_block`` value, so a merged shard can never contain a full-row
duplicate. The tests here generate "pipeline-shaped" shard inputs that honor that invariant
(including heavy (subject, time) ties across blocks, where ordering bugs hide), chain them through
the real df->df stage helpers (``_filter_to_subjects`` then ``merge_subdirs_and_sort``), and assert:

- EQUIVALENCE: the new settings reproduce the pre-#241 behavior (full-column unique + stable
  single-threaded sort) exactly, including row order;
- SORTED: the output is non-decreasing on ``(subject_id, time)``;
- UNIQUE: the output contains no full-row duplicates;
- LOSSLESS: no row is dropped relative to the (post-filter) inputs.

The same suite also pins the #240 change to ``convert_to_subject_sharded``: its write side moved
from MEDS-transforms' eager ``write_df`` (collect the whole joined table in memory, then write) to
:func:`sink_df` (execute the plan on polars' streaming engine, peak memory O(shard)). That swap is
only safe because the join in ``JoinConfig.apply`` is order-pinned (``maintain_order="left_right"``)
— the write-path properties assert the sunk output equals the eager path's exactly, row order
included, and that repeated sinks are byte-identical (order determinism leaks into final MEDS bytes
through merge's stable sort, so it must hold at the file level, not just as multisets).

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

SUBJECT_POOL = [1, 2, 3, 4, 5]

# Small pools (plus None for static events) so ties on (subject_id, time) are common — ties across
# blocks are exactly where instability in the merge/sort would surface.
TIME_POOL = [
    None,
    datetime(2020, 1, 1, 0, 0, 0),
    datetime(2020, 1, 1, 12, 30, 0),
    datetime(2021, 6, 1, 8, 0, 0),
]
CODE_POOL = ["A", "B//1", "C"]
NUMERIC_POOL = [None, 0.0, 1.5, -2.25]
TEXT_POOL = [None, "low", "high"]

PREFIX_POOL = ["labs", "patients", "vitals"]


def _block_frame(prefix: str, block_idx: int, rows: list[tuple], with_components: bool) -> pl.DataFrame:
    """Build one event block's frame, mirroring ``EventConfig.extract``'s output contract.

    Every row gets the block's distinct ``source_block`` stamp and the block is deduped over all
    columns with ``.unique(maintain_order=True)`` — the two upstream guarantees that make
    ``unique_by=None`` safe downstream.
    """
    schema: dict[str, pl.DataType] = {
        "subject_id": pl.Int64,
        "time": pl.Datetime("us"),
        "code": pl.String,
        "numeric_value": pl.Float32,
        "text_value": pl.String,
        "source_block": pl.String,
    }
    data = {
        "subject_id": [r[0] for r in rows],
        "time": [r[1] for r in rows],
        "code": [r[2] for r in rows],
        "numeric_value": [r[3] for r in rows],
        "text_value": [r[4] for r in rows],
        "source_block": [f"{prefix}/e{block_idx}" for _ in rows],
    }
    if with_components:
        # Per-prefix struct fields, so merging exercises diagonal_relaxed's struct unification.
        schema["code_components"] = pl.Struct({f"{prefix}_component": pl.String})
        data["code_components"] = [{f"{prefix}_component": r[2]} for r in rows]
    return pl.DataFrame(data, schema=schema).unique(maintain_order=True)


@st.composite
def shard_inputs(draw) -> tuple[list[str], dict[str, pl.DataFrame], list[int]]:
    """Generate pipeline-shaped per-prefix shard inputs plus a subject subset for the shard.

    Returns:
        A tuple of (table prefixes in config order, prefix -> merged-blocks frame, shard subjects).
    """
    n_prefixes = draw(st.integers(min_value=1, max_value=3))
    prefixes = PREFIX_POOL[:n_prefixes]

    row = st.tuples(
        st.sampled_from(SUBJECT_POOL),
        st.sampled_from(TIME_POOL),
        st.sampled_from(CODE_POOL),
        st.sampled_from(NUMERIC_POOL),
        st.sampled_from(TEXT_POOL),
    )

    frames: dict[str, pl.DataFrame] = {}
    for prefix in prefixes:
        with_components = draw(st.booleans())
        n_blocks = draw(st.integers(min_value=1, max_value=3))
        blocks = [
            _block_frame(prefix, block_idx, draw(st.lists(row, max_size=12)), with_components)
            for block_idx in range(n_blocks)
        ]
        frames[prefix] = pl.concat(blocks, how="vertical")

    subjects = sorted(draw(st.sets(st.sampled_from(SUBJECT_POOL), min_size=1)))
    return prefixes, frames, subjects


def _pre_241_reference(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Replicate the pre-#241 merge behavior as an oracle.

    Diagonal-relaxed concat in prefix order, full-column ``unique(maintain_order=True)``
    (the old ``unique_by: "*"`` default), then the old single-threaded stable sort.
    """
    lf = pl.concat([f.lazy() for f in frames], how="diagonal_relaxed")
    lf = lf.unique(maintain_order=True)
    return lf.sort(by=["subject_id", "time"], maintain_order=True, multithreaded=False).collect()


def _check_merge_properties(prefixes: list[str], frames: dict[str, pl.DataFrame]) -> None:
    """Write the per-prefix frames to a shard dir, run the real merge, and assert all properties."""
    with TemporaryDirectory() as tmpdir:
        sp_dir = Path(tmpdir)
        for prefix in prefixes:
            frames[prefix].write_parquet(sp_dir / f"{prefix}.parquet")

        out = merge_subdirs_and_sort(sp_dir, table_prefixes=prefixes, unique_by=None).collect()
        out_star = merge_subdirs_and_sort(sp_dir, table_prefixes=prefixes, unique_by="*").collect()

    reference = _pre_241_reference([frames[p] for p in prefixes])

    # EQUIVALENCE: the new defaults (no dedup, multithreaded stable sort) reproduce the old
    # behavior exactly, row order included; and the "*" opt-in remains a byte-identical no-op.
    assert_frame_equal(out, reference, check_row_order=True)
    assert_frame_equal(out_star, reference, check_row_order=True)

    # SORTED: a stable re-sort of the output on the sort key is the identity, i.e. the output is
    # already non-decreasing on (subject_id, time) (with polars' nulls-first placement).
    key = out.select("subject_id", "time")
    assert_frame_equal(key, key.sort(by=["subject_id", "time"], maintain_order=True), check_row_order=True)

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


def test_merge_properties_tie_heavy() -> None:
    """Regression: identical (subject_id, time) keys across blocks and prefixes."""
    t = datetime(2020, 1, 1)
    rows = [(1, t, "A", 1.0, None), (1, t, "B//1", None, "low"), (1, t, "C", None, None)]
    frames = {
        "labs": pl.concat(
            [
                _block_frame("labs", 0, rows, with_components=True),
                _block_frame("labs", 1, rows, with_components=True),
            ],
            how="vertical",
        ),
        "vitals": _block_frame("vitals", 0, rows, with_components=False),
    }
    _check_merge_properties(["labs", "vitals"], frames)


def test_merge_properties_empty_prefix_file() -> None:
    """Regression: one prefix contributes a schema-only (zero-row) parquet file."""
    frames = {
        "labs": _block_frame("labs", 0, [], with_components=False),
        "patients": _block_frame("patients", 0, [(2, None, "C", None, "high")], with_components=True),
    }
    _check_merge_properties(["labs", "patients"], frames)


def _check_write_path_properties(lf: pl.LazyFrame, tmpdir: str) -> None:
    """Assert the #240 write-path contract for one stage plan.

    The sunk file must hold exactly what the eager path would have written — same rows, same
    row order — and sinking the same plan twice must produce byte-identical files (order
    determinism is load-bearing: merge's stable sort propagates it into final MEDS bytes).
    """
    eager = lf.collect()
    fp_a, fp_b = Path(tmpdir) / "a.parquet", Path(tmpdir) / "b.parquet"
    sink_df(lf, fp_a)
    sink_df(lf, fp_b)
    assert_frame_equal(pl.read_parquet(fp_a), eager, check_row_order=True)
    assert fp_a.read_bytes() == fp_b.read_bytes(), "Repeated sinks of one plan must be byte-identical."


@given(inputs=shard_inputs())
def test_subject_sharded_sink_write_properties(
    inputs: tuple[list[str], dict[str, pl.DataFrame], list[int]],
) -> None:
    """#240: the sink-based write reproduces the eager write exactly on generated stage plans."""
    prefixes, frames, subjects = inputs
    with TemporaryDirectory() as tmpdir:
        for prefix in prefixes:
            table = TableConfig.parse(prefix, {"e": {"code": "X", "time": None}})
            lf = _filter_to_subjects(frames[prefix].lazy(), table=table, subjects=subjects)
            _check_write_path_properties(lf, tmpdir)


def test_subject_sharded_sink_write_with_fanout_join() -> None:
    """Regression (#240): the full stage plan — scan, an order-pinned join whose right side has duplicate keys
    (fan-out), then the subject filter — sinks deterministically and equal to eager.

    The duplicate right keys are the case where an unordered streaming join is free to reorder;
    ``JoinConfig.apply``'s ``maintain_order="left_right"`` is what pins it.
    """
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        raw = root / "raw"
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
        _check_write_path_properties(lf, tmpdir)


def test_subject_sharded_sink_write_empty_shard() -> None:
    """Regression (#240): a shard whose subject filter matches nothing sinks a valid, zero-row, schema-correct
    parquet file (polars' empty-sink bug was fixed pre-1.38; pin it stays fixed)."""
    frame = _block_frame("labs", 0, [(1, None, "A", 1.0, None)], with_components=True)
    table = TableConfig.parse("labs", {"e": {"code": "X", "time": None}})
    lf = _filter_to_subjects(frame.lazy(), table=table, subjects=[999])
    with TemporaryDirectory() as tmpdir:
        fp = Path(tmpdir) / "empty.parquet"
        sink_df(lf, fp)
        out = pl.read_parquet(fp)
        assert out.height == 0
        assert out.schema == frame.schema


def test_merge_properties_single_subject_shard() -> None:
    """Regression: every row belongs to one subject, with within-subject time ties."""
    t1 = datetime(2020, 1, 1)
    t2 = datetime(2021, 6, 1, 8)
    frames = {
        "labs": pl.concat(
            [
                _block_frame("labs", 0, [(3, t1, "A", 0.5, None), (3, t2, "B//1", None, None)], True),
                _block_frame("labs", 1, [(3, t1, "A", 0.5, None), (3, None, "C", None, "low")], True),
            ],
            how="vertical",
        ),
    }
    _check_merge_properties(["labs"], frames)
