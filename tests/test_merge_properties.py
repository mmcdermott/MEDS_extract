"""Property tests pinning the merge-stage invariants as of #241 (multithreaded sort) and #254.

Issue #241 made the merge's final sort multithreaded (keeping ``maintain_order=True``, so it stays
a stable — and therefore deterministic — sort). Issue #254 then changed what the merge *contains*:

1. the internal ``code_components`` struct is dropped per input scan, before the diagonal concat,
   so the per-table structs are never unified into a field-union superstruct (the memory blow-up
   #254 fixed); and
2. the shipped default ``unique_by`` is ``"*"``: with the components gone, two source observations
   that differed *only* in their raw components collapse into byte-identical rows, and
   identical-looking rows in the merged output must not be duplicated. Full-row uniqueness is a
   semantic guarantee of the merged output (this supersedes #241's ``unique_by: None`` default,
   which was only a provable no-op while the merge still carried ``code_components``).

Upstream, every event block still ends with a full-column ``.unique(maintain_order=True)``
(``EventConfig.extract``) and stamps its rows with a distinct per-block ``source_block`` value.
The tests here generate "pipeline-shaped" shard inputs honoring that contract (including heavy
(subject, time) ties across blocks, where ordering bugs hide), chain them through the real df->df
stage helpers (``_filter_to_subjects`` then ``merge_subdirs_and_sort``), and assert:

- DROPPED: ``code_components`` never appears in the merge output, even when inputs carry it;
- EQUIVALENCE: the shipped settings ("*" dedup + multithreaded stable sort) reproduce the oracle
  (concat -> drop components -> full-column unique -> stable single-threaded sort) exactly,
  including row order — and ``unique_by=None`` reproduces the same plan minus the unique;
- SORTED: the output is non-decreasing on ``(subject_id, time)``;
- UNIQUE: the output contains no full-row duplicates (the "*" contract);
- LOSSLESS: the output row count equals the input row count minus rows collapsed by the
  post-drop dedup — for the generated inputs, whose components are a pure function of the code
  value, nothing collapses and plain equality holds.

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
    columns (components included) with ``.unique(maintain_order=True)``.
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


def _reference_merge(frames: list[pl.DataFrame], unique: bool) -> pl.DataFrame:
    """Replicate the post-#254 merge behavior as an oracle.

    Diagonal-relaxed concat in prefix order with ``code_components`` dropped per input (#254),
    full-column ``unique(maintain_order=True)`` when ``unique`` (the shipped ``unique_by: "*"``
    default), then a single-threaded stable sort (the multithreaded real sort must match it —
    the #241 invariant).
    """
    lf = pl.concat([f.lazy().drop("code_components", strict=False) for f in frames], how="diagonal_relaxed")
    if unique:
        lf = lf.unique(maintain_order=True)
    return lf.sort(by=["subject_id", "time"], maintain_order=True, multithreaded=False).collect()


def _check_merge_properties(prefixes: list[str], frames: dict[str, pl.DataFrame]) -> None:
    """Write the per-prefix frames to a shard dir, run the real merge, and assert all properties."""
    with TemporaryDirectory() as tmpdir:
        sp_dir = Path(tmpdir)
        for prefix in prefixes:
            frames[prefix].write_parquet(sp_dir / f"{prefix}.parquet")

        out = merge_subdirs_and_sort(sp_dir, table_prefixes=prefixes, unique_by="*").collect()
        out_none = merge_subdirs_and_sort(sp_dir, table_prefixes=prefixes, unique_by=None).collect()

    # DROPPED: the internal components struct never survives the merge, whether or not any
    # input carried it (#254).
    assert "code_components" not in out.columns
    assert "code_components" not in out_none.columns

    # EQUIVALENCE: the shipped settings ("*" dedup, multithreaded stable sort) reproduce the
    # oracle (drop -> unique -> single-threaded stable sort) exactly, row order included; and
    # ``unique_by=None`` reproduces the same plan minus the unique (raw concat semantics).
    reference = _reference_merge([frames[p] for p in prefixes], unique=True)
    assert_frame_equal(out, reference, check_row_order=True)
    assert_frame_equal(
        out_none, _reference_merge([frames[p] for p in prefixes], unique=False), check_row_order=True
    )

    # SORTED: a stable re-sort of the output on the sort key is the identity, i.e. the output is
    # already non-decreasing on (subject_id, time) (with polars' nulls-first placement).
    key = out.select("subject_id", "time")
    assert_frame_equal(key, key.sort(by=["subject_id", "time"], maintain_order=True), check_row_order=True)

    # UNIQUE: no full-row duplicates survive the merge — with the "*" default this is the
    # contract of the merged output, asserted here directly rather than via the oracle.
    assert out.unique().height == out.height

    # LOSSLESS (minus collapse): the output holds every input row except those the post-drop
    # dedup collapsed. The generators build ``code_components`` as a pure function of the code
    # value, so no generated pair can differ only in components and nothing collapses — plain
    # equality holds here. The collapse case is pinned deterministically in
    # ``test_merge_dedups_rows_that_differed_only_in_components``.
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


def test_merge_dedups_rows_that_differed_only_in_components() -> None:
    """Regression (#254): rows that differed ONLY in ``code_components`` collapse to one output row.

    Two raw observations can interpolate to the same code string from different raw component
    values (e.g. via ``?? 'UNK'`` coalesces), so upstream's full-column dedup keeps both — they
    differ in the struct. Once the merge drops the struct they are byte-identical, and the
    maintainer's rule is that identical-looking rows in the merged output must not be duplicated:
    the shipped ``unique_by: "*"`` default collapses the same-block pair to ONE row (2 in ->
    1 out). Rows from DIFFERENT blocks still differ on ``source_block`` after the drop, so those
    are both kept. ``unique_by=None`` remains the escape hatch for raw concat semantics that
    keep the collapsed pair.
    """
    t = datetime(2020, 1, 1)
    schema: dict[str, pl.DataType] = {
        "subject_id": pl.Int64,
        "time": pl.Datetime("us"),
        "code": pl.String,
        "numeric_value": pl.Float32,
        "source_block": pl.String,
        "code_components": pl.Struct({"units": pl.String}),
    }
    # One block ("labs/e0"): two rows identical except for the components struct (a raw null
    # unit vs. an explicit "UNK", both rendering the same code string). Upstream's full-column
    # unique keeps both. A second block ("labs/e1"): a row identical to the first pair except
    # for its source_block stamp.
    labs = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [t, t, t],
            "code": ["GLU//UNK", "GLU//UNK", "GLU//UNK"],
            "numeric_value": [1.0, 1.0, 1.0],
            "source_block": ["labs/e0", "labs/e0", "labs/e1"],
            "code_components": [{"units": None}, {"units": "UNK"}, {"units": "UNK"}],
        },
        schema=schema,
    )
    assert labs.unique().height == 3, "the fixture must survive upstream's full-column dedup intact"

    with TemporaryDirectory() as tmpdir:
        sp_dir = Path(tmpdir)
        labs.write_parquet(sp_dir / "labs.parquet")
        out = merge_subdirs_and_sort(sp_dir, table_prefixes=["labs"], unique_by="*").collect()
        out_none = merge_subdirs_and_sort(sp_dir, table_prefixes=["labs"], unique_by=None).collect()

    assert "code_components" not in out.columns
    # The same-block pair collapses to one row; the cross-block row survives on source_block.
    assert out.height == 2
    assert sorted(out["source_block"].to_list()) == ["labs/e0", "labs/e1"]
    # Raw concat semantics (unique_by=None) keep all three rows, collapsed pair included.
    assert out_none.height == 3


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
