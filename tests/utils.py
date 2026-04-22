"""Shared test helpers.

The prior bespoke stage harnesses (``single_stage_tester``, ``multi_stage_tester``, etc.) that
lived here were superseded by ``StageExample`` auto-discovery via
``tests/test_stage_examples.py``. ``assert_df_equal`` is kept for the few in-process tests in
``tests/test_inprocess_pipeline.py`` that need a DataFrame comparison helper with ``List[str]``
column flattening and the "extra columns are OK" escape hatch.
"""

import polars as pl
from polars.testing import assert_frame_equal


def assert_df_equal(
    want: pl.DataFrame,
    got: pl.DataFrame,
    msg: str | None = None,
    allow_extra_columns: bool = False,
    **kwargs,
):
    try:
        update_exprs = {}
        for k, v in want.schema.items():
            assert k in got.schema, f"missing column {k}."
            if kwargs.get("check_dtypes", False):
                assert v == got.schema[k], f"column {k} has different types."
            if v == pl.List(pl.String) and got.schema[k] == pl.List(pl.String):
                update_exprs[k] = pl.col(k).list.join("||")
        if update_exprs:
            want_cols = want.columns
            got_cols = got.columns

            want = want.with_columns(**update_exprs).select(want_cols)
            got = got.with_columns(**update_exprs).select(got_cols)

        if allow_extra_columns:
            got = got.select(want.columns)
        else:
            extra_cols = set(got.columns) - set(want.columns)
            if extra_cols:
                raise AssertionError(f"got has extra columns not in want: {extra_cols}")

        assert_frame_equal(want, got, **kwargs)
    except AssertionError as e:
        pl.Config.set_tbl_rows(-1)
        raise AssertionError(f"{msg}:\nWant:\n{want}\nGot:\n{got}\n{e}") from e
