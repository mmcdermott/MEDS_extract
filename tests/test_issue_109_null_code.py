"""Demonstration / regression test for issue #109 — multi-column ``code`` null handling.

https://github.com/mmcdermott/MEDS_extract/issues/109

``EventConfig.extract`` drops rows whose ``code`` references a null source column by
checking only the **alphabetically-first** referenced column::

    # src/MEDS_extract/config.py
    first_col = sorted(self.code_source_columns)[0]
    df = df.filter(pl.col(first_col).is_not_null())

When a *later* code component is null, polars string-interpolation propagates the null
and the whole ``code`` evaluates to ``null`` — a value the MEDS schema forbids — yet the
row is written out anyway. The sibling *time* filter does this correctly, masking on
**all** referenced columns via ``pl.all_horizontal(*ts_filters)``.

The ``xfail(strict=True)`` test encodes the desired behaviour (no null codes in the
output). It fails today — demonstrating the bug — and flips to a passing regression test
the moment the filter is fixed, at which point ``strict=True`` flags the now-stale marker.
"""

from __future__ import annotations

import polars as pl
import pytest

from MEDS_extract.config import EventConfig


def _extract(code: str) -> pl.DataFrame:
    """Run a single event's extraction over a 3-row frame and return the output.

    Row layout, chosen to isolate the defect:
      * subject 1 — both components present (a kept, valid code).
      * subject 2 — the *trailing* component ``b`` is null (the bug: this leaks a null code).
      * subject 3 — the *leading* component ``a`` is null (correctly dropped today).
    """
    raw = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "a": ["LAB", "LAB", None],
            "b": ["glucose", None, "sodium"],
            "ts": ["2020-01-01", "2020-01-02", "2020-01-03"],
        }
    )
    ev = EventConfig.parse("e", {"code": code, "time": '$ts::"%Y-%m-%d"'})
    return ev.extract(raw.lazy(), "labs/e").collect()


def test_single_column_code_drops_null_rows():
    """Control: a single-column code correctly drops its null rows (no null codes).

    Here the sole referenced column *is* ``first_col``, so the existing single-column
    filter suffices. This passes before and after the #109 fix, isolating the defect to
    the multi-column case below.
    """
    out = _extract("$a")  # references only column `a`
    assert out["code"].null_count() == 0
    assert sorted(out["subject_id"].to_list()) == [1, 2]  # subject 3 (a=null) dropped


def test_multicolumn_code_currently_emits_a_null_code():
    """Documents the *undesired* behaviour concretely: a null code reaches the output.

    ``code = f"{$a}//{$b}"`` references [a, b]; ``first_col`` is ``a``. Subject 2 has
    ``a='LAB'`` (non-null) but ``b=null``; it survives the ``a``-only filter and its code
    interpolates to null. This assertion captures the bug as-is so the failure mode is
    unmistakable; it should be **deleted** together with the fix (the companion xfail
    test below then becomes the permanent regression guard).
    """
    out = _extract('f"{$a}//{$b}"')  # references [a, b]; sorted-first = a
    null_codes = out.filter(pl.col("code").is_null())
    assert null_codes.height == 1
    assert null_codes["subject_id"].to_list() == [2]


@pytest.mark.xfail(
    strict=True,
    reason="#109: the code null-filter only checks the alphabetically-first referenced "
    "column, so a null in a later component leaks a NULL `code` into the output.",
)
def test_multicolumn_code_drops_rows_with_any_null_component():
    """Desired behaviour: drop a row if **any** referenced code component is null.

    MEDS forbids a null ``code``, so the correct output here contains zero null codes and
    keeps only subject 1 (subjects 2 and 3 each have a null code component). Fix: mask on
    all code source columns, mirroring the time filter::

        df = df.filter(
            pl.all_horizontal(*[pl.col(c).is_not_null() for c in sorted(self.code_source_columns)])
        )
    """
    out = _extract('f"{$a}//{$b}"')
    assert out["code"].null_count() == 0
    assert sorted(out["subject_id"].to_list()) == [1]
