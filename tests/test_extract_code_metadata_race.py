"""Regression test for #51 — ``extract_code_metadata`` Polars race when ``N_WORKERS>1``.

Pre-fix, the reducer's polling loop in
``src/MEDS_extract/extract_code_metadata/extract_code_metadata.py`` used
``Path.exists()`` to decide when all map-phase outputs were ready. Because
``pl.DataFrame.write_parquet`` opens its destination with ``O_TRUNC`` (no
temp+rename), ``Path.exists()`` returns ``True`` as soon as another worker's
``write_parquet`` creates the path — even before the parquet's header + footer
bytes have been flushed. The reducer's downstream
``pl.scan_parquet(fp, glob=False)`` then crashes with::

    polars.exceptions.ComputeError: parquet: File out of specification:
    A parquet file must contain a header and footer with at least 12 bytes

This is a real, intermittent failure reported against the MIMIC-IV_MEDS pipeline
(see MIMIC-IV_MEDS PR 32 linked on the issue) — the same ETL running with
``N_WORKERS=1`` was unaffected because the single worker serialized map and reduce.

The fix replaces the polling check with
``MEDS_transforms.mapreduce.rwlock.is_complete_parquet_file`` — the same
parquet-completeness check the stage's own ``rwlock_wrap`` calls already use via
``default_file_checker`` for the "is this output already done?" decision.

Why this test is structured the way it is
-----------------------------------------

The race only occurs with two OS-level workers talking through a shared
filesystem. We can't deterministically schedule a subprocess's ``write_parquet``
to be "mid-write" at the exact moment another subprocess's polling loop probes.
Instead, the test:

1. Pre-creates one valid parquet (``complete_fp``) and one zero-byte placeholder
   (``partial_fp``) — the synthetic mid-write state.
2. Spawns a background thread that, after a 0.5s delay, atomically replaces
   ``partial_fp`` with a real parquet — modelling the racing worker's flush.
3. Calls the production polling helper :func:`wait_for_complete_parquets`
   directly, then reads both parquets the same way the reducer does.

The thread-based fixture makes the outcome deterministic:

- **Pre-fix** (polling used ``Path.exists()``): the loop returns immediately
  because both paths exist, ``pl.scan_parquet(partial_fp).collect()`` fires
  before the writer flushes, and the test fails with ``ComputeError``.
- **Post-fix** (polling uses ``is_complete_parquet_file``): the loop blocks
  past the zero-byte stage; the writer flushes; the reducer reads both rows.

The test imports the production helper instead of inlining a copy of the
polling logic, so a future regression that re-introduces ``Path.exists()`` in
the helper fails this test in isolation rather than only manifesting in
the multi-worker integration test.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import polars as pl
import pytest

from MEDS_extract.extract_code_metadata.extract_code_metadata import wait_for_complete_parquets

if TYPE_CHECKING:
    from pathlib import Path


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

    Pre-fix this test failed via the ``ComputeError`` handler. Post-fix the
    polling helper waits for the writer thread to flush, ``scan_parquet``
    succeeds, and both rows are visible.
    """
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

    # Polling time matches ``_make_cfg``'s default in
    # tests/test_inprocess_pipeline.py (0.1s) so the loop's cadence is
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
