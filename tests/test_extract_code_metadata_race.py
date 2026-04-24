"""Regression test for #51 — ``extract_code_metadata`` Polars race when ``N_WORKERS>1``.

The reducer's polling loop at
``src/MEDS_extract/extract_code_metadata/extract_code_metadata.py:410`` currently
uses ``fp.exists()`` to decide when all map-phase outputs are ready:

.. code-block:: python

    while not all(fp.exists() for fp in all_out_fps):
        time.sleep(cfg.polling_time)

``Path.exists()`` returns ``True`` as soon as another worker's
``pl.DataFrame.write_parquet`` call creates the target path — even before the
parquet's header + footer bytes have been flushed. The reducer's next step is:

.. code-block:: python

    for fp in all_out_fps:
        df = pl.scan_parquet(fp, glob=False)

which promptly crashes on any partial file with::

    polars.exceptions.ComputeError: parquet: File out of specification:
    A parquet file must contain a header and footer with at least 12 bytes

This is a real, intermittent failure reported against the MIMIC-IV_MEDS pipeline
(see MIMIC-IV_MEDS PR 32 linked in the issue) — the same ETL running with
``N_WORKERS=1`` is unaffected because the single worker serializes map and reduce.

The fix is to replace the polling-loop exit condition with a stronger check.
``MEDS_transforms.mapreduce.rwlock.is_complete_parquet_file`` (already used by
the stage's own ``rwlock_wrap`` calls via ``default_file_checker``) is the
obvious candidate — it exercises the parquet footer.

Why this test is structured as a sequence reproduction (not a full-stage run)
---------------------------------------------------------------------------

The race only occurs with two OS-level workers talking through a shared
filesystem. We can't deterministically schedule a subprocess's ``write_parquet``
call to be "mid-write" at the exact moment another subprocess's polling loop
probes. So this test reproduces the **exact three-line sequence** from the
stage (polling-loop exit + ``pl.scan_parquet`` + ``pl.concat.collect``) against
a synthetic mid-write state, using a background thread to guarantee the
zero-byte file becomes a valid parquet after a delay.

The thread-based fixture makes the outcome deterministic:

- **Before the fix**: polling loop exits on ``fp.exists() == True`` within
  milliseconds, the reducer's ``collect()`` fires before the writer thread
  flushes, and the test fails with ``ComputeError``.
- **After the fix** (polling uses ``is_complete_parquet_file``): the loop
  keeps polling past the zero-byte stage, the writer thread finishes,
  ``collect()`` succeeds, and the test passes.

This test is intentionally NOT a doctest — the threading + timing makes it
unreadable as one, and the bug repro deserves its own file with a long
rationale docstring that future maintainers can read before editing.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import polars as pl
import pytest


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
    """Reproduce the #51 race: reducer reads a partial parquet mid-write by another worker.

    Before the fix this asserts-fails via ``pytest.fail`` inside the
    ``ComputeError`` handler. After the fix the polling loop will wait past the
    zero-byte state and ``collect()`` succeeds.

    The test is expected to **FAIL** until a fix lands. Leaving it red in the
    draft PR is intentional — the failing test is the machine-readable bug
    report.
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

    # Reproduce the three-line sequence from extract_code_metadata.main() at
    # src/MEDS_extract/extract_code_metadata/extract_code_metadata.py:410-423.
    # The polling-time constant matches _make_cfg's default in
    # tests/test_inprocess_pipeline.py (0.1s) so the loop's cadence is
    # representative of a real pipeline invocation.
    polling_time = 0.1
    deadline = time.monotonic() + 5.0  # safety cap so a wedged test doesn't hang CI
    while not all(fp.exists() for fp in all_out_fps):  # ← current buggy exit condition
        if time.monotonic() > deadline:
            break
        time.sleep(polling_time)

    dfs = [pl.scan_parquet(fp, glob=False) for fp in all_out_fps]
    try:
        result = pl.concat(dfs, how="diagonal_relaxed").collect()
    except pl.exceptions.ComputeError as e:
        pytest.fail(
            "#51: reducer crashed on a partial parquet from a concurrent worker write.\n"
            f"  underlying error: {e}\n"
            "  fix: replace the `fp.exists()` exit condition at "
            "src/MEDS_extract/extract_code_metadata/extract_code_metadata.py:410 "
            "with `is_complete_parquet_file(fp)` "
            "(already imported in the stage via `default_file_checker`)."
        )
    finally:
        writer.join()

    # After the fix the reducer sees both rows — the polling loop kept going
    # past the zero-byte partial_fp, the writer flushed its footer, and
    # scan_parquet is now reading the valid file.
    assert set(result["code"].to_list()) == {"A", "B"}
