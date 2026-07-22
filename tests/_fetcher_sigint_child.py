"""Child-process script for :func:`tests.test_download.test_download_all_sigint_cancels_queued_work`.

Not a test module. The ``_`` prefix keeps ``pytest --collect-only`` from matching it as a
test file; the ``if __name__ == "__main__"`` guard below keeps it safe against the
``--doctest-modules`` addopts pass, which would otherwise execute the body at import time.

Why a subprocess test at all: we need real SIGINT semantics — sending a signal to a
live :meth:`~MEDS_extract.download.source.Source.download_all` call and observing exit
behavior. pytest itself catches ``KeyboardInterrupt`` and stops the session, so there's
no way to observe "does the download exit quickly?" from inside a pytest worker. A
subprocess running ``Source.download_all`` with a stub :class:`Source` is the cleanest
isolation.

The parent test asserts on the count of files written to ``sys.argv[1]`` rather than
on wall-clock elapsed time — CI subprocess startup adds several seconds of variance
that makes timing assertions flaky, while the file-count signal is deterministic.
With ``shutdown(wait=False, cancel_futures=True)`` only the in-flight batch
(``_CONCURRENCY`` + a small race margin) completes before ``KeyboardInterrupt``
propagates out; if that ever regressed to a ``wait=True`` shutdown, every submitted
future would drain first and all ``_N_FILES`` files would end up on disk.

Usage: ``python tests/_fetcher_sigint_child.py <dest_dir>`` — exits 0 on success (i.e.
KeyboardInterrupt was caught cleanly), 99 on "download_all completed despite SIGINT"
(the unexpected success that would indicate a regression).
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from MEDS_extract.download import RemoteFile, Source

# 100 files, 0.2 s per-file sleep, concurrency=2 → serial drain ~= 10 s. SIGINT fires
# at 0.15 s, long before all 100 submissions could complete.
_N_FILES = 100
_PER_FILE_SLEEP_S = 0.2
_CONCURRENCY = 2
_SIGINT_AT_S = 0.15


class _SlowSource(Source):
    def _list_files(self):
        return [RemoteFile(f"file_{i}.txt", "") for i in range(_N_FILES)]

    def _pull(self, source_path, target):
        time.sleep(_PER_FILE_SLEEP_S)
        target.write_text("ok", encoding="utf-8")


def _kill_self_after_delay() -> None:
    time.sleep(_SIGINT_AT_S)
    os.kill(os.getpid(), signal.SIGINT)


def main() -> int:
    dest_dir = Path(sys.argv[1])
    threading.Thread(target=_kill_self_after_delay, daemon=True).start()
    # Shut down with ``cancel_futures=True`` on SIGINT so queued submissions die
    # immediately. A bare ``with ThreadPoolExecutor`` block would call
    # ``shutdown(wait=True)`` on exit and block ``Ctrl+C`` until every queued
    # future drains, which the parent test would observe as the regression.
    pool = ThreadPoolExecutor(max_workers=_CONCURRENCY)
    try:
        _SlowSource().download_all(dest_dir, pool=pool)
    except KeyboardInterrupt:
        return 0
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return 99  # download_all completed despite SIGINT — regression.


if __name__ == "__main__":
    sys.exit(main())
