"""Child-process script for :func:`tests.test_download.test_fetcher_sigint_cancels_queued_work`.

Not a test module. The ``_`` prefix keeps ``pytest --collect-only`` from matching it as a
test file; the ``if __name__ == "__main__"`` guard below keeps it safe against the
``--doctest-modules`` addopts pass, which would otherwise execute the body at import time.

Why a subprocess test at all: we need real SIGINT semantics — sending a signal to a
live :class:`~MEDS_extract.download.fetcher.Fetcher` and observing exit behavior.
pytest itself catches ``KeyboardInterrupt`` and stops the session, so there's no way
to observe "does the Fetcher exit quickly?" from inside a pytest worker. A subprocess
running a ``Fetcher.fetch_all`` with a stub ``Source`` is the cleanest isolation.

The parent test asserts on the count of files written to ``sys.argv[1]`` rather than
on wall-clock elapsed time — CI subprocess startup adds several seconds of variance
that makes timing assertions flaky, while the file-count signal is deterministic:

* With the ``shutdown(wait=False, cancel_futures=True)`` fix, only the in-flight
  batch (~``max_concurrency`` files + a small race margin) completes before
  ``KeyboardInterrupt`` propagates out.
* Without the fix, ``pool.__exit__`` drains every submitted future, so ALL
  ``n_files`` files are on disk by the time ``KeyboardInterrupt`` re-raises.

Usage: ``python tests/_fetcher_sigint_child.py <dest_dir>`` — exits 0 on success (i.e.
KeyboardInterrupt was caught cleanly), 99 on "fetch_all completed despite SIGINT"
(the unexpected success that would indicate the fix regressed).
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path

from MEDS_extract.download import Fetcher
from MEDS_extract.download.source import RemoteFile, Source

# 100 files, 0.2 s per-file sleep, concurrency=2 → serial drain ~= 10 s. SIGINT fires
# at 0.15 s, long before all 100 submissions could complete serially.
_N_FILES = 100
_PER_FILE_SLEEP_S = 0.2
_CONCURRENCY = 2
_SIGINT_AT_S = 0.15


class _SlowSource(Source):
    def list_files(self):
        return [RemoteFile(f"file_{i}.txt") for i in range(_N_FILES)]

    def _fetch(self, remote, dest):
        time.sleep(_PER_FILE_SLEEP_S)
        dest.write_text("ok", encoding="utf-8")


def _kill_self_after_delay() -> None:
    time.sleep(_SIGINT_AT_S)
    os.kill(os.getpid(), signal.SIGINT)


def main() -> int:
    dest_dir = Path(sys.argv[1])
    threading.Thread(target=_kill_self_after_delay, daemon=True).start()
    try:
        Fetcher(dest_dir, max_concurrency=_CONCURRENCY).fetch_all(_SlowSource())
    except KeyboardInterrupt:
        return 0
    return 99  # fetch_all completed despite SIGINT — the fix regressed.


if __name__ == "__main__":
    sys.exit(main())
