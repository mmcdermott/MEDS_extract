"""Pytest configuration: skip the download layer + its doctests when the ``download`` extra
(``httpx``, ``tenacity``) isn't installed.

The ``download`` module raises ``ImportError`` at import time if its extras are missing. That's
the right behavior for the library (users get a clear install hint), but it trips up pytest's
``--doctest-modules`` collector, which imports every module unconditionally. This hook skips
the relevant files when the extras aren't present.
"""

from __future__ import annotations

collect_ignore_glob = []

try:
    import httpx  # noqa: F401
    import tenacity  # noqa: F401
except ImportError:
    collect_ignore_glob.extend(
        [
            "src/MEDS_extract/download/*.py",
            "src/MEDS_extract/download/**/*.py",
            "tests/test_download.py",
        ]
    )
