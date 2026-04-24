"""Pytest configuration: doctest namespace + download-extra skip logic.

The doctest namespace fixture pre-populates common imports (``json``, ``pl``, ``datetime``,
``tempfile``, ``Path``) so every doctest in the repo can use them without ``>>> import ...``
lines cluttering the example. ``yaml_to_disk`` (``yaml_disk``) and ``pretty-print-directory``
(``print_directory``) auto-register via their own pytest plugins once installed.

The ``collect_ignore_glob`` block skips the download layer (+ its doctests) when the
``download`` extra (``httpx``, ``tenacity``) isn't installed. The download module raises
``ImportError`` at import time when the extras are missing, which is the right library
behavior but trips up pytest's ``--doctest-modules`` collector that imports every module
unconditionally. The sibling ``run_tests_download`` CI job installs the extra and runs
only the download tests; see ``.github/workflows/tests.yaml``.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest


@pytest.fixture(scope="session", autouse=True)
def _setup_doctest_namespace(doctest_namespace: dict[str, Any]):
    doctest_namespace.update(
        {
            "json": json,
            "pl": pl,
            "datetime": datetime,
            "tempfile": tempfile,
            "Path": Path,
        }
    )


collect_ignore_glob: list[str] = []

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
