"""Pytest configuration: doctest namespace + download-extra skip logic.

The doctest namespace fixture pre-populates common imports (``json``, ``pl``, ``datetime``,
``tempfile``, ``Path``) so every doctest in the repo can use them without ``>>> import ...``
lines cluttering the example. ``yaml_to_disk`` (``yaml_disk``) and ``pretty-print-directory``
(``print_directory``) auto-register via their own pytest plugins once installed.

The ``collect_ignore_glob`` block skips the HTTP-backed download modules (+ their
doctests) when the ``download`` extra (``httpx``, ``tenacity``) isn't installed. Those
two modules raise ``ImportError`` at import time when the extras are missing, which is
the right library behavior but trips up pytest's ``--doctest-modules`` collector that
imports every module unconditionally; ``spec.py`` imports fine but its doctests
construct http/physionet sources, which trips the same ImportError at example time.
The rest of the download layer (``source.py``, ``cli.py``, the fsspec backend, and
``tests/test_download_fsspec.py``) stays collected, so the no-extras job exercises
the fsspec-only download path end-to-end. The sibling ``run_tests_download`` CI job
installs the extra and runs the full download surface; see
``.github/workflows/tests.yaml``.
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
            "src/MEDS_extract/download/backends/http.py",
            "src/MEDS_extract/download/backends/physionet.py",
            # Imports lazily, but its doctests CONSTRUCT http/physionet sources,
            # which triggers the extras ImportError at example-execution time.
            "src/MEDS_extract/download/spec.py",
            # Imports httpx at module top for its MockTransport-based tests.
            "tests/test_download.py",
        ]
    )
