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

import importlib
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


@pytest.fixture
def fake_pipeline_registry(tmp_path, monkeypatch):
    """A synthetic installed dataset package plus ``MEDS_extract.pipelines`` registrations.

    Installs an importable ``fake_ds_pkg`` (one bundled ``event_configs.yaml`` MESSY
    spec, ``etl:``-free with ``sources.dataset_version: "0.9"``) and monkeypatches
    the entry-point lookup ``MEDS_extract.config`` uses so three registrations are
    visible, mimicking a pip-installed dataset package (distribution version 1.2.3):

    - ``Fake-DS = "fake_ds_pkg:event_configs.yaml"`` — well-formed;
    - ``Bare-Mod = "fake_ds_pkg"`` — malformed (no ``:filename``);
    - ``Missing-Res = "fake_ds_pkg:nope.yaml"`` — names a resource that isn't bundled.

    Usable from doctests via ``getfixture("fake_pipeline_registry")``.
    """
    import sys as _sys

    class _FakeDist:
        name = "fake-ds-dist"
        version = "1.2.3"

    class _FakeEntryPoint:
        group = "MEDS_extract.pipelines"
        dist = _FakeDist()

        def __init__(self, name: str, value: str):
            self.name = name
            self.value = value

    pkg_dir = tmp_path / "fake_ds_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "event_configs.yaml").write_text(
        'sources:\n  dataset_version: "0.9"\npatients:\n  dob: {code: BIRTH, time: null}\n'
    )
    # Each use materializes the package under a fresh tmp_path, so a module object
    # cached by an earlier test would point at a deleted directory.
    _sys.modules.pop("fake_ds_pkg", None)
    importlib.invalidate_caches()
    monkeypatch.syspath_prepend(str(tmp_path))

    eps = [
        _FakeEntryPoint("Fake-DS", "fake_ds_pkg:event_configs.yaml"),
        _FakeEntryPoint("Bare-Mod", "fake_ds_pkg"),
        _FakeEntryPoint("Missing-Res", "fake_ds_pkg:nope.yaml"),
    ]
    monkeypatch.setattr("MEDS_extract.config.entry_points", lambda group: list(eps))


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
