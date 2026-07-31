"""End-to-end test for ``meds-extract-run`` over the ``example/`` walkthrough.

The one-command counterpart of ``tests/test_example.py``: where that test invokes
``meds-extract-download`` + ``MEDS_transform-pipeline`` separately (the two-command
walkthrough), this one drives the exact same MESSY file through the generic runner —
``meds-extract-run spec=example/messy.yaml root_output_dir=... download_key=null``
against pre-staged raw data — and regression-compares the final ``data/`` +
``metadata/`` outputs against the same committed golden fixtures. A green run proves
the ``etl:`` block in ``example/messy.yaml`` reproduces ``example/pipeline.yaml``'s
outputs bit-for-bit, and that the synthesized pipeline config carries fully inlined
values.

``download_key=null`` (with the bundled CSVs copied into place) keeps this test
offline: with downloading on, the spec's always-appended ``common`` bucket would pull
the MIMIC-IV demo from PhysioNet, which ``test_example.py`` already covers. The
runner's ``meds-extract-download`` subprocess leg is covered offline in
``tests/test_run.py`` via a local fsspec source.

This drives the REAL subprocess chain: the ``meds-extract-run`` console script,
which itself spawns ``MEDS_transform-pipeline`` with the healed (activation-
equivalent) child PATH — so the console-script-resolution healing is exercised
end-to-end. Unlike
``test_example.py`` (whose PhysioNet leg forces the ``integration`` marker), this
test is fully offline and runs in a few seconds on the tiny example dataset, so it
lives in the default (non-integration) lane — every plain ``pytest`` run proves the
runner end-to-end.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from io import StringIO
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal
from pretty_print_directory import print_directory

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example"
RAW_DATA = EXAMPLE_DIR / "raw_data"
MESSY_YAML = EXAMPLE_DIR / "messy.yaml"
EXPECTED = EXAMPLE_DIR / "expected_output"


def _debug(root: Path, run: subprocess.CompletedProcess) -> str:
    sio = StringIO()
    if root.exists():
        print_directory(root, file=sio)
    log_fp = root / "MEDS_output" / ".logs" / "pipeline.log"
    log = log_fp.read_text(encoding="utf-8") if log_fp.exists() else "(pipeline.log did not exist)"
    return (
        f"root tree:\n{sio.getvalue()}\n\npipeline log:\n{log}\n\n"
        f"returncode={run.returncode}\nstdout:\n{run.stdout}\nstderr:\n{run.stderr}"
    )


def test_meds_extract_run_example_end_to_end():
    """``meds-extract-run`` over ``example/messy.yaml`` reproduces the golden outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "run_root"

        # Pre-stage the bundled raw CSVs where the runner's pipeline will read them
        # (``<root>/raw_input`` — the documented default). ``download_key=null`` below
        # then skips the sources stage entirely, keeping the test offline.
        shutil.copytree(RAW_DATA, root / "raw_input")

        cmd = [
            "meds-extract-run",
            f"spec={MESSY_YAML}",
            f"root_output_dir={root}",
            "download_key=null",
            f"hydra.run.dir={Path(tmpdir) / '.hydra'}",
        ]
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        assert run.returncode == 0, f"meds-extract-run failed:\n{_debug(root, run)}"

        output_dir = root / "MEDS_output"
        debug = _debug(root, run)

        # The synthesized pipeline config landed at the documented location with every
        # value inlined — no env-var or interpolation indirection survives synthesis.
        pipeline_fp = root / ".meds_extract_run" / "pipeline.yaml"
        assert pipeline_fp.exists(), f"synthesized pipeline config missing\n{debug}"
        pipeline_text = pipeline_fp.read_text(encoding="utf-8")
        assert "${" not in pipeline_text and "oc.env" not in pipeline_text, pipeline_text
        assert str(MESSY_YAML) in pipeline_text  # event_conversion_config_fp, inlined
        assert "dataset_name: MEDS_extract_example" in pipeline_text
        # The computed version stamp reaches the MEDS_transform-pipeline subprocess
        # through this file — there is no in-process seam — so it must be inlined here
        # (and it surfaces in metadata/dataset.json, asserted below).
        assert "dataset_version: '0.1'" in pipeline_text

        # Golden regression: identical assertions to test_example.py, over the runner's
        # output tree.
        want_data = sorted((EXPECTED / "data").rglob("*.parquet"))
        assert want_data, "expected-output fixture has no data parquets — regenerate it"
        for want_fp in want_data:
            rel = want_fp.relative_to(EXPECTED)
            got_fp = output_dir / rel
            assert got_fp.exists(), f"missing output {rel}\n{debug}"
            assert_frame_equal(pl.read_parquet(got_fp), pl.read_parquet(want_fp))

        want_metadata = sorted(p for p in (EXPECTED / "metadata").rglob("*") if p.is_file())
        assert want_metadata, "expected-output fixture has no metadata — regenerate it"
        for want_fp in want_metadata:
            rel = want_fp.relative_to(EXPECTED)
            got_fp = output_dir / rel
            assert got_fp.exists(), f"missing output {rel}\n{debug}"
            if want_fp.suffix == ".parquet":
                assert_frame_equal(pl.read_parquet(got_fp), pl.read_parquet(want_fp), check_row_order=False)
            elif want_fp.suffix == ".json":
                assert json.loads(got_fp.read_text(encoding="utf-8")) == json.loads(
                    want_fp.read_text(encoding="utf-8")
                ), f"json mismatch {rel}\n{debug}"

        # ``dataset.json``: name from ``etl.dataset_name``; version stamped by the
        # runner — path-mode resolution has no providing distribution, so the stamp is
        # the spec's ``sources.dataset_version`` alone ("0.1", matching pipeline.yaml).
        dataset_json = json.loads((output_dir / "metadata" / "dataset.json").read_text(encoding="utf-8"))
        assert dataset_json["dataset_name"] == "MEDS_extract_example"
        assert dataset_json["dataset_version"] == "0.1"
