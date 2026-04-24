"""End-to-end test for the ``example/`` walkthrough.

Runs the full pipeline described in ``example/README.md`` — ``meds-extract-download``
stages the bundled synthetic CSVs into a tmp ``raw_input_dir``, then
``MEDS_transform-pipeline`` runs every MEDS_extract stage end-to-end — and regression-
compares the final ``data/`` + ``metadata/`` outputs to ``example/expected_output/``.

The test mirrors the `MEDS_transforms` project's ``test_simple_example_pipeline``
pattern: subprocess the CLIs the way users invoke them, diff parquet frames with
``polars.testing.assert_frame_equal``, and surface full pipeline logs on failure so
debugging an unexpected regression doesn't require a rerun.

Gated behind the ``integration`` marker — the run is a real end-to-end walkthrough
(several seconds, touches every stage) so we keep default ``pytest`` snappy. CI runs
this in a dedicated ``integration_tests`` job that fires only after the fast unit-test
jobs are green.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pretty_print_directory import print_directory

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example"
RAW_DATA = EXAMPLE_DIR / "raw_data"
SOURCES_YAML = EXAMPLE_DIR / "sources.yaml"
PIPELINE_YAML = EXAMPLE_DIR / "pipeline.yaml"
EVENT_CFG = EXAMPLE_DIR / "event_cfg.yaml"
EXPECTED = EXAMPLE_DIR / "expected_output"


def _format_debug(output_dir: Path, stages: list[dict]) -> str:
    """Format subprocess output + output-dir tree + pipeline log for assertion messages."""
    sio = StringIO()
    if output_dir.exists():
        print_directory(output_dir, file=sio)
    tree = sio.getvalue() or "(output_dir did not exist)"

    log_fp = output_dir / ".logs" / "pipeline.log"
    pipeline_log = log_fp.read_text() if log_fp.exists() else "(pipeline.log did not exist)"

    parts = [f"Output tree:\n{tree}", f"Pipeline log:\n{pipeline_log}"]
    for stage in stages:
        parts.append(f"[{stage['name']}] returncode={stage['returncode']}")
        parts.append(f"[{stage['name']}] stdout:\n{stage['stdout']}")
        parts.append(f"[{stage['name']}] stderr:\n{stage['stderr']}")
    return "\n\n".join(parts)


@pytest.mark.integration
def test_example_pipeline_end_to_end():
    """End-to-end walkthrough: download → 8 stages → regression-diff.

    This is the automated version of the tutorial in ``example/README.md``. A green
    run proves the whole pipeline (including the ``meds-extract-download`` pre-stage)
    still produces bit-for-bit identical ``data/*.parquet`` + ``metadata/*.parquet``
    outputs on the committed synthetic input.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_p = Path(tmpdir)
        raw_input = tmpdir_p / "raw"
        output_dir = tmpdir_p / "out"

        # Env-var interpolation lets ``example/sources.yaml`` and ``example/pipeline.yaml``
        # stay portable: the configs live in the repo with ``${oc.env:...}`` placeholders
        # and every caller (this test, the README walkthrough, downstream ETLs) supplies
        # the concrete paths at invocation time.
        env = {**os.environ, "EXAMPLE_RAW_DATA": str(RAW_DATA), "EXAMPLE_EVENT_CFG": str(EVENT_CFG)}

        stages: list[dict] = []

        # Stage 0: ``meds-extract-download`` — populates the tmp ``raw_input`` from the
        # ``fsspec`` source bound to the committed ``example/raw_data/``.
        download_cmd = [
            sys.executable,
            "-m",
            "MEDS_extract.download.cli",
            f"spec={SOURCES_YAML}",
            f"raw_input_dir={raw_input}",
            f"hydra.run.dir={tmpdir_p / '.hydra_download'}",
        ]
        download_run = subprocess.run(download_cmd, capture_output=True, text=True, env=env)
        stages.append(
            {
                "name": "meds-extract-download",
                "returncode": download_run.returncode,
                "stdout": download_run.stdout,
                "stderr": download_run.stderr,
            }
        )
        assert download_run.returncode == 0, (
            f"meds-extract-download failed:\n{_format_debug(output_dir, stages)}"
        )
        # ``event_cfg.yaml`` is co-located with ``raw_input_dir`` in the example because the
        # pipeline's default ``event_conversion_config_fp`` looks there. The download CLI
        # only stages ``sources:`` entries; the event config ships with the ETL itself.
        (raw_input / "event_cfg.yaml").write_text(EVENT_CFG.read_text())

        # Stages 1–8: the full MEDS_extract pipeline via ``MEDS_transform-pipeline``.
        pipeline_cmd = [
            "MEDS_transform-pipeline",
            str(PIPELINE_YAML),
            "--overrides",
            f"input_dir={raw_input}",
            f"output_dir={output_dir}",
            f"hydra.run.dir={tmpdir_p / '.hydra_pipeline'}",
        ]
        pipeline_run = subprocess.run(pipeline_cmd, capture_output=True, text=True, env=env)
        stages.append(
            {
                "name": "MEDS_transform-pipeline",
                "returncode": pipeline_run.returncode,
                "stdout": pipeline_run.stdout,
                "stderr": pipeline_run.stderr,
            }
        )
        assert pipeline_run.returncode == 0, (
            f"MEDS_transform-pipeline failed:\n{_format_debug(output_dir, stages)}"
        )

        debug = _format_debug(output_dir, stages)

        # Regression-diff every committed expected file against its fresh counterpart.
        # Missing files and content mismatches both surface the pipeline log so a
        # reviewer can tell at a glance which stage diverged.
        want_data = sorted((EXPECTED / "data").rglob("*.parquet"))
        assert want_data, "expected-output fixture has no data parquets — regenerate it"
        for want_fp in want_data:
            rel = want_fp.relative_to(EXPECTED)
            got_fp = output_dir / rel
            assert got_fp.exists(), f"missing output {rel}\n{debug}"
            assert_frame_equal(pl.read_parquet(got_fp), pl.read_parquet(want_fp))

        want_metadata = sorted((EXPECTED / "metadata").rglob("*.*"))
        assert want_metadata, "expected-output fixture has no metadata — regenerate it"
        for want_fp in want_metadata:
            rel = want_fp.relative_to(EXPECTED)
            got_fp = output_dir / rel
            assert got_fp.exists(), f"missing output {rel}\n{debug}"
            if want_fp.suffix == ".parquet":
                assert_frame_equal(
                    pl.read_parquet(got_fp), pl.read_parquet(want_fp), check_row_order=False
                )
            elif want_fp.suffix == ".json":
                assert got_fp.read_text() == want_fp.read_text(), f"json mismatch {rel}\n{debug}"

        # ``dataset.json`` is generated by ``finalize_MEDS_metadata`` but not committed
        # because three of its fields are non-deterministic across runs:
        #   * ``created_at`` — wall-clock timestamp.
        #   * ``etl_version``, ``meds_version`` — pinned at install time, drift with deps.
        # We still verify the stable schema so a malformed dataset.json still fails.
        dataset_json = json.loads((output_dir / "metadata" / "dataset.json").read_text())
        assert dataset_json["dataset_name"] == "MEDS_extract_example"
        assert dataset_json["dataset_version"] == "0.1"
        for required in ("etl_name", "etl_version", "meds_version", "created_at"):
            assert required in dataset_json, f"dataset.json missing {required!r}: {dataset_json}"
