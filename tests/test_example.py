"""End-to-end test for the ``example/`` walkthrough.

Runs the full pipeline described in ``example/README.md`` — the download step is
invoked as ``python -m MEDS_extract.download.cli`` (rather than the installed
``meds-extract-download`` console script) so this test continues to exercise the
``if __name__ == "__main__"`` block's ``sys.exit(main())`` wiring. The pipeline step
shells out to the installed ``MEDS_transform-pipeline`` console script. Final
``data/`` + ``metadata/`` outputs are regression-compared to
``example/expected_output/``.

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
MESSY_YAML = EXAMPLE_DIR / "messy.yaml"
PIPELINE_YAML = EXAMPLE_DIR / "pipeline.yaml"
EXPECTED = EXAMPLE_DIR / "expected_output"


def _format_debug(raw_input: Path, output_dir: Path, stages: list[dict]) -> str:
    """Format subprocess output + directory trees + pipeline log for assertion messages.

    Both ``raw_input`` (where the download stage writes) and ``output_dir`` (where the
    pipeline writes) are rendered when present, so a download-stage failure surfaces the
    download artifacts and a pipeline-stage failure surfaces the pipeline artifacts
    without the caller having to pick the right dir per failure site.
    """

    def _tree(dir_path: Path, label: str) -> str:
        if not dir_path.exists():
            return f"{label}:\n({dir_path} did not exist)"
        sio = StringIO()
        print_directory(dir_path, file=sio)
        return f"{label}:\n{sio.getvalue()}"

    log_fp = output_dir / ".logs" / "pipeline.log"
    pipeline_log = log_fp.read_text(encoding="utf-8") if log_fp.exists() else "(pipeline.log did not exist)"

    parts = [
        _tree(raw_input, "raw_input tree"),
        _tree(output_dir, "output_dir tree"),
        f"Pipeline log:\n{pipeline_log}",
    ]
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

        # The single MESSY file (``example/messy.yaml``) carries both the ``sources:``
        # block for ``meds-extract-download`` and the event-conversion config for the
        # pipeline. ``${oc.env:EXAMPLE_RAW_DATA}`` inside the spec resolves the fsspec
        # mirror path at invocation time; ``${oc.env:EXAMPLE_MESSY}`` in ``pipeline.yaml``
        # resolves to the same file so the downstream stages see the event tables.
        env = {
            **os.environ,
            "EXAMPLE_RAW_DATA": str(RAW_DATA),
            "EXAMPLE_MESSY": str(MESSY_YAML),
        }

        stages: list[dict] = []

        # Stage 0: ``meds-extract-download`` — populates ``raw_input`` from both the
        # ``fsspec`` source (bundled synthetic CSVs) AND the ``physionet`` source
        # (MIMIC-IV demo, ~5 MiB of real network traffic). ``concurrency=8`` shaves
        # seconds off the PhysioNet leg. ``hydra.run.dir`` redirects the Hydra output
        # tree under ``tmp_path`` so the test doesn't leave droppings in CWD — users
        # running the README's invocation don't need this override. ``timeout=`` caps a
        # wedged PhysioNet fetch at ~10 min so the whole job fails fast instead of
        # hitting GHA's 30-minute job timeout.
        download_cmd = [
            sys.executable,
            "-m",
            "MEDS_extract.download.cli",
            f"spec={MESSY_YAML}",
            f"raw_input_dir={raw_input}",
            "concurrency=8",
            f"hydra.run.dir={tmpdir_p / '.hydra_download'}",
        ]
        download_run = subprocess.run(download_cmd, capture_output=True, text=True, env=env, timeout=600)
        stages.append(
            {
                "name": "meds-extract-download",
                "returncode": download_run.returncode,
                "stdout": download_run.stdout,
                "stderr": download_run.stderr,
            }
        )
        assert download_run.returncode == 0, (
            f"meds-extract-download failed:\n{_format_debug(raw_input, output_dir, stages)}"
        )

        # Stages 1-8: the full MEDS_extract pipeline via ``MEDS_transform-pipeline``.
        # Uses the SAME MESSY file — the library's ``MessyConfig.parse`` treats
        # ``sources:`` as a reserved top-level key so the event-conversion path doesn't
        # try to interpret it as a table.
        env_pipeline = env
        pipeline_cmd = [
            "MEDS_transform-pipeline",
            str(PIPELINE_YAML),
            "--overrides",
            f"input_dir={raw_input}",
            f"output_dir={output_dir}",
            f"hydra.run.dir={tmpdir_p / '.hydra_pipeline'}",
        ]
        pipeline_run = subprocess.run(
            pipeline_cmd, capture_output=True, text=True, env=env_pipeline, timeout=600
        )
        stages.append(
            {
                "name": "MEDS_transform-pipeline",
                "returncode": pipeline_run.returncode,
                "stdout": pipeline_run.stdout,
                "stderr": pipeline_run.stderr,
            }
        )
        assert pipeline_run.returncode == 0, (
            f"MEDS_transform-pipeline failed:\n{_format_debug(raw_input, output_dir, stages)}"
        )

        debug = _format_debug(raw_input, output_dir, stages)

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

        # ``rglob("*")`` + ``is_file()`` captures dotfile metadata (``.shards.json``) that
        # the narrower ``*.*`` pattern would silently skip.
        want_metadata = sorted(p for p in (EXPECTED / "metadata").rglob("*") if p.is_file())
        assert want_metadata, "expected-output fixture has no metadata — regenerate it"
        for want_fp in want_metadata:
            rel = want_fp.relative_to(EXPECTED)
            got_fp = output_dir / rel
            assert got_fp.exists(), f"missing output {rel}\n{debug}"
            if want_fp.suffix == ".parquet":
                assert_frame_equal(pl.read_parquet(got_fp), pl.read_parquet(want_fp), check_row_order=False)
            elif want_fp.suffix == ".json":
                # Parse before comparing so pre-commit's end-of-file-fixer (which adds a
                # trailing newline to the committed fixture) doesn't diverge from the
                # pipeline's raw-JSON output.
                assert json.loads(got_fp.read_text(encoding="utf-8")) == json.loads(
                    want_fp.read_text(encoding="utf-8")
                ), f"json mismatch {rel}\n{debug}"

        # Beyond "every expected file matches", also assert the output file set equals
        # the fixture set (modulo a known-uncommitted exception list). Catches the case
        # where a new stage starts emitting a file that the committed fixture doesn't
        # know about — the existing regression loop would happily pass while silently
        # letting undocumented output files slip in.
        ignored_outputs = {
            # ``dataset.json`` carries a ``created_at`` timestamp + dep-version fields;
            # schema-checked below, not content-diffed.
            Path("metadata") / "dataset.json",
        }
        expected_rel = {p.relative_to(EXPECTED) for p in (want_data + want_metadata)}
        got_rel = {
            p.relative_to(output_dir)
            for sub in ("data", "metadata")
            for p in (output_dir / sub).rglob("*")
            if p.is_file() and ".logs" not in p.parts
        }
        unexpected = got_rel - expected_rel - ignored_outputs
        assert not unexpected, (
            f"pipeline emitted new outputs the committed fixture doesn't cover: "
            f"{sorted(unexpected)}. Update example/expected_output/ to match, or "
            f"add the new path to ``ignored_outputs`` if it's intentionally "
            f"non-regressable.\n{debug}"
        )

        # ``dataset.json`` is generated by ``finalize_MEDS_metadata`` but not committed
        # because three of its fields are non-deterministic across runs:
        #   * ``created_at`` — wall-clock timestamp.
        #   * ``etl_version``, ``meds_version`` — pinned at install time, drift with deps.
        # We still verify the stable schema so a malformed dataset.json still fails.
        dataset_json = json.loads((output_dir / "metadata" / "dataset.json").read_text(encoding="utf-8"))
        assert dataset_json["dataset_name"] == "MEDS_extract_example"
        assert dataset_json["dataset_version"] == "0.1"
        for required in ("etl_name", "etl_version", "meds_version", "created_at"):
            assert required in dataset_json, f"dataset.json missing {required!r}: {dataset_json}"

        # MIMIC-IV demo files (from the ``common`` / ``physionet`` source in ``sources.yaml``)
        # landed in ``raw_input``. These aren't consumed by the example pipeline — the
        # assertion just proves the PhysioNet download really hit the wire and the
        # sha-verified fetch path completed for every file in ``SHA256SUMS.txt``. Note:
        # ``SHA256SUMS.txt`` itself isn't staged (``PhysioNetSource.list_files`` consumes
        # it for manifest enumeration but doesn't yield it as a fetchable entry), so we
        # check known files from the manifest instead.
        mimic_files = [
            raw_input / "LICENSE.txt",
            raw_input / "hosp" / "patients.csv.gz",
            raw_input / "icu" / "icustays.csv.gz",
        ]
        missing = [p for p in mimic_files if not p.exists()]
        assert not missing, f"MIMIC-IV demo files missing from raw_input:\n{missing}\n\n{debug}"
        # No stale ``.part`` files — atomic-rename invariant held across the real fetch.
        leftover_parts = list(raw_input.rglob("*.part"))
        assert not leftover_parts, f"stale .part files after download: {leftover_parts}"
