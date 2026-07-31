"""Tests for the ``meds-extract-run`` CLI.

Deliberately small: spec loading, resolution, and version/stamping semantics are
pinned by ``MessyConfig``'s doctests in ``config.py`` (including the registry rung,
via the ``fake_pipeline_registry`` fixture), and the golden end-to-end run over
``example/`` lives in ``tests/test_run_example.py``. The CLI tests here drive the
REAL console script as a subprocess — exactly how users invoke it, with Hydra
behaving as it does in production — over tiny local fixtures. The one in-process
test covers a genuine library seam: the healed child-environment contract of
``run_command``, which requires patching ``subprocess.run`` to observe and which no
subprocess-level run from an activated test environment could distinguish.
"""

from __future__ import annotations

import os
import subprocess
import sysconfig
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

from MEDS_extract.config import EtlConfig
from MEDS_extract.run import cli as run_cli

if TYPE_CHECKING:
    from pathlib import Path

# A complete tiny ETL spec: enough subjects for the default split fractions to
# yield a train shard, one static event per subject.
_TINY_MESSY = """\
sources:
  dataset_version: "2.0"
  dataset:
    - type: fsspec
      root: {mirror}
etl:
  dataset_name: CLITest
  row_chunksize: 7
  split_fracs:
    train: 0.5
    tuning: 0.25
    held_out: 0.25
_defaults:
  subject_id: $patient_id
patients:
  eye_color:
    code: 'f"EYE_COLOR//{{$eye_color}}"'
    time: null
"""

_TINY_CSV = "patient_id,eye_color\n1,BLUE\n2,BROWN\n3,GREEN\n4,BLUE\n"


def _run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    """Invoke the real ``meds-extract-run`` console script."""
    return subprocess.run(
        ["meds-extract-run", *args, f"hydra.run.dir={tmp_path / '.hydra'}"],
        capture_output=True,
        text=True,
        check=False,
    )


def _write_tiny_spec(tmp_path: Path) -> Path:
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "patients.csv").write_text(_TINY_CSV)
    spec_fp = tmp_path / "spec.yaml"
    spec_fp.write_text(_TINY_MESSY.format(mirror=mirror))
    return spec_fp


def test_run_command_spawns_with_activation_equivalent_child_path(monkeypatch):
    """The child env carries this environment's scripts dir at the front of PATH (the console-script-
    resolution healing the runner exists to provide — it must hold even when the invoking environment is not
    activated, which no subprocess-level run from an activated test environment can distinguish), argv[0] is
    pre-flight resolved, the child's exit code propagates verbatim, and the parent env is untouched."""
    scripts_dir = sysconfig.get_path("scripts")
    stripped = ":".join(p for p in os.environ.get("PATH", "").split(":") if p and p != scripts_dir)
    monkeypatch.setenv("PATH", stripped)

    seen: dict[str, object] = {}

    def fake_run(argv, env, check):
        seen["argv"] = argv
        seen["env_path"] = env["PATH"]
        return subprocess.CompletedProcess(args=argv, returncode=3)

    monkeypatch.setattr(run_cli.subprocess, "run", fake_run)
    rc = run_cli.run_command(["MEDS_transform-pipeline", "cfg.yaml"])

    assert rc == 3
    assert seen["env_path"].split(":")[0] == scripts_dir
    assert seen["argv"][0].startswith(scripts_dir)  # pre-flight resolved to an absolute exe
    assert seen["argv"][1:] == ["cfg.yaml"]
    assert os.environ["PATH"] == stripped  # parent env untouched


def test_run_cli_full_flow(tmp_path):
    """The real console script end-to-end over a tiny local spec: downloads from the
    fsspec mirror into the default destination, synthesizes a fully inlined pipeline
    config under the work dir, runs the real pipeline, and exits 0."""
    spec_fp = _write_tiny_spec(tmp_path)
    out_dir = tmp_path / "meds"

    result = _run_cli(tmp_path, f"spec={spec_fp}", f"output_dir={out_dir}")
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    # Raw data staged into the documented default destination (under the work dir).
    assert (out_dir / ".meds_extract_run" / "raw_input" / "patients.csv").exists()

    # The final cohort landed in output_dir itself.
    assert (out_dir / "data" / "train" / "0.parquet").exists()
    assert (out_dir / "metadata" / "codes.parquet").exists()

    # Synthesized config: every value inlined (no env-var or interpolation
    # indirection) — the only channel through which the computed identity reaches
    # the pipeline subprocess.
    pipeline_fp = out_dir / ".meds_extract_run" / "pipeline.yaml"
    text = pipeline_fp.read_text()
    assert "${" not in text and "oc.env" not in text
    cfg = OmegaConf.load(pipeline_fp)
    assert cfg.etl_metadata.dataset_name == "CLITest"
    assert cfg.etl_metadata.dataset_version == "2.0"  # from sources.dataset_version
    assert cfg.event_conversion_config_fp == str(spec_fp)
    assert cfg.output_dir == str(out_dir)
    stages = OmegaConf.to_container(cfg.stages)
    assert stages[0] == {"shard_events": {"row_chunksize": 7}}
    assert [s if isinstance(s, str) else next(iter(s)) for s in stages] == list(EtlConfig.DEFAULT_PIPELINE)


def test_run_cli_download_failure_stops_the_run(tmp_path):
    """A failing download child sinks the run non-zero; the pipeline is never started."""
    spec_fp = tmp_path / "spec.yaml"
    # The fsspec root does not exist -> the download child fails.
    spec_fp.write_text(_TINY_MESSY.format(mirror=tmp_path / "no_such_mirror"))
    out_dir = tmp_path / "meds"

    result = _run_cli(tmp_path, f"spec={spec_fp}", f"output_dir={out_dir}")
    assert result.returncode != 0
    assert "meds-extract-download failed" in result.stdout + result.stderr
    # The pipeline config is only written (and the pipeline only spawned) after a
    # successful download.
    assert not (out_dir / ".meds_extract_run" / "pipeline.yaml").exists()
    assert not (out_dir / "data").exists()


def test_run_cli_config_errors_exit_one(tmp_path):
    """Bad inputs fail before any work: an unresolvable spec; a path-resolved spec
    omitting dataset_name; an implausible flag combination (input_dir with a
    download_key)."""
    result = _run_cli(tmp_path, "spec=No-Such-Pipeline", f"output_dir={tmp_path / 'o1'}")
    assert result.returncode == 1
    assert "not a registered pipeline name" in result.stdout + result.stderr

    nameless = tmp_path / "nameless.yaml"
    nameless.write_text('sources:\n  dataset_version: "1"\npatients:\n  dob: {code: BIRTH, time: null}\n')
    result = _run_cli(tmp_path, f"spec={nameless}", f"output_dir={tmp_path / 'o2'}")
    assert result.returncode == 1
    assert "omits dataset_name" in result.stdout + result.stderr

    spec_fp = _write_tiny_spec(tmp_path)
    result = _run_cli(tmp_path, f"spec={spec_fp}", f"output_dir={tmp_path / 'o3'}", f"input_dir={tmp_path}")
    assert result.returncode == 1
    assert "input_dir= is for download-free runs" in result.stdout + result.stderr

    for d in ("o1", "o2", "o3"):
        assert not (tmp_path / d).exists()  # nothing was written


def test_run_cli_download_free_run_and_pipeline_failure_propagates(tmp_path):
    """``download_key=null input_dir=...`` skips the download; a pipeline-stage failure (missing raw table
    file) surfaces as a non-zero runner exit."""
    spec_fp = _write_tiny_spec(tmp_path)
    empty_input = tmp_path / "empty_input"
    empty_input.mkdir()  # no patients.csv -> shard_events fails

    result = _run_cli(
        tmp_path,
        f"spec={spec_fp}",
        f"output_dir={tmp_path / 'meds'}",
        "download_key=null",
        f"input_dir={empty_input}",
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "skipping the download stage" in combined
