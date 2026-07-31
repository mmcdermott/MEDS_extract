"""Unit tests for the ``meds-extract-run`` CLI wiring.

Fast and offline, and deliberately small: spec loading, resolution, and
version/stamping semantics are pinned by ``EtlConfig``'s doctests in ``config.py``
(including the registry rung, via the ``fake_pipeline_registry`` fixture), and the
real end-to-end subprocess run over ``example/`` lives in
``tests/test_run_example.py``. What remains here is what neither can cover: the
healed child-environment contract of ``run_command`` and the CLI's process
orchestration (child argv, exit-code propagation, error mapping) with the spawns
stubbed — except one test that lets the real ``meds-extract-download`` child run
against a local fsspec mirror.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from typing import TYPE_CHECKING

import pytest
from omegaconf import OmegaConf

from MEDS_extract.config import EtlConfig
from MEDS_extract.run import cli as run_cli

if TYPE_CHECKING:
    from pathlib import Path

_MINIMAL_MESSY = """\
etl:
  dataset_name: FakeDS
  raw_dataset_version: "0.9"
patients:
  dob: {code: BIRTH, time: null}
"""

_SPEC_WITH_SOURCES = """\
sources:
  dataset_version: "2.0"
  dataset:
    - type: fsspec
      root: {mirror}
etl:
  dataset_name: CLITest
  row_chunksize: 7
patients:
  dob: {{code: BIRTH, time: null}}
"""


def test_run_command_spawns_with_activation_equivalent_child_path(monkeypatch):
    """The child env carries this environment's scripts dir at the front of PATH (the console-script-
    resolution healing the runner exists to provide — it must hold even when the invoking environment is not
    activated, which no e2e run from an activated test environment can distinguish), argv[0] is pre-flight
    resolved, the child's exit code propagates verbatim, and the parent env is untouched."""
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


def _invoke_cli(tmp_path, monkeypatch, *args: str) -> SystemExit:
    """Run ``run_cli.main`` in-process with Hydra argv and return the SystemExit."""
    monkeypatch.setattr(sys, "argv", ["meds-extract-run", *args, f"hydra.run.dir={tmp_path / '.hydra'}"])
    with pytest.raises(SystemExit) as excinfo:
        run_cli.main()
    return excinfo.value


def _stub_run_command(monkeypatch, rc: int = 0) -> list[list[str]]:
    """Replace the CLI's child spawns with a recorder; returns the recorded argvs."""
    calls: list[list[str]] = []
    monkeypatch.setattr("MEDS_extract.run.cli.run_command", lambda argv: calls.append(argv) or rc)
    return calls


def _write_spec(tmp_path: Path, body: str) -> Path:
    spec_fp = tmp_path / "spec.yaml"
    spec_fp.write_text(body)
    return spec_fp


def test_run_cli_full_flow_with_real_download_child(tmp_path, monkeypatch):
    """The full CLI flow with a REAL ``meds-extract-download`` child (local fsspec mirror) and a stubbed
    pipeline child: staging lands in ``raw_input/``, the synthesized config is fully inlined at the documented
    location, and exit is 0."""
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "patients.csv").write_text("patient_id,dob\n1,2000-01-01\n")
    spec_fp = _write_spec(tmp_path, _SPEC_WITH_SOURCES.format(mirror=mirror))
    root = tmp_path / "out"

    pipeline_calls: list[list[str]] = []
    real_run_command = run_cli.run_command

    def hybrid(argv):
        if argv[0] == "meds-extract-download":
            return real_run_command(argv)  # the real subprocess, against the local mirror
        pipeline_calls.append(argv)
        return 0

    monkeypatch.setattr("MEDS_extract.run.cli.run_command", hybrid)

    err = _invoke_cli(tmp_path, monkeypatch, f"spec={spec_fp}", f"root_output_dir={root}")
    assert err.code == 0

    # The download child staged the mirror into the documented default location.
    assert (root / "raw_input" / "patients.csv").exists()

    # The pipeline child was pointed at the synthesized config.
    pipeline_fp = root / ".meds_extract_run" / "pipeline.yaml"
    assert pipeline_calls == [["MEDS_transform-pipeline", str(pipeline_fp)]]

    # Synthesized config: every value inlined (no env-var or interpolation
    # indirection) — the only channel through which the computed identity reaches
    # the pipeline subprocess.
    text = pipeline_fp.read_text()
    assert "${" not in text and "oc.env" not in text
    cfg = OmegaConf.load(pipeline_fp)
    assert cfg.etl_metadata.dataset_name == "CLITest"
    assert cfg.etl_metadata.dataset_version == "2.0"
    assert cfg.event_conversion_config_fp == str(spec_fp)
    assert cfg.input_dir == str(root / "raw_input")
    assert cfg.output_dir == str(root / "MEDS_output")
    stages = OmegaConf.to_container(cfg.stages)
    assert stages[0] == {"shard_events": {"row_chunksize": 7}}
    assert [s if isinstance(s, str) else next(iter(s)) for s in stages] == list(EtlConfig.DEFAULT_PIPELINE)


def test_run_cli_download_child_argv_and_failure_propagates(tmp_path, monkeypatch):
    """The download child is spawned with the renamed ``output_dir=`` argument; its failure code propagates
    verbatim and the pipeline child is never spawned."""
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    spec_fp = _write_spec(tmp_path, _SPEC_WITH_SOURCES.format(mirror=mirror))
    root = tmp_path / "out"

    calls = _stub_run_command(monkeypatch, rc=7)
    err = _invoke_cli(tmp_path, monkeypatch, f"spec={spec_fp}", f"root_output_dir={root}")
    assert err.code == 7

    (download_argv,) = calls  # pipeline never spawned after a failed download
    assert download_argv[0] == "meds-extract-download"
    assert f"output_dir={root / 'raw_input'}" in download_argv
    assert "key=dataset" in download_argv


def test_run_cli_config_errors_exit_one(tmp_path, monkeypatch):
    """Spec-loading failures (unresolvable spec; path-mode spec omitting dataset_name) map to exit 1 before
    any synthesis or child spawn."""
    calls = _stub_run_command(monkeypatch)

    err = _invoke_cli(tmp_path, monkeypatch, "spec=No-Such-Pipeline", f"root_output_dir={tmp_path / 'o1'}")
    assert err.code == 1

    nameless_fp = _write_spec(
        tmp_path, 'sources:\n  dataset_version: "1"\npatients:\n  dob: {code: BIRTH, time: null}\n'
    )
    err = _invoke_cli(tmp_path, monkeypatch, f"spec={nameless_fp}", f"root_output_dir={tmp_path / 'o2'}")
    assert err.code == 1

    assert calls == []  # no child ever spawned
    assert not (tmp_path / "o1").exists() and not (tmp_path / "o2").exists()


def test_run_cli_pipeline_failure_code_propagates(tmp_path, monkeypatch):
    """With ``download_key=null`` the download child is skipped entirely, and the pipeline child's non-zero
    exit code becomes the runner's exit code."""
    spec_fp = _write_spec(tmp_path, _MINIMAL_MESSY)
    calls = _stub_run_command(monkeypatch, rc=5)

    err = _invoke_cli(
        tmp_path, monkeypatch, f"spec={spec_fp}", f"root_output_dir={tmp_path / 'out'}", "download_key=null"
    )
    assert err.code == 5
    assert [argv[0] for argv in calls] == ["MEDS_transform-pipeline"]  # no download spawn
