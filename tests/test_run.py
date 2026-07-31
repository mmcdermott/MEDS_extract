"""Unit tests for the ``meds-extract-run`` layer (spec loading + CLI wiring).

Fast and offline. Registry resolution is exercised with synthetic packages +
monkeypatched entry points; CLI tests stub the child-process spawns
(``run_command``) except one that lets the real ``meds-extract-download`` child run
against a local fsspec mirror. Version/stamping/validation semantics are pinned by
``EtlConfig``'s doctests in ``config.py`` — these tests cover only what needs a
process or registry seam. The real end-to-end subprocess run over ``example/``
lives in ``tests/test_run_example.py``.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import pytest
from omegaconf import OmegaConf

from MEDS_extract.config import EtlConfig

if TYPE_CHECKING:
    from pathlib import Path

# ── Synthetic entry-point scaffolding ──────────────────────────────────────────────


class _FakeDist:
    name = "fake-ds-dist"
    version = "1.2.3"


class _FakeEntryPoint:
    """The duck-typed slice of ``importlib.metadata.EntryPoint`` the loader uses:

    ``.name``, the raw ``.value`` string (parsed, never ``load()``-ed), and ``.dist``.
    """

    group = "MEDS_extract.pipelines"
    dist = _FakeDist()

    def __init__(self, name: str, value: str):
        self.name = name
        self.value = value


def _register(monkeypatch, *eps: _FakeEntryPoint) -> None:
    monkeypatch.setattr("MEDS_extract.config.entry_points", lambda group: list(eps))


def _install_fake_pkg(tmp_path: Path, monkeypatch, name: str, files: dict[str, str]) -> None:
    """Materialize an importable package ``name`` under ``tmp_path`` with ``files``."""
    pkg_dir = tmp_path / name
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    for fname, body in files.items():
        (pkg_dir / fname).write_text(body)
    monkeypatch.syspath_prepend(str(tmp_path))


_MINIMAL_MESSY = """\
etl:
  dataset_name: FakeDS
  raw_dataset_version: "0.9"
patients:
  dob: {code: BIRTH, time: null}
"""

# No etl: block at all — identity must come from the registry name and the
# sources-owned dataset_version.
_ETL_FREE_MESSY = """\
sources:
  dataset_version: "0.9"
patients:
  dob: {code: BIRTH, time: null}
"""


# ── EtlConfig.load: the registry rung (needs synthetic entry points) ───────────────


def test_load_registered_name(tmp_path, monkeypatch):
    """A registered name resolves via the entry point's ``module:file`` value: local file + portable pkg://
    ref + distribution version, with ``dataset_name`` inherited from the registered name when the (absent)
    ``etl:`` block omits it."""
    _install_fake_pkg(tmp_path, monkeypatch, "fake_ds_pkg", {"event_configs.yaml": _ETL_FREE_MESSY})
    _register(monkeypatch, _FakeEntryPoint("Fake-DS", "fake_ds_pkg:event_configs.yaml"))

    etl = EtlConfig.load("Fake-DS")
    assert etl.spec_fp.read_text() == _ETL_FREE_MESSY
    assert etl.spec_ref == "pkg://fake_ds_pkg.event_configs.yaml"
    assert etl.dataset_name == "Fake-DS"
    assert etl.dataset_version_for() == "0.9:1.2.3"


def test_load_malformed_registrations(tmp_path, monkeypatch):
    """Bare-module registrations and registrations naming a missing resource both fail with targeted messages;
    a name matching nothing lists the registered names."""
    _install_fake_pkg(tmp_path, monkeypatch, "bad_reg_pkg", {"other.yaml": "x: 1"})
    _register(
        monkeypatch,
        _FakeEntryPoint("Bare-Mod", "bad_reg_pkg"),
        _FakeEntryPoint("Missing-Res", "bad_reg_pkg:event_configs.yaml"),
    )

    with pytest.raises(ValueError, match=r"'<package\.module>:<filename\.yaml>'"):
        EtlConfig.load("Bare-Mod")
    with pytest.raises(ValueError, match="no resource named 'event_configs.yaml'"):
        EtlConfig.load("Missing-Res")
    with pytest.raises(FileNotFoundError, match=r"Registered pipelines: Bare-Mod, Missing-Res\."):
        EtlConfig.load("Fake-DS-Typo")


# ── run_command: healed child PATH ─────────────────────────────────────────────────


def test_run_command_spawns_with_activation_equivalent_child_path(monkeypatch):
    """The child env carries the scripts dir at the front of PATH; the parent env is untouched.

    This is the #398 healing seam: the pipeline runner's own bare
    ``MEDS_transform-stage`` spawns inherit the child env.
    """
    import subprocess
    import sysconfig

    from MEDS_extract.run import cli as cli_mod

    scripts_dir = sysconfig.get_path("scripts")
    stripped = ":".join(p for p in os.environ.get("PATH", "").split(":") if p and p != scripts_dir)
    monkeypatch.setenv("PATH", stripped)

    seen: dict[str, object] = {}

    def fake_run(argv, env, check):
        seen["argv"] = argv
        seen["env_path"] = env["PATH"]
        return subprocess.CompletedProcess(args=argv, returncode=3)

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    rc = cli_mod.run_command(["MEDS_transform-pipeline", "cfg.yaml"])

    assert rc == 3  # the child's exit code propagates verbatim
    assert seen["env_path"].split(":")[0] == scripts_dir
    # argv[0] was pre-flight resolved to an absolute executable in the scripts dir.
    assert seen["argv"][0].startswith(scripts_dir)
    assert seen["argv"][1:] == ["cfg.yaml"]
    assert os.environ["PATH"] == stripped  # parent env untouched


# ── CLI wiring (child spawns stubbed unless noted) ─────────────────────────────────


def _invoke_cli(tmp_path, monkeypatch, *args: str) -> SystemExit:
    """Run ``MEDS_extract.run.cli.main`` in-process with Hydra argv and return the SystemExit."""
    from MEDS_extract.run.cli import main

    monkeypatch.setattr(sys, "argv", ["meds-extract-run", *args, f"hydra.run.dir={tmp_path / '.hydra'}"])
    with pytest.raises(SystemExit) as excinfo:
        main()
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


def test_run_cli_full_flow_with_real_download_child(tmp_path, monkeypatch):
    """The full CLI flow with a REAL ``meds-extract-download`` child (local fsspec mirror) and a stubbed
    pipeline child: staging lands in ``raw_input/``, the synthesized config is fully inlined at the documented
    location, and exit is 0."""
    from MEDS_extract.run.cli import run_command as real_run_command

    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "patients.csv").write_text("patient_id,dob\n1,2000-01-01\n")
    spec_fp = _write_spec(tmp_path, _SPEC_WITH_SOURCES.format(mirror=mirror))
    root = tmp_path / "out"

    pipeline_calls: list[list[str]] = []

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
    # indirection); the version stamp reaches the pipeline THROUGH this file (there
    # is no in-process seam), sourced from sources.dataset_version (path mode: no
    # dist suffix); the curated stage option landed on its stage.
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
    """The download child gets spec/raw_input_dir/key/do_overwrite; its failure code propagates and the
    pipeline child is never spawned."""
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    spec_fp = _write_spec(tmp_path, _SPEC_WITH_SOURCES.format(mirror=mirror))
    root = tmp_path / "out"

    calls = _stub_run_command(monkeypatch, rc=7)
    err = _invoke_cli(
        tmp_path, monkeypatch, f"spec={spec_fp}", f"root_output_dir={root}", "do_overwrite=true"
    )
    assert err.code == 7  # the child's exit code, verbatim

    (download_argv,) = calls  # pipeline never spawned after a failed download
    assert download_argv[0] == "meds-extract-download"
    assert f"spec={spec_fp}" in download_argv
    assert f"raw_input_dir={root / 'raw_input'}" in download_argv
    assert "key=dataset" in download_argv
    assert "do_overwrite=True" in download_argv


def test_run_cli_registry_mode_synthesizes_portable_ref(tmp_path, monkeypatch):
    """An etl:-free registered spec runs with zero config: name from the entry point, version from
    sources.dataset_version:dist, and the synthesized config carries the portable pkg:// reference (not a
    machine-local resolved path)."""
    _install_fake_pkg(tmp_path, monkeypatch, "cli_reg_pkg", {"event_configs.yaml": _ETL_FREE_MESSY})
    _register(monkeypatch, _FakeEntryPoint("Fake-DS", "cli_reg_pkg:event_configs.yaml"))
    _stub_run_command(monkeypatch)

    root = tmp_path / "out"
    err = _invoke_cli(tmp_path, monkeypatch, "spec=Fake-DS", f"root_output_dir={root}", "do_download=false")
    assert err.code == 0

    cfg = OmegaConf.load(root / ".meds_extract_run" / "pipeline.yaml")
    assert cfg.etl_metadata.dataset_name == "Fake-DS"
    assert cfg.etl_metadata.dataset_version == "0.9:1.2.3"
    assert cfg.event_conversion_config_fp == "pkg://cli_reg_pkg.event_configs.yaml"


def test_run_cli_raw_input_dir_override_feeds_pipeline_input(tmp_path, monkeypatch):
    """``raw_input_dir=`` (pre-staged data elsewhere) lands as the pipeline's input_dir."""
    spec_fp = _write_spec(tmp_path, _MINIMAL_MESSY)
    staged = tmp_path / "prestaged"
    staged.mkdir()
    _stub_run_command(monkeypatch)

    root = tmp_path / "out"
    err = _invoke_cli(
        tmp_path,
        monkeypatch,
        f"spec={spec_fp}",
        f"root_output_dir={root}",
        "do_download=false",
        f"raw_input_dir={staged}",
    )
    assert err.code == 0
    cfg = OmegaConf.load(root / ".meds_extract_run" / "pipeline.yaml")
    assert cfg.input_dir == str(staged)


def test_run_cli_config_errors_exit_one(tmp_path, monkeypatch):
    """Spec-loading failures (unresolvable spec; path-mode spec omitting dataset_name) map to exit 1 before
    any synthesis or child spawn."""
    calls = _stub_run_command(monkeypatch)

    err = _invoke_cli(tmp_path, monkeypatch, "spec=No-Such-Pipeline", f"root_output_dir={tmp_path / 'o1'}")
    assert err.code == 1

    nameless_fp = _write_spec(tmp_path, _ETL_FREE_MESSY)
    err = _invoke_cli(tmp_path, monkeypatch, f"spec={nameless_fp}", f"root_output_dir={tmp_path / 'o2'}")
    assert err.code == 1

    assert calls == []  # no child ever spawned
    assert not (tmp_path / "o1").exists() and not (tmp_path / "o2").exists()


def test_run_cli_pipeline_failure_code_propagates(tmp_path, monkeypatch):
    """The pipeline child's non-zero exit code becomes the runner's exit code."""
    spec_fp = _write_spec(tmp_path, _MINIMAL_MESSY)
    _stub_run_command(monkeypatch, rc=5)

    err = _invoke_cli(
        tmp_path, monkeypatch, f"spec={spec_fp}", f"root_output_dir={tmp_path / 'out'}", "do_download=false"
    )
    assert err.code == 5
