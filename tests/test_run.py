"""Unit tests for the ``meds-extract-run`` layer (registry, synthesis, CLI wiring).

Everything here is fast and offline: the registry ladder is exercised with synthetic
packages + monkeypatched entry points, and the CLI tests stub the child-process
spawns (``run_command``) so no real ``meds-extract-download`` /
``MEDS_transform-pipeline`` processes start — except one test that lets the real
download child run against a local fsspec mirror. The real end-to-end subprocess run
over ``example/`` lives in ``tests/test_run_example.py``.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import pytest
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from pathlib import Path

from MEDS_extract.run.registry import (
    CANONICAL_MESSY_FILENAME,
    ResolvedSpec,
    messy_file_for_module,
    resolve_spec,
)

# ── Synthetic entry-point scaffolding ──────────────────────────────────────────────


class _FakeDist:
    name = "fake-ds-dist"
    version = "1.2.3"


_FAKE_DIST = _FakeDist()


class _FakeEntryPoint:
    """The duck-typed slice of ``importlib.metadata.EntryPoint`` the registry uses.

    ``resolve_spec`` reads ``.name`` (registry key), ``.module`` (resource lookup —
    never ``load()``-ed) and ``.dist`` (provenance); a real installed dataset
    package's entry point provides exactly these.
    """

    group = "MEDS_extract.pipelines"

    def __init__(self, name: str, module: str, dist: _FakeDist | None = _FAKE_DIST):
        self.name = name
        self.module = module
        self.dist = dist


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


# ── messy_file_for_module: the bundled-file convention ─────────────────────────────


def test_messy_convention_single_yaml_wins(tmp_path, monkeypatch):
    """A module with exactly one YAML resource resolves to it, regardless of its name."""
    _install_fake_pkg(tmp_path, monkeypatch, "one_yaml_pkg", {"anything.yaml": _MINIMAL_MESSY})
    assert messy_file_for_module("one_yaml_pkg").name == "anything.yaml"


def test_messy_convention_canonical_name_disambiguates(tmp_path, monkeypatch):
    """With several YAMLs, the canonical ``event_configs.yaml`` wins."""
    _install_fake_pkg(
        tmp_path,
        monkeypatch,
        "multi_yaml_pkg",
        {"other.yaml": "x: 1", CANONICAL_MESSY_FILENAME: _MINIMAL_MESSY, "third.yml": "y: 2"},
    )
    assert messy_file_for_module("multi_yaml_pkg").name == CANONICAL_MESSY_FILENAME


def test_messy_convention_ambiguous_layout_errors(tmp_path, monkeypatch):
    """Several YAMLs with no canonical name is a config error naming the candidates."""
    _install_fake_pkg(tmp_path, monkeypatch, "ambiguous_pkg", {"a.yaml": "x: 1", "b.yaml": "y: 2"})
    with pytest.raises(ValueError, match=r"several YAML resources \['a.yaml', 'b.yaml'\]"):
        messy_file_for_module("ambiguous_pkg")


# ── resolve_spec: the three-rung ladder ────────────────────────────────────────────


def test_resolve_spec_registered_name(tmp_path, monkeypatch):
    """A registered name resolves through the entry point, carrying the dist version."""
    _install_fake_pkg(tmp_path, monkeypatch, "fake_ds_pkg", {CANONICAL_MESSY_FILENAME: _MINIMAL_MESSY})
    monkeypatch.setattr(
        "MEDS_extract.run.registry.entry_points",
        lambda group: [_FakeEntryPoint("Fake-DS", "fake_ds_pkg")],
    )

    rs = resolve_spec("Fake-DS")
    assert rs.origin == "registry"
    assert rs.spec_fp.name == CANONICAL_MESSY_FILENAME
    assert rs.spec_fp.read_text() == _MINIMAL_MESSY
    assert rs.dist_version == "1.2.3"


def test_resolve_spec_registered_name_without_dist(tmp_path, monkeypatch):
    """An entry point with no attached distribution still resolves; provenance is None."""
    _install_fake_pkg(tmp_path, monkeypatch, "distless_pkg", {CANONICAL_MESSY_FILENAME: _MINIMAL_MESSY})
    monkeypatch.setattr(
        "MEDS_extract.run.registry.entry_points",
        lambda group: [_FakeEntryPoint("Dist-Less", "distless_pkg", dist=None)],
    )

    rs = resolve_spec("Dist-Less")
    assert rs.origin == "registry"
    assert rs.dist_version is None


def test_resolve_spec_path_mode_beats_nothing_and_reports_registry(tmp_path, monkeypatch):
    """A real file resolves as a path; a miss lists the registered names for typo diagnosis."""
    monkeypatch.setattr(
        "MEDS_extract.run.registry.entry_points",
        lambda group: [_FakeEntryPoint("Fake-DS", "fake_ds_pkg")],
    )

    spec_fp = tmp_path / "spec.yaml"
    spec_fp.write_text(_MINIMAL_MESSY)
    rs = resolve_spec(str(spec_fp))
    assert rs == ResolvedSpec(spec_fp=spec_fp.resolve(), origin="path")

    with pytest.raises(FileNotFoundError, match=r"Registered pipelines: Fake-DS\."):
        resolve_spec("Fake-DS-Typo")


def test_resolve_spec_pkg_mode_missing_file(tmp_path, monkeypatch):
    """A pkg:// reference to a real package but missing file fails naming the resolved path."""
    _install_fake_pkg(tmp_path, monkeypatch, "pkg_mode_pkg", {"spec.yaml": _MINIMAL_MESSY})

    rs = resolve_spec("pkg://pkg_mode_pkg.spec.yaml")
    assert (rs.origin, rs.spec_fp.name) == ("pkg", "spec.yaml")

    with pytest.raises(FileNotFoundError, match="resolved to .* which does not exist"):
        resolve_spec("pkg://pkg_mode_pkg.nope.yaml")


def test_resolve_spec_path_resolver_is_used(tmp_path):
    """The injected ``path_resolver`` (Hydra original-CWD resolution in the CLI) is honored."""
    spec_fp = tmp_path / "spec.yaml"
    spec_fp.write_text(_MINIMAL_MESSY)

    rs = resolve_spec("spec.yaml", path_resolver=lambda s: tmp_path / s)
    assert rs.spec_fp == spec_fp.resolve()


# ── run_command: healed child PATH + missing-script teeth ──────────────────────────


def test_run_command_spawns_with_activation_equivalent_child_path(monkeypatch):
    """The child env carries the scripts dir at the front of PATH; the parent env is untouched.

    This is the #398 healing seam: the pipeline runner's own bare
    ``MEDS_transform-stage`` spawns inherit the child env, so healing the child's
    PATH here retires the console-script-resolution failure class for the whole
    process tree.
    """
    import subprocess
    import sysconfig

    from MEDS_extract.run import pipeline as pipeline_mod

    scripts_dir = sysconfig.get_path("scripts")
    stripped = ":".join(p for p in os.environ.get("PATH", "").split(":") if p and p != scripts_dir)
    monkeypatch.setenv("PATH", stripped)

    seen: dict[str, object] = {}

    def fake_run(argv, env, check):
        seen["argv"] = argv
        seen["env_path"] = env["PATH"]
        return subprocess.CompletedProcess(args=argv, returncode=3)

    monkeypatch.setattr(pipeline_mod.subprocess, "run", fake_run)
    rc = pipeline_mod.run_command(["MEDS_transform-pipeline", "cfg.yaml"])

    assert rc == 3  # the child's exit code propagates verbatim
    assert seen["env_path"].split(":")[0] == scripts_dir
    # argv[0] was pre-flight resolved to an absolute executable in the scripts dir.
    assert seen["argv"][0].startswith(scripts_dir)
    assert seen["argv"][1:] == ["cfg.yaml"]
    assert os.environ["PATH"] == stripped  # parent env untouched


def test_run_command_missing_script_fails_fast(monkeypatch):
    """An unresolvable command fails before spawning, with an actionable message."""
    from MEDS_extract.run import pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod.shutil, "which", lambda *a, **k: None)
    with pytest.raises(FileNotFoundError, match="no-such-script"):
        pipeline_mod.run_command(["no-such-script"])


# ── CLI wiring: synthesis, stamping, exit codes (child spawns stubbed) ─────────────


def _invoke_cli(tmp_path, monkeypatch, *args: str) -> SystemExit:
    """Run ``MEDS_extract.run.cli.main`` in-process with Hydra argv and return the SystemExit."""
    from MEDS_extract.run.cli import main

    monkeypatch.setattr(
        sys,
        "argv",
        ["meds-extract-run", *args, f"hydra.run.dir={tmp_path / '.hydra'}"],
    )
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
    """The full CLI flow with a REAL ``meds-extract-download`` child (local fsspec mirror)
    and a stubbed pipeline child: staging lands in ``raw_input/``, the synthesized config
    is fully inlined at the documented location, and the exit code is 0."""
    from MEDS_extract.run.pipeline import run_command as real_run_command

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
    # is no in-process seam), sourced here from sources.dataset_version (path mode:
    # no dist suffix); the curated stage option landed on its stage.
    text = pipeline_fp.read_text()
    assert "${" not in text and "oc.env" not in text
    cfg = OmegaConf.load(pipeline_fp)
    assert cfg.etl_metadata.dataset_name == "CLITest"
    assert cfg.etl_metadata.dataset_version == "2.0"
    assert cfg.event_conversion_config_fp == str(spec_fp)
    assert cfg.input_dir == str(root / "raw_input")
    assert cfg.output_dir == str(root / "MEDS_output")
    assert cfg.shards_map_fp == f"{root / 'MEDS_output'}/metadata/.shards.json"
    stages = OmegaConf.to_container(cfg.stages)
    assert stages[0] == {"shard_events": {"row_chunksize": 7}}
    assert stages[1:] == [
        "split_and_shard_subjects",
        "convert_to_subject_sharded",
        "convert_to_MEDS_events",
        "merge_to_MEDS_cohort",
        "extract_code_metadata",
        "finalize_MEDS_metadata",
        "finalize_MEDS_data",
    ]


def test_run_cli_download_child_argv_and_failure_propagates(tmp_path, monkeypatch):
    """The download child gets spec/raw_input_dir/key/do_overwrite; its failure code propagates."""
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


def test_run_cli_registry_mode_stamps_dist_version_and_inherits_name(tmp_path, monkeypatch):
    """An etl:-free spec resolved via the registry gets name + version with zero config:

    dataset_name from the entry-point name, version from sources.dataset_version:dist.
    """
    _install_fake_pkg(tmp_path, monkeypatch, "cli_reg_pkg", {CANONICAL_MESSY_FILENAME: _ETL_FREE_MESSY})
    monkeypatch.setattr(
        "MEDS_extract.run.registry.entry_points",
        lambda group: [_FakeEntryPoint("Fake-DS", "cli_reg_pkg")],
    )
    _stub_run_command(monkeypatch)

    root = tmp_path / "out"
    err = _invoke_cli(tmp_path, monkeypatch, "spec=Fake-DS", f"root_output_dir={root}", "do_download=false")
    assert err.code == 0

    cfg = OmegaConf.load(root / ".meds_extract_run" / "pipeline.yaml")
    assert cfg.etl_metadata.dataset_name == "Fake-DS"
    assert cfg.etl_metadata.dataset_version == "0.9:1.2.3"


def test_run_cli_mapping_dataset_version_follows_selected_key(tmp_path, monkeypatch):
    """A mapping-form sources.dataset_version stamps the SELECTED bucket's version."""
    spec_fp = _write_spec(
        tmp_path,
        """\
sources:
  dataset_version: {dataset: "3.1", demo: "2.2"}
etl:
  dataset_name: Mapped
patients:
  dob: {code: BIRTH, time: null}
""",
    )
    _stub_run_command(monkeypatch)

    root = tmp_path / "out"
    err = _invoke_cli(
        tmp_path, monkeypatch, f"spec={spec_fp}", f"root_output_dir={root}", "key=demo", "do_download=false"
    )
    assert err.code == 0
    cfg = OmegaConf.load(root / ".meds_extract_run" / "pipeline.yaml")
    assert cfg.etl_metadata.dataset_version == "2.2"


def test_run_cli_version_divergence_exits_one(tmp_path, monkeypatch):
    """sources.dataset_version and etl.raw_dataset_version must match when both resolve."""
    spec_fp = _write_spec(
        tmp_path,
        """\
sources:
  dataset_version: "3.1"
etl:
  dataset_name: Diverged
  raw_dataset_version: "9.9"
patients:
  dob: {code: BIRTH, time: null}
""",
    )
    _stub_run_command(monkeypatch)

    root = tmp_path / "out"
    err = _invoke_cli(tmp_path, monkeypatch, f"spec={spec_fp}", f"root_output_dir={root}")
    assert err.code == 1
    assert not (root / ".meds_extract_run").exists()


def test_run_cli_no_version_anywhere_exits_one(tmp_path, monkeypatch):
    """A spec with neither sources.dataset_version nor etl.raw_dataset_version cannot run."""
    spec_fp = _write_spec(tmp_path, "etl: {dataset_name: X}\npatients:\n  dob: {code: BIRTH, time: null}\n")
    _stub_run_command(monkeypatch)

    err = _invoke_cli(tmp_path, monkeypatch, f"spec={spec_fp}", f"root_output_dir={tmp_path / 'out'}")
    assert err.code == 1


def test_run_cli_explicit_dataset_version_override_wins(tmp_path, monkeypatch):
    """``dataset_version=`` beats both the registry stamp and the raw-version fallback."""
    spec_fp = _write_spec(tmp_path, _MINIMAL_MESSY)
    _stub_run_command(monkeypatch)

    root = tmp_path / "out"
    err = _invoke_cli(
        tmp_path,
        monkeypatch,
        f"spec={spec_fp}",
        f"root_output_dir={root}",
        "do_download=false",
        "dataset_version=0.9:custom",
    )
    assert err.code == 0
    cfg = OmegaConf.load(root / ".meds_extract_run" / "pipeline.yaml")
    assert cfg.etl_metadata.dataset_version == "0.9:custom"


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


def test_run_cli_path_mode_omitted_dataset_name_exits_one(tmp_path, monkeypatch):
    """``etl.dataset_name`` is required for path-resolved specs: no registry name to inherit."""
    spec_fp = _write_spec(tmp_path, _ETL_FREE_MESSY)
    _stub_run_command(monkeypatch)

    root = tmp_path / "out"
    err = _invoke_cli(
        tmp_path, monkeypatch, f"spec={spec_fp}", f"root_output_dir={root}", "do_download=false"
    )
    assert err.code == 1
    assert not (root / ".meds_extract_run").exists()


def test_run_cli_unresolvable_spec_exits_one(tmp_path, monkeypatch):
    _stub_run_command(monkeypatch)
    err = _invoke_cli(tmp_path, monkeypatch, "spec=No-Such-Pipeline", f"root_output_dir={tmp_path / 'out'}")
    assert err.code == 1


def test_run_cli_pipeline_failure_code_propagates(tmp_path, monkeypatch):
    """The pipeline child's non-zero exit code becomes the runner's exit code."""
    spec_fp = _write_spec(tmp_path, _MINIMAL_MESSY)
    _stub_run_command(monkeypatch, rc=5)

    err = _invoke_cli(
        tmp_path, monkeypatch, f"spec={spec_fp}", f"root_output_dir={tmp_path / 'out'}", "do_download=false"
    )
    assert err.code == 5
