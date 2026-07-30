"""Unit tests for the ``meds-extract-run`` layer (registry, synthesis, CLI wiring).

Everything here is fast and offline: the registry ladder is exercised with synthetic
packages + monkeypatched entry points, and the CLI tests stub out the actual
pipeline invocation (``run_pipeline``) so no MEDS-transforms stages are spawned. The
real end-to-end run over ``example/`` lives in ``tests/test_run_example.py``.
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
  pipeline: [shard_events]
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
    assert (rs.dist_name, rs.dist_version) == ("fake-ds-dist", "1.2.3")


def test_resolve_spec_registered_name_without_dist(tmp_path, monkeypatch):
    """An entry point with no attached distribution still resolves; provenance is None."""
    _install_fake_pkg(tmp_path, monkeypatch, "distless_pkg", {CANONICAL_MESSY_FILENAME: _MINIMAL_MESSY})
    monkeypatch.setattr(
        "MEDS_extract.run.registry.entry_points",
        lambda group: [_FakeEntryPoint("Dist-Less", "distless_pkg", dist=None)],
    )

    rs = resolve_spec("Dist-Less")
    assert rs.origin == "registry"
    assert rs.dist_name is None and rs.dist_version is None


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


# ── run_pipeline: PATH healing + missing-script teeth ──────────────────────────────


def test_run_pipeline_heals_path_and_restores_it(tmp_path, monkeypatch):
    """The in-process runner sees an activation-equivalent PATH; the caller's PATH survives.

    This is the #398 healing seam: the upstream runner's ``shell=True`` children
    inherit ``os.environ["PATH"]``, so upgrading it for the duration of the call is
    what makes the bare ``MEDS_transform-stage`` spawn resolve without activation.
    """
    import sysconfig

    from MEDS_extract.run.pipeline import run_pipeline

    scripts_dir = sysconfig.get_path("scripts")
    stripped = ":".join(p for p in (os.environ.get("PATH", "").split(":")) if p and p != scripts_dir)
    monkeypatch.setenv("PATH", stripped)

    seen: dict[str, str] = {}

    def fake_main(argv):
        seen["path"] = os.environ["PATH"]
        seen["argv"] = argv
        return 0

    monkeypatch.setattr("MEDS_transforms.runner.main", fake_main)
    # The real script may legitimately be missing from the stripped PATH view only if
    # healing works — which is exactly what the pre-flight which() relies on.
    rc = run_pipeline(tmp_path / "pipeline.yaml")

    assert rc == 0
    assert seen["argv"] == [str(tmp_path / "pipeline.yaml")]
    assert seen["path"].split(":")[0] == scripts_dir
    assert os.environ["PATH"] == stripped  # restored after the call


def test_run_pipeline_missing_stage_script_fails_fast(tmp_path, monkeypatch):
    """If ``MEDS_transform-stage`` is unresolvable even after healing, fail before running."""
    from MEDS_extract.run import pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod.shutil, "which", lambda *a, **k: None)
    with pytest.raises(FileNotFoundError, match="MEDS_transform-stage"):
        pipeline_mod.run_pipeline(tmp_path / "pipeline.yaml")


# ── CLI wiring: synthesis, stamping, exit codes (run_pipeline stubbed) ─────────────


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


def _write_spec(tmp_path: Path, body: str) -> Path:
    spec_fp = tmp_path / "spec.yaml"
    spec_fp.write_text(body)
    return spec_fp


_SPEC_WITH_SOURCES = """\
sources:
  dataset:
    - type: fsspec
      root: {mirror}
etl:
  dataset_name: CLITest
  raw_dataset_version: "2.0"
  pipeline:
    - shard_events:
        row_chunksize: 5
    - finalize_MEDS_data
patients:
  dob: {{code: BIRTH, time: null}}
"""


def test_run_cli_downloads_synthesizes_inlined_config_and_exits_zero(tmp_path, monkeypatch):
    """The full CLI flow minus the pipeline itself: in-process download staging, a fully
    inlined synthesized pipeline config at the documented location, and exit 0."""
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "patients.csv").write_text("patient_id,dob\n1,2000-01-01\n")
    spec_fp = _write_spec(tmp_path, _SPEC_WITH_SOURCES.format(mirror=mirror))
    root = tmp_path / "out"

    invoked: list[Path] = []
    monkeypatch.setattr("MEDS_extract.run.cli.run_pipeline", lambda fp: invoked.append(fp) or 0)

    err = _invoke_cli(tmp_path, monkeypatch, f"spec={spec_fp}", f"root_output_dir={root}")
    assert err.code == 0

    # Download staged in-process into the documented default location.
    assert (root / "raw_input" / "patients.csv").exists()

    # Synthesized config: documented location, every value inlined (no env-var or
    # interpolation indirection), path-mode version stamp = raw_dataset_version alone.
    pipeline_fp = root / ".meds_extract_run" / "pipeline.yaml"
    assert invoked == [pipeline_fp]
    text = pipeline_fp.read_text()
    assert "${" not in text and "oc.env" not in text
    cfg = OmegaConf.load(pipeline_fp)
    assert cfg.etl_metadata.dataset_name == "CLITest"
    assert cfg.etl_metadata.dataset_version == "2.0"
    assert cfg.event_conversion_config_fp == str(spec_fp)
    assert cfg.input_dir == str(root / "raw_input")
    assert cfg.output_dir == str(root / "MEDS_output")
    assert cfg.shards_map_fp == f"{root / 'MEDS_output'}/metadata/.shards.json"
    assert OmegaConf.to_container(cfg.stages) == [
        {"shard_events": {"row_chunksize": 5}},
        "finalize_MEDS_data",
    ]


def test_run_cli_registry_mode_stamps_dist_version(tmp_path, monkeypatch):
    """A registry-resolved spec stamps ``{raw_dataset_version}:{dist version}``."""
    _install_fake_pkg(tmp_path, monkeypatch, "cli_reg_pkg", {CANONICAL_MESSY_FILENAME: _MINIMAL_MESSY})
    monkeypatch.setattr(
        "MEDS_extract.run.registry.entry_points",
        lambda group: [_FakeEntryPoint("Fake-DS", "cli_reg_pkg")],
    )
    monkeypatch.setattr("MEDS_extract.run.cli.run_pipeline", lambda fp: 0)

    root = tmp_path / "out"
    err = _invoke_cli(tmp_path, monkeypatch, "spec=Fake-DS", f"root_output_dir={root}", "do_download=false")
    assert err.code == 0

    cfg = OmegaConf.load(root / ".meds_extract_run" / "pipeline.yaml")
    assert cfg.etl_metadata.dataset_name == "FakeDS"
    assert cfg.etl_metadata.dataset_version == "0.9:1.2.3"


def test_run_cli_explicit_dataset_version_override_wins(tmp_path, monkeypatch):
    """``dataset_version=`` beats both the registry stamp and the raw-version fallback."""
    spec_fp = _write_spec(tmp_path, _MINIMAL_MESSY)
    monkeypatch.setattr("MEDS_extract.run.cli.run_pipeline", lambda fp: 0)

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
    monkeypatch.setattr("MEDS_extract.run.cli.run_pipeline", lambda fp: 0)

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


def test_run_cli_spec_without_etl_block_exits_one(tmp_path, monkeypatch):
    """A MESSY file with no ``etl:`` block cannot drive the runner: exit 1, no synthesis."""
    spec_fp = _write_spec(tmp_path, "patients:\n  dob: {code: BIRTH, time: null}\n")
    root = tmp_path / "out"

    err = _invoke_cli(tmp_path, monkeypatch, f"spec={spec_fp}", f"root_output_dir={root}")
    assert err.code == 1
    assert not (root / ".meds_extract_run").exists()


def test_run_cli_unresolvable_spec_exits_one(tmp_path, monkeypatch):
    err = _invoke_cli(tmp_path, monkeypatch, "spec=No-Such-Pipeline", f"root_output_dir={tmp_path / 'out'}")
    assert err.code == 1


def test_run_cli_bad_sources_key_exits_one(tmp_path, monkeypatch):
    """A typo'd ``key=`` is a config error surfaced before any pipeline synthesis."""
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    spec_fp = _write_spec(tmp_path, _SPEC_WITH_SOURCES.format(mirror=mirror))
    root = tmp_path / "out"

    err = _invoke_cli(tmp_path, monkeypatch, f"spec={spec_fp}", f"root_output_dir={root}", "key=dataste")
    assert err.code == 1
    assert not (root / ".meds_extract_run").exists()


def test_run_cli_pipeline_failure_exits_one(tmp_path, monkeypatch):
    """A stage failure inside the (stubbed) pipeline runner maps to exit code 1."""
    spec_fp = _write_spec(tmp_path, _MINIMAL_MESSY)

    def boom(fp):
        raise ValueError("Stage shard_events failed via ...")

    monkeypatch.setattr("MEDS_extract.run.cli.run_pipeline", boom)
    err = _invoke_cli(
        tmp_path, monkeypatch, f"spec={spec_fp}", f"root_output_dir={tmp_path / 'out'}", "do_download=false"
    )
    assert err.code == 1
