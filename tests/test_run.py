"""Tests for the ``meds-extract-run`` CLI.

Deliberately small: spec loading, resolution, and version/stamping semantics are
pinned by ``MessyConfig``'s doctests in ``config.py`` (including the registry rung,
via the ``fake_pipeline_registry`` fixture), and the golden end-to-end run over
``example/`` lives in ``tests/test_run_example.py``. The CLI tests here drive the
REAL console script as a subprocess — exactly how users invoke it, with Hydra
behaving as it does in production — over tiny local fixtures. The one in-process
test covers a genuine library seam: ``run_command``'s ``sys.executable -m`` spawn
contract, which requires patching ``subprocess.run`` to observe.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
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
  n_subjects_per_shard: 7
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
    """Invoke the real ``meds-extract-run`` console script from ``tmp_path``.

    Deliberately passes no ``hydra.run.dir`` override and runs with ``cwd=tmp_path``:
    the default run dir (``${output_dir}/.meds_extract_run/hydra_run``) and the
    no-CWD-litter contract are part of what these tests pin.
    """
    return subprocess.run(
        ["meds-extract-run", *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )


def _write_tiny_spec(tmp_path: Path) -> Path:
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "patients.csv").write_text(_TINY_CSV)
    spec_fp = tmp_path / "spec.yaml"
    spec_fp.write_text(_TINY_MESSY.format(mirror=mirror))
    return spec_fp


def test_run_command_spawns_via_sys_executable_dash_m(monkeypatch):
    """The child is spawned as ``sys.executable -m <module> <args...>`` — pinned to this interpreter's
    environment with no console-script PATH resolution — and the child's exit code propagates verbatim."""
    seen: dict[str, object] = {}

    def fake_run(argv, check):
        seen["argv"] = argv
        return subprocess.CompletedProcess(args=argv, returncode=3)

    monkeypatch.setattr(run_cli.subprocess, "run", fake_run)
    rc = run_cli.run_command(["MEDS_transforms.runner", "cfg.yaml"])

    assert rc == 3
    assert seen["argv"] == [sys.executable, "-m", "MEDS_transforms.runner", "cfg.yaml"]


def test_run_cli_full_flow(tmp_path):
    """The real console script end-to-end over a tiny local spec: downloads from the
    fsspec mirror into the default destination, synthesizes a fully inlined pipeline
    config under the work dir, runs the real pipeline, and exits 0.

    Invoked with an explicit relative spec (``./spec.yaml``) and a relative
    ``output_dir`` from ``tmp_path``, pinning that ``hydra.job.chdir=false`` keeps
    relative paths anchored to the invoking CWD — and that the entire run writes
    NOTHING outside the output tree (every Hydra run dir included)."""
    spec_fp = _write_tiny_spec(tmp_path)
    out_dir = tmp_path / "meds"

    result = _run_cli(tmp_path, "spec=./spec.yaml", "output_dir=meds")
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    # No CWD litter: the only new entry beside the fixtures is the output tree itself.
    assert {p.name for p in tmp_path.iterdir()} == {"mirror", "spec.yaml", "meds"}
    # The CLIs' Hydra run dirs live inside the output tree, not the CWD.
    assert (out_dir / ".meds_extract_run" / "hydra_run" / "cli.log").exists()
    assert (out_dir / ".meds_extract_run" / "hydra_download" / ".hydra").is_dir()

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
    assert cfg.MESSY_config_fp == str(spec_fp)
    assert cfg.output_dir == str(out_dir)
    stages = OmegaConf.to_container(cfg.stages)
    assert stages[0] == "convert_to_parquet"
    assert stages[1]["split_and_shard_subjects"]["n_subjects_per_shard"] == 7
    assert [s if isinstance(s, str) else next(iter(s)) for s in stages] == list(EtlConfig.DEFAULT_PIPELINE)


@pytest.mark.parametrize("dirname", ["comma,dir", "eq=dir"])
def test_run_cli_paths_with_hydra_metacharacters(tmp_path, dirname):
    """The full flow succeeds when every path (spec, output tree) lives under a
    directory whose name contains a Hydra override-grammar metacharacter: a bare
    ``,`` parses as a choice sweep and a bare ``=`` splits the override, so the
    runner must quote the path-valued overrides it synthesizes for the download
    child. The *parent's* args are quoted here the way a user would quote them —
    Hydra's single-quote syntax — which fixes only the parent's own parse; what
    this pins is that the child spawn survives too."""
    base = tmp_path / dirname
    base.mkdir()
    spec_fp = _write_tiny_spec(base)
    out_dir = base / "meds"

    result = _run_cli(base, f"spec='{spec_fp}'", f"output_dir='{out_dir}'")
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    # The download child received the metacharacter paths intact and staged the data.
    assert (out_dir / ".meds_extract_run" / "raw_input" / "patients.csv").exists()
    # The pipeline ran to completion over them as well.
    assert (out_dir / "data" / "train" / "0.parquet").exists()
    assert (out_dir / "metadata" / "codes.parquet").exists()


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
    omitting dataset_name; implausible flag combinations (input_dir with a
    download_key; a download-only knob on a download-free run). The only thing a
    failed run may create is its own Hydra run dir (log + config snapshot) inside
    the target output tree."""
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

    result = _run_cli(
        tmp_path,
        f"spec={spec_fp}",
        f"output_dir={tmp_path / 'o4'}",
        "download_key=null",
        f"input_dir={tmp_path}",
        "download_concurrency=8",
    )
    assert result.returncode == 1
    assert "no effect with download_key=null" in result.stdout + result.stderr

    for d in ("o1", "o2", "o3", "o4"):
        # No data, no staging — at most the run's own Hydra log dir under the target.
        created = {p.name for p in (tmp_path / d).iterdir()} if (tmp_path / d).exists() else set()
        assert created <= {".meds_extract_run"}, created
    assert not (tmp_path / "outputs").exists()  # no Hydra litter in the CWD


def test_run_cli_bare_and_pathless_invocations(tmp_path):
    """A bare invocation prints a one-line usage and creates NOTHING; a bare relative
    spec (no ``./``) is refused with the explicit-path hint; a URL output_dir is
    rejected before Hydra can materialize a literal ``s3:/`` directory."""
    result = _run_cli(tmp_path)
    assert result.returncode == 1
    assert "missing required argument" in result.stderr
    assert not any(tmp_path.iterdir())

    _write_tiny_spec(tmp_path)
    result = _run_cli(tmp_path, "spec=spec.yaml", "output_dir=out")
    assert result.returncode == 1
    assert "./spec.yaml" in result.stdout + result.stderr

    result = _run_cli(tmp_path, "spec=./spec.yaml", "output_dir=s3://bucket/raw")
    assert result.returncode == 1
    assert "must be a local filesystem path" in result.stderr
    assert not any(p.name.startswith("s3:") for p in tmp_path.iterdir())


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


def test_run_cli_forwards_child_flags_end_to_end(tmp_path):
    """The passthrough knobs reach the real children and are accepted by them.

    Unit-level spelling is pinned by the ``download_argv`` / ``pipeline_argv`` doctests;
    what this adds is that the spellings are the ones the *children* actually parse. A
    wrong flag name fails loudly here — argparse rejects an unknown option on the
    pipeline runner, and Hydra rejects an unknown key on the download CLI — so a
    successful run is the assertion.

    ``stage_runner_fp`` carries a real ``parallelize`` block (the flag's whole point),
    ``download_do_overwrite`` routes to the download child as ``do_overwrite=``, and
    ``overrides`` carries ``seed``, which is a genuine pipeline-config key.
    """
    spec_fp = _write_tiny_spec(tmp_path)
    out_dir = tmp_path / "meds"

    stage_runner_fp = tmp_path / "stage_runner.yaml"
    stage_runner_fp.write_text("parallelize:\n  n_workers: 1\n")

    result = _run_cli(
        tmp_path,
        f"spec={spec_fp}",
        f"output_dir={out_dir}",
        f"stage_runner_fp={stage_runner_fp}",
        "download_do_overwrite=True",
        "download_concurrency=2",
        "download_continue_on_error=True",
        "overrides=['seed=2']",
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert (out_dir / "data" / "train" / "0.parquet").exists()

    # The flags reached the children rather than being silently dropped.
    combined = result.stdout + result.stderr
    assert "--stage_runner_fp" in combined
    assert "do_overwrite=True" in combined
    assert "concurrency=2" in combined
    assert "seed=2" in combined


def test_run_config_rejects_download_knobs_on_download_free_runs(tmp_path):
    """``download_key=null`` spawns no download child, so a non-default download-only knob could only be
    silently ignored — ``RunConfig`` rejects the combination instead, naming the offending knob."""
    base = {
        "spec": "X",
        "output_dir": str(tmp_path),
        "download_key": None,
        "input_dir": str(tmp_path),
    }
    run_cli.RunConfig(**base)  # the download-free shape itself is valid

    for knob in (
        {"download_do_overwrite": True},
        {"download_concurrency": 8},
        {"download_continue_on_error": True},
    ):
        with pytest.raises(ValueError, match="no effect with download_key=null"):
            run_cli.RunConfig(**base, **knob)
