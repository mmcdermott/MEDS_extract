"""End-to-end tests for extension stages (``etl.pre_stages`` / ``etl.post_stages``) and ``skip_stages``.

Extension stages come from *other* installed packages, so these tests install one: a
tiny package of two stages plus a ``.dist-info`` directory whose ``entry_points.txt``
registers them in the ``MEDS_transforms.stages`` group, written to a temporary
directory that the real ``meds-extract-run`` subprocess (and every stage process it
spawns) gets on ``PYTHONPATH``. Stage discovery is ``importlib.metadata``, exactly as
for a pip-installed package.

- ``tiny_pre`` (a pre-stage) pivots a long ``(key, field, value)`` table into the wide
  table the spec reads. The test feeds the ``example/`` dataset with
  ``lab_descriptions.csv`` replaced by its long form, so the golden outputs are only
  reproduced if both stages that read raw sources (``convert_to_parquet``, which reads
  it as a join target, and ``extract_code_metadata``, which reads it as a ``_metadata``
  table) read the staged table.
- ``tiny_post`` (a post-stage) edits the finished ``metadata/`` in place: it appends its
  ``description_suffix`` option to every description (idempotently) and records what it
  saw in ``dataset.json``, keeping the keys ``finalize_MEDS_metadata`` wrote.

Spec-level validation (shapes, names, ``skip_stages`` rules, stage kinds) is doctested on
``EtlConfig``; the done-marker rules on ``reconcile_done_markers``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import polars as pl
import pytest
from omegaconf import OmegaConf
from polars.testing import assert_frame_equal

from MEDS_extract.config import EtlConfig

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example"
EXPECTED = EXAMPLE_DIR / "expected_output"

_TINY_PRE = '''\
"""Pre-stage: pivot a long (key, field, value) csv into the wide csv the spec reads."""

from pathlib import Path

import polars as pl
from MEDS_transforms.stages import Stage


@Stage.register(is_metadata=False, default_config={"source": None, "target": None, "key": None})
def main(cfg):
    """Reads ``<input_dir>/<source>.csv``; writes ``<output_dir>/<target>.csv``."""
    if cfg.worker != 0:
        return
    long = pl.read_csv(Path(cfg.input_dir) / f"{cfg.stage_cfg.source}.csv", infer_schema=False)
    wide = long.pivot(on="field", index=cfg.stage_cfg.key, values="value")
    out_fp = Path(cfg.stage_cfg.output_dir) / f"{cfg.stage_cfg.target}.csv"
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    wide.write_csv(out_fp)
'''

_TINY_POST = '''\
"""Post-stage: suffix every code description, in place, and record the run in dataset.json."""

import json
from pathlib import Path

import polars as pl
from MEDS_transforms.stages import Stage


@Stage.register(is_metadata=True, default_config={"description_suffix": ""})
def main(cfg):
    """Edits ``codes.parquet`` and ``dataset.json`` in the cohort's ``metadata/``."""
    if cfg.worker != 0:
        return
    metadata_in = Path(cfg.stage_cfg.metadata_input_dir)
    metadata_out = Path(cfg.stage_cfg.reducer_output_dir)
    suffix = cfg.stage_cfg.description_suffix

    desc = pl.col("description")
    codes = pl.read_parquet(metadata_in / "codes.parquet").with_columns(
        pl.when(desc.str.ends_with(suffix)).then(desc).otherwise(desc + suffix).alias("description")
    )
    tmp_fp = metadata_out / ".codes.parquet.tmp"
    codes.write_parquet(tmp_fp)
    tmp_fp.replace(metadata_out / "codes.parquet")

    dataset = json.loads((metadata_in / "dataset.json").read_text())
    dataset["tiny_post"] = {
        "dataset_name": cfg.etl_metadata.dataset_name,
        "n_data_files": len(list(Path(cfg.stage_cfg.data_input_dir).rglob("*.parquet"))),
    }
    (metadata_out / "dataset.json").write_text(json.dumps(dataset))
'''

_ENTRY_POINTS = """\
[MEDS_transforms.stages]
tiny_pre = meds_extract_test_stages.tiny_pre:main
tiny_post = meds_extract_test_stages.tiny_post:main
"""

_SUFFIX = " [tiny_post]"

_EXTENSION_ETL = f"""\
etl:
  license: CC0-1.0
  pre_stages:
    - tiny_pre:
        source: lab_descriptions_long
        target: lab_descriptions
        key: test_name
  post_stages:
    - tiny_post:
        description_suffix: "{_SUFFIX}"
"""


@pytest.fixture(scope="module")
def stages_env(tmp_path_factory) -> dict[str, str]:
    """An environment in which ``tiny_pre`` and ``tiny_post`` are installed, registered stages."""
    site = tmp_path_factory.mktemp("site")
    pkg = site / "meds_extract_test_stages"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "tiny_pre.py").write_text(_TINY_PRE)
    (pkg / "tiny_post.py").write_text(_TINY_POST)
    dist_info = site / "meds_extract_test_stages-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: meds-extract-test-stages\nVersion: 0.0.0\n"
    )
    (dist_info / "entry_points.txt").write_text(_ENTRY_POINTS)
    return {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(site), os.getenv("PYTHONPATH")]))}


@pytest.fixture
def example_run(tmp_path):
    """The example spec with the two extension stages, and its raw input in long form.

    ``lab_descriptions.csv`` is replaced by ``lab_descriptions_long.csv``, which only
    ``tiny_pre`` can turn back into the table the spec reads.
    """
    spec_fp = tmp_path / "messy.yaml"
    spec_fp.write_text((EXAMPLE_DIR / "messy.yaml").read_text().replace("etl:\n", _EXTENSION_ETL, 1))

    raw = tmp_path / "raw_input"
    shutil.copytree(EXAMPLE_DIR / "raw_data", raw)
    wide = pl.read_csv(raw / "lab_descriptions.csv", infer_schema=False)
    wide.unpivot(index="test_name", variable_name="field", value_name="value").write_csv(
        raw / "lab_descriptions_long.csv"
    )
    (raw / "lab_descriptions.csv").unlink()
    return spec_fp, raw, tmp_path / "meds"


def _run(env: dict[str, str], spec_fp: Path, raw: Path, out_dir: Path, *args: str) -> None:
    result = subprocess.run(
        [
            "meds-extract-run",
            f"spec={spec_fp}",
            f"output_dir={out_dir}",
            "do_download=false",
            f"input_dir={raw}",
            *args,
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=600,
    )
    stage_logs = "\n".join(
        f"--- {fp}\n{fp.read_text(errors='replace')[-3000:]}"
        for fp in sorted(out_dir.glob("*/.logs/*/*.log"))
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}\n{stage_logs}"


def _assert_cohort(out_dir: Path, *, suffix: str) -> None:
    """``data/`` matches the goldens; ``codes.parquet`` matches them with ``suffix`` on each description."""
    for want_fp in sorted((EXPECTED / "data").rglob("*.parquet")):
        got_fp = out_dir / want_fp.relative_to(EXPECTED)
        assert_frame_equal(pl.read_parquet(got_fp), pl.read_parquet(want_fp))
    want_codes = pl.read_parquet(EXPECTED / "metadata" / "codes.parquet").with_columns(
        pl.col("description") + suffix
    )
    got_codes = pl.read_parquet(out_dir / "metadata" / "codes.parquet")
    assert_frame_equal(got_codes, want_codes, check_row_order=False)


def _stage_names(out_dir: Path) -> list[str]:
    stages = OmegaConf.to_container(OmegaConf.load(out_dir / ".meds_extract_run" / "pipeline.yaml").stages)
    return [s if isinstance(s, str) else next(iter(s)) for s in stages]


def test_extension_stages_end_to_end(stages_env, example_run):
    """A pre-stage stages a reshaped raw table that both raw readers use, and a post-stage edits the finished
    cohort in place, all from one ``meds-extract-run``.

    Re-running is a no-op, and deleting the post-stage's done marker re-runs just that stage, which leaves the
    same cohort.
    """
    spec_fp, raw, out_dir = example_run
    _run(stages_env, spec_fp, raw, out_dir)

    assert _stage_names(out_dir) == ["tiny_pre", *EtlConfig.DEFAULT_PIPELINE, "tiny_post"]
    pipeline = OmegaConf.load(out_dir / ".meds_extract_run" / "pipeline.yaml")
    assert "${" not in (out_dir / ".meds_extract_run" / "pipeline.yaml").read_text()
    assert pipeline.stages[1].convert_to_parquet.input_overlay_dirs == [str(out_dir / "tiny_pre")]
    assert pipeline.stages[-1].tiny_post.reducer_output_dir == str(out_dir / "metadata")
    # The staged table is the pre-stage's only output, beside its logs; the raw input is untouched.
    assert sorted(p.name for p in (out_dir / "tiny_pre").iterdir()) == [".logs", "lab_descriptions.csv"]
    assert not (raw / "lab_descriptions.csv").exists()

    _assert_cohort(out_dir, suffix=_SUFFIX)
    dataset = json.loads((out_dir / "metadata" / "dataset.json").read_text())
    assert dataset["dataset_name"] == "MEDS_extract_example"
    assert dataset["tiny_post"] == {"dataset_name": "MEDS_extract_example", "n_data_files": 3}
    # finalize_MEDS_metadata's fields survive the post-stage's rewrite of dataset.json.
    assert dataset["license"] == "CC0-1.0"
    assert dataset["other_extension_columns"] == ["source_block"]

    # A finished run re-runs nothing.
    _run(stages_env, spec_fp, raw, out_dir)
    _assert_cohort(out_dir, suffix=_SUFFIX)

    # Retrying one extension stage: delete its done marker. The runner's global marker is dropped for
    # it, and the stage re-runs on its own output.
    (out_dir / ".logs" / "tiny_post.done").unlink()
    _run(stages_env, spec_fp, raw, out_dir)
    assert (out_dir / ".logs" / "tiny_post.done").exists()
    _assert_cohort(out_dir, suffix=_SUFFIX)


def test_skip_stages_then_complete_the_run(stages_env, example_run):
    """``skip_stages=[tiny_post]`` gives the plain cohort; the same command without it then runs only the
    skipped post-stage on that cohort.

    With ``do_cleanup_intermediates``, extension stage dirs go too.
    """
    spec_fp, raw, out_dir = example_run

    _run(stages_env, spec_fp, raw, out_dir, "skip_stages=[tiny_post]")
    assert _stage_names(out_dir) == ["tiny_pre", *EtlConfig.DEFAULT_PIPELINE]
    _assert_cohort(out_dir, suffix="")
    assert "tiny_post" not in json.loads((out_dir / "metadata" / "dataset.json").read_text())
    canonical_done = (out_dir / ".logs" / "convert_to_parquet.done").stat().st_mtime_ns

    _run(stages_env, spec_fp, raw, out_dir, "do_cleanup_intermediates=true")
    assert (out_dir / ".logs" / "convert_to_parquet.done").stat().st_mtime_ns == canonical_done
    _assert_cohort(out_dir, suffix=_SUFFIX)
    assert {p.name for p in out_dir.iterdir()} == {".logs", ".meds_extract_run", "data", "metadata"}


def test_extension_stage_errors_fail_before_any_work(stages_env, example_run):
    """A spec naming an uninstalled stage, or a ``skip_stages`` naming a canonical stage, exits 1 before the
    pipeline starts.

    Skipping the uninstalled stage lets the run proceed.
    """
    spec_fp, raw, out_dir = example_run
    spec_fp.write_text(
        spec_fp.read_text().replace("    - tiny_post:", "    - no_such_stage\n    - tiny_post:")
    )

    for args, message in [
        ((), "['no_such_stage'] that no installed package registers"),
        (("skip_stages=[no_such_stage,convert_to_parquet]",), "canonical stage(s) ['convert_to_parquet']"),
    ]:
        result = subprocess.run(
            ["meds-extract-run", f"spec={spec_fp}", f"output_dir={out_dir}", "do_download=false",
             f"input_dir={raw}", *args],
            capture_output=True, text=True, env=stages_env, check=False,
        )  # fmt: skip
        assert result.returncode == 1
        assert message in result.stdout + result.stderr
        assert not (out_dir / ".meds_extract_run" / "pipeline.yaml").exists()

    _run(stages_env, spec_fp, raw, out_dir, "skip_stages=[no_such_stage]")
    _assert_cohort(out_dir, suffix=_SUFFIX)


@pytest.mark.parallelism
def test_extension_stages_under_parallel_workers(stages_env, example_run):
    """With every stage fanned out to two workers, the extension stages (which work only in worker 0) still
    leave the same cohort."""
    pytest.importorskip("hydra_plugins.hydra_joblib_launcher", reason="needs the 'local_parallelism' extra")
    spec_fp, raw, out_dir = example_run
    stage_runner_fp = out_dir.parent / "stage_runner.yaml"
    stage_runner_fp.write_text("parallelize:\n  n_workers: 2\n  launcher: joblib\n")

    _run(stages_env, spec_fp, raw, out_dir, f"stage_runner_fp={stage_runner_fp}")
    # The post-stage really ran as two workers: one Hydra job dir per worker.
    log_dir = out_dir / "tiny_post" / ".logs"
    assert {p.name for p in log_dir.iterdir() if p.is_dir() and not p.name.startswith(".")} == {"0", "1"}
    _assert_cohort(out_dir, suffix=_SUFFIX)
