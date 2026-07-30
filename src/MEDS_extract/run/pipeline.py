"""Pipeline-config synthesis and in-process MEDS-transforms pipeline invocation.

The ``etl:`` block of a MESSY spec lists stages; this module turns that into a full
MEDS-transforms pipeline config (with every value **inlined** — no ``${oc.env:...}``
indirection, so the written file is self-contained, diffable provenance) and runs it
through ``MEDS_transforms.runner.main`` **in-process**.

The PATH-healing story (mmcdermott/MEDS_transforms#398): the upstream runner's
``run_stage`` executes each stage as a *shell command* spawning the bare console
script name ``MEDS_transform-stage``, which the shell resolves via the child's
``PATH``. When the invoking environment isn't "activated" (cron, Slurm, Makefiles,
``/path/to/venv/bin/meds-extract-run``), the venv's scripts directory is absent from
``PATH`` and the spawn dies with ``command not found`` — a failure class found
independently in three downstream repos. ``run_stage`` does expose an injectable
``runner_fn`` seam, but ``MEDS_transforms.runner.main`` does **not** forward it
(``main`` calls ``run_stage`` without ``runner_fn``), so the seam is unreachable
from the public entry point. Rather than re-implementing ``main``'s orchestration
loop (done-file resume, logging, parallelization defaults) just to reach the seam,
:func:`run_pipeline` heals at the environment level: it prepends this interpreter's
scripts directory (``sysconfig.get_path("scripts")`` — exactly what ``source
.../activate`` prepends) to ``os.environ["PATH"]`` for the duration of the call, so
every ``shell=True`` child inherits an activation-equivalent view and resolves
``MEDS_transform-stage`` from this environment. If MEDS_transforms#398 lands an
upstream fix, this healing becomes a harmless no-op.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import sysconfig
from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf

if TYPE_CHECKING:
    from pathlib import Path

    from ..config import EtlConfig

logger = logging.getLogger(__name__)

# The bare console-script name MEDS_transforms' runner spawns per stage (via
# ``shell=True``), and therefore the executable whose resolvability we must
# guarantee before starting a multi-stage run.
_STAGE_SCRIPT = "MEDS_transform-stage"


def activation_equivalent_path(path: str | None = None, *, scripts_dir: str | None = None) -> str:
    """Return ``PATH`` with this environment's scripts directory prepended.

    Prepending ``sysconfig.get_path("scripts")`` (the venv's ``bin/``, or
    ``Scripts\\`` on Windows) is exactly what ``source .../activate`` does to
    ``PATH`` — no bespoke resolution order is invented beyond the standard,
    documented one.

    Args:
        path: The ``PATH`` value to prepend to. Defaults to ``os.environ["PATH"]``.
        scripts_dir: Override of the scripts directory (for tests). Defaults to
            ``sysconfig.get_path("scripts")``.

    Examples:
        >>> activation_equivalent_path("/usr/bin", scripts_dir="/my/venv/bin")
        '/my/venv/bin:/usr/bin'

        Idempotent — an already-activated ``PATH`` is not double-prepended — and
        empty entries (which would mean "current directory" on POSIX) are dropped:

        >>> activation_equivalent_path("/my/venv/bin:/usr/bin:", scripts_dir="/my/venv/bin")
        '/my/venv/bin:/usr/bin'
    """
    if path is None:
        path = os.environ.get("PATH", "")
    if scripts_dir is None:
        scripts_dir = sysconfig.get_path("scripts")
    parts = [p for p in path.split(os.pathsep) if p and p != scripts_dir]
    return os.pathsep.join([scripts_dir, *parts])


def synthesize_pipeline_config(
    etl: EtlConfig,
    *,
    event_conversion_config_fp: Path,
    input_dir: Path,
    output_dir: Path,
    dataset_version: str,
) -> dict[str, Any]:
    """Build the MEDS-transforms pipeline config dict from an ``etl:`` block.

    Every value is **inlined as a resolved literal** — where cloned dataset-package
    runners set ``DATASET_NAME`` / ``DATASET_VERSION`` / ``EVENT_CONVERSION_CONFIG_FP``
    env vars for the pipeline YAML to interpolate, the synthesized config carries the
    final values directly. The written file is then self-contained: re-runnable and
    diffable without reconstructing the environment that produced it, and immune to
    the child-env plumbing bugs the env-var pattern invited.

    Examples:
        >>> from MEDS_extract.config import EtlConfig
        >>> etl = EtlConfig.parse({
        ...     "dataset_name": "Example",
        ...     "raw_dataset_version": "3.1",
        ...     "pipeline": [{"shard_events": {"row_chunksize": 2}}, "merge_to_MEDS_cohort"],
        ... })
        >>> cfg = synthesize_pipeline_config(
        ...     etl,
        ...     event_conversion_config_fp=Path("/specs/messy.yaml"),
        ...     input_dir=Path("/data/raw_input"),
        ...     output_dir=Path("/data/MEDS_output"),
        ...     dataset_version="3.1:1.0.0",
        ... )
        >>> print(OmegaConf.to_yaml(OmegaConf.create(cfg)).strip())
        etl_metadata:
          dataset_name: Example
          dataset_version: 3.1:1.0.0
        event_conversion_config_fp: /specs/messy.yaml
        input_dir: /data/raw_input
        output_dir: /data/MEDS_output
        shards_map_fp: /data/MEDS_output/metadata/.shards.json
        stages:
        - shard_events:
            row_chunksize: 2
        - merge_to_MEDS_cohort
    """
    return {
        "etl_metadata": {
            "dataset_name": etl.dataset_name,
            "dataset_version": dataset_version,
        },
        "event_conversion_config_fp": str(event_conversion_config_fp),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "shards_map_fp": f"{output_dir}/metadata/.shards.json",
        "stages": etl.stages_container(),
    }


def write_pipeline_config(cfg: dict[str, Any], fp: Path) -> None:
    """Serialize a synthesized pipeline config to ``fp`` (parents created)."""
    fp.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg), fp)
    logger.info(f"Wrote synthesized pipeline config to {fp}")


def run_pipeline(pipeline_config_fp: Path) -> int:
    """Run a pipeline config through ``MEDS_transforms.runner.main``, in-process.

    ``os.environ["PATH"]`` is upgraded to the activation-equivalent view (see module
    docstring / :func:`activation_equivalent_path`) for the duration of the call and
    restored afterwards, so the runner's ``shell=True`` per-stage children resolve
    the bare ``MEDS_transform-stage`` name exactly as an activated environment would
    (mmcdermott/MEDS_transforms#398). Resolvability is checked *before* starting so a
    broken environment fails in seconds with an actionable message, not mid-pipeline.

    Coupling note: this intentionally calls the upstream runner's public
    ``main(argv)`` rather than its internal ``run_stage(..., runner_fn=...)`` seam —
    ``main`` does not forward ``runner_fn``, and duplicating its orchestration loop
    (done-file resume, log layout, parallelization defaults) here would couple us to
    far more upstream internals than one env-var tweak does.

    Returns:
        The runner's exit code (``0`` on success). Stage failures raise (the
        upstream runner raises ``ValueError`` naming the stage); callers map
        exceptions to a non-zero process exit.
    """
    from MEDS_transforms.runner import main as pipeline_main

    healed_path = activation_equivalent_path()
    if shutil.which(_STAGE_SCRIPT, path=healed_path) is None:
        raise FileNotFoundError(
            f"{_STAGE_SCRIPT!r} not found in this environment's scripts directory "
            f"({sysconfig.get_path('scripts')}) or on PATH. The MEDS-transforms pipeline runner "
            f"spawns it per stage; reinstall MEDS-transforms in this environment "
            f"({sys.executable})."
        )

    prior_path = os.environ.get("PATH")
    os.environ["PATH"] = healed_path
    try:
        return pipeline_main([str(pipeline_config_fp)])
    finally:
        if prior_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = prior_path
