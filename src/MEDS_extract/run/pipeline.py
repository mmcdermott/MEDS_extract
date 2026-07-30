"""Pipeline-config synthesis and healed-PATH subprocess spawning for ``meds-extract-run``.

The runner shells out twice — ``meds-extract-download`` for the sources stage and
``MEDS_transform-pipeline`` for the extraction pipeline — by explicit design:
forwarding through subprocesses keeps this CLI a thin orchestrator over the two
public entry points (in-module invocation modes may come later, upstream). The
``etl:`` block of a MESSY spec is turned into a full MEDS-transforms pipeline config
with every value **inlined** (no ``${oc.env:...}`` indirection), so the written file
is self-contained, diffable provenance.

The PATH-healing story (mmcdermott/MEDS_transforms#398): console-script children —
both the ones this module spawns and the bare ``MEDS_transform-stage`` commands the
pipeline runner spawns per stage — resolve via the child's ``PATH``. When the
invoking environment isn't "activated" (cron, Slurm, Makefiles,
``/path/to/venv/bin/meds-extract-run``), the venv's scripts directory is absent from
``PATH`` and the spawn dies with ``command not found`` — a failure class found
independently in three downstream repos. :func:`run_command` therefore builds the
child environment with this interpreter's scripts directory
(``sysconfig.get_path("scripts")`` — exactly what ``source .../activate`` prepends)
at the front of ``PATH``; the healed view is inherited transitively by the pipeline
runner's own per-stage spawns, retiring the failure class in one place. If
MEDS_transforms#398 lands an upstream fix, the healing becomes a harmless no-op.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import sysconfig
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from ..config import EtlConfig

logger = logging.getLogger(__name__)


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


def run_command(argv: list[str]) -> int:
    """Spawn a console-script command with an activation-equivalent child ``PATH``.

    The command's output streams straight through to this process's stdout/stderr
    (nothing is captured or swallowed), and the child's exit code is returned for
    the caller to propagate. ``argv[0]`` is pre-flight resolved with
    :func:`shutil.which` against the healed ``PATH``, so a broken environment fails
    immediately with an actionable message instead of mid-run.

    Examples:
        >>> run_command(["definitely-not-a-real-script-xyz"])
        Traceback (most recent call last):
            ...
        FileNotFoundError: 'definitely-not-a-real-script-xyz' not found in this environment's
        scripts directory ... or on PATH. Reinstall the package that provides it in this
        environment ...
    """
    env = {**os.environ, "PATH": activation_equivalent_path()}
    exe = shutil.which(argv[0], path=env["PATH"])
    if exe is None:
        raise FileNotFoundError(
            f"{argv[0]!r} not found in this environment's scripts directory "
            f"({sysconfig.get_path('scripts')}) or on PATH. Reinstall the package that "
            f"provides it in this environment ({sys.executable})."
        )
    logger.info(f"Running: {argv}")
    return subprocess.run([exe, *argv[1:]], env=env, check=False).returncode


def synthesize_pipeline_config(
    etl: EtlConfig,
    *,
    dataset_name: str,
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
    the child-env plumbing bugs the env-var pattern invited. Inlining is also how the
    computed identity reaches the ``MEDS_transform-pipeline`` subprocess — there is
    deliberately no in-process seam.

    ``dataset_name`` and ``dataset_version`` are passed explicitly (rather than read
    off ``etl``) because the runner computes both: the name defaults to the
    registered pipeline name when ``etl.dataset_name`` is omitted, and the version
    combines ``sources.dataset_version`` / ``etl.raw_dataset_version`` with the
    providing distribution's version.

    Examples:
        >>> from MEDS_extract.config import EtlConfig
        >>> from omegaconf import OmegaConf
        >>> etl = EtlConfig.parse({"row_chunksize": 100000})
        >>> cfg = synthesize_pipeline_config(
        ...     etl,
        ...     dataset_name="Example",
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
            row_chunksize: 100000
        - split_and_shard_subjects
        - convert_to_subject_sharded
        - convert_to_MEDS_events
        - merge_to_MEDS_cohort
        - extract_code_metadata
        - finalize_MEDS_metadata
        - finalize_MEDS_data
    """
    return {
        "etl_metadata": {
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
        },
        "event_conversion_config_fp": str(event_conversion_config_fp),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "shards_map_fp": f"{output_dir}/metadata/.shards.json",
        "stages": etl.stages_container(),
    }
