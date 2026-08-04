"""``meds-extract-run`` — CLI entry point for the generic dataset-ETL runner.

One command runs a whole dataset ETL from its MESSY spec::

    meds-extract-run spec=MIMIC-IV output_dir=/data/mimic_meds download_key=demo
    meds-extract-run spec=pkg://MIMIC_IV_MEDS.configs.event_configs.yaml output_dir=...
    meds-extract-run spec=/path/to/messy.yaml output_dir=... download_key=null input_dir=...

The CLI itself just shuttles commands: it loads the spec into the one MESSY config
object (:meth:`~MEDS_extract.config.MessyConfig.load` — resolution ladder,
validation, and identity defaulting all live there), then spawns
``meds-extract-download`` and ``MEDS_transform-pipeline`` in turn via
:func:`run_command`, propagating child exit codes. The synthesized pipeline config
(written under ``<output_dir>/.meds_extract_run/``, every value inlined) is the
only channel through which the computed identity reaches the pipeline.

Both children run with an **activation-equivalent** ``PATH`` (this interpreter's
scripts directory prepended — exactly what ``source .../activate`` does), inherited
transitively by the pipeline runner's own bare ``MEDS_transform-stage`` spawns. That
keeps console-script resolution working when the children are spawned from an
environment whose ``bin/`` is not on ``PATH``. Once the console-script modules are
``python -m``-runnable (mmcdermott/MEDS_transforms#398), subprocesses can be spawned
via ``sys.executable`` directly and this healing becomes a no-op worth deleting.

Exits ``0`` on full success; config errors exit ``1``, child failures propagate the
child's exit code — via explicit :func:`sys.exit`, since Hydra discards the task
function's return value.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import hydra
from MEDS_transforms.configs.utils import hydra_registered_dataclass
from omegaconf import MISSING, DictConfig, OmegaConf

from ..config import MessyConfig, user_path

logger = logging.getLogger(__name__)


def activation_equivalent_path(path: str | None = None, *, scripts_dir: str | None = None) -> str:
    """Return ``PATH`` with this environment's scripts directory prepended.

    Prepending ``sysconfig.get_path("scripts")`` (the venv's ``bin/``, or
    ``Scripts\\`` on Windows) is exactly what ``source .../activate`` does to
    ``PATH`` — no bespoke resolution order is invented beyond the standard one.

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

    Output streams straight through to this process's stdout/stderr and the child's
    exit code is returned for the caller to propagate. ``argv[0]`` is pre-flight
    resolved against the healed ``PATH`` so a broken environment fails immediately
    with an actionable log line (returning 127, the shell's command-not-found code)
    instead of mid-run.

    Examples:
        An unresolvable command (an absolute path that cannot exist) returns 127
        without spawning anything:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     run_command([str(Path(d) / "no-such-script")])
        127
    """
    env = {**os.environ, "PATH": activation_equivalent_path()}
    exe = shutil.which(argv[0], path=env["PATH"])
    if exe is None:
        logger.error(
            f"{argv[0]!r} not found in this environment's scripts directory "
            f"({sysconfig.get_path('scripts')}) or on PATH. Reinstall the package that "
            f"provides it in this environment ({sys.executable})."
        )
        return 127
    logger.info(f"Running: {argv}")
    return subprocess.run([exe, *argv[1:]], env=env, check=False).returncode


@hydra_registered_dataclass(group=None, name="run_defaults")
class RunConfig:
    """Typed config for ``meds-extract-run``.

    Fields:
        spec: What to run — a name registered in the ``MEDS_extract.pipelines``
            entry-point group, a ``pkg://`` reference, or a path to a MESSY file.
        output_dir: Where the FINAL MEDS cohort lives (``data/``, ``metadata/``).
            Run-internal artifacts (the synthesized pipeline config, the download
            child's Hydra run dir) live under the derived :attr:`work_dir`
            (``<output_dir>/.meds_extract_run``), alongside the pipeline's own
            ``.logs``/intermediate stage outputs.
        download_key: Which ``sources:`` bucket to download (``dataset`` /
            ``demo`` / ...); ``common`` is always appended, and the per-bucket
            entry of a mapping-form ``sources.dataset_version`` follows it.
            ``null`` skips downloading entirely (``input_dir`` is then required;
            version stamping uses the default ``dataset`` bucket).
        download_dest_dir: Where to download raw data (only with a
            ``download_key``); it is then also the pipeline's effective input.
            Defaults under :attr:`work_dir` — point it somewhere durable to keep
            raw data across output trees.
        input_dir: Where pre-staged raw input data lives (only with
            ``download_key=null``).
        dataset_version: Explicit override for ``etl_metadata.dataset_version``
            (default: computed — see ``MessyConfig.dataset_version_for``).
        do_overwrite: If ``True``, the download stage re-fetches files even when
            the local copy matches.

    ``__post_init__`` runs when the CLI materializes the Hydra config via
    ``OmegaConf.to_object`` (Hydra itself always hands the task function a
    ``DictConfig``; ``to_object`` is the idiomatic bridge back to the registered
    dataclass). It validates the plausible combinations — exactly one of "download
    into ``download_dest_dir``" (``download_key`` set) or "read pre-staged
    ``input_dir``" (``download_key=null``) describes where the pipeline's raw
    input comes from (:attr:`effective_input_dir`) — and absolutizes the directory
    fields against the user's original working directory, so no path handling is
    left to the CLI body.
    """

    spec: str = MISSING
    output_dir: str = MISSING
    download_key: str | None = "dataset"
    download_dest_dir: str | None = None
    input_dir: str | None = None
    dataset_version: str | None = None
    do_overwrite: bool = False

    def __post_init__(self):
        for f in ("output_dir", "download_dest_dir", "input_dir"):
            if getattr(self, f) not in (None, MISSING):
                setattr(self, f, str(user_path(getattr(self, f))))
        if self.download_key is None and self.input_dir is None:
            raise ValueError(
                "download_key=null (no download) requires input_dir= pointing at pre-staged raw data."
            )
        if self.download_key is not None and self.input_dir is not None:
            raise ValueError(
                "input_dir= is for download-free runs (download_key=null). With a download_key, "
                "data is downloaded into download_dest_dir=, which is also the pipeline's input "
                "— set one or the other."
            )
        if self.download_key is None and self.download_dest_dir is not None:
            raise ValueError("download_dest_dir= has no effect with download_key=null; use input_dir=.")

    @property
    def work_dir(self) -> Path:
        """Run-internal artifact dir: ``<output_dir>/.meds_extract_run``."""
        return Path(self.output_dir) / ".meds_extract_run"

    @property
    def effective_input_dir(self) -> Path:
        """The pipeline's raw-data input: ``input_dir`` or the download destination."""
        if self.input_dir is not None:
            return Path(self.input_dir)
        return Path(self.download_dest_dir) if self.download_dest_dir else self.work_dir / "raw_input"


@hydra.main(version_base=None, config_name="run_defaults")
def main(cfg: DictConfig) -> None:
    """Entry point for the ``meds-extract-run`` console script.

    Required args (Hydra dotlist syntax): ``spec=...`` and ``output_dir=...``; see
    :class:`RunConfig` for the optional knobs.
    """

    try:
        # Materialize the registered dataclass: __post_init__ validates the flag
        # combinations and absolutizes the directory fields.
        run: RunConfig = OmegaConf.to_object(cfg)
        # One load of the one config object; everything else comes off it.
        messy = MessyConfig.load(run.spec, path_resolver=user_path)
        _ = messy.event_tables  # the pipeline needs event tables: fail before any work
        # Version stamping follows the selected sources bucket; a download-free run
        # (download_key=null) has no selected bucket and stamps the default
        # ``dataset`` entry (only relevant for mapping-form sources.dataset_version).
        stamp_key = run.download_key if run.download_key is not None else "dataset"
        pipeline_cfg = messy.pipeline_config(
            input_dir=run.effective_input_dir,
            output_dir=Path(run.output_dir),
            key=stamp_key,
            dataset_version=run.dataset_version,
        )
    except (ValueError, FileNotFoundError) as e:
        logger.error(str(e))
        sys.exit(1)
    logger.info(f"Resolved spec={run.spec!r} to {messy.spec_ref}")

    if run.download_key is not None:
        rc = run_command(
            [
                "meds-extract-download",
                f"spec={messy.spec_ref}",
                f"output_dir={run.effective_input_dir}",
                f"key={run.download_key}",
                f"do_overwrite={run.do_overwrite}",
                # Keep the child's Hydra run dir out of the user's CWD.
                f"hydra.run.dir={run.work_dir / 'hydra_download'}",
            ]
        )
        if rc != 0:
            logger.error(f"meds-extract-download failed with exit code {rc}.")
            sys.exit(rc)
    else:
        logger.info("download_key=null: skipping the download stage.")

    pipeline_fp = run.work_dir / "pipeline.yaml"
    pipeline_fp.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(pipeline_cfg), pipeline_fp)
    logger.info(f"Wrote synthesized pipeline config to {pipeline_fp}")

    sys.exit(run_command(["MEDS_transform-pipeline", str(pipeline_fp)]))
