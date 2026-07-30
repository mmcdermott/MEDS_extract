"""``meds-extract-run`` — CLI entry point for the generic dataset-ETL runner.

One command runs a whole dataset ETL from its MESSY spec::

    meds-extract-run spec=MIMIC-IV root_output_dir=/data/mimic key=demo
    meds-extract-run spec=pkg://MIMIC_IV_MEDS.configs.event_configs.yaml root_output_dir=...
    meds-extract-run spec=/path/to/messy.yaml root_output_dir=... do_download=false

Flow: resolve the spec down the registered-name → ``pkg://`` → path ladder
(:mod:`.registry`); spawn ``meds-extract-download`` to stage the selected
``sources:`` bucket (unless ``do_download=false``); synthesize a fully-inlined
MEDS-transforms pipeline config from the spec's ``etl:`` block under
``<root_output_dir>/.meds_extract_run/``; and spawn ``MEDS_transform-pipeline`` on
it, propagating its exit code. Both children run with an activation-equivalent
``PATH`` (:mod:`.pipeline`), which also heals the pipeline runner's own per-stage
console-script spawns (MEDS_transforms#398).

Written as a Hydra entry point to match ``meds-extract-download``'s override syntax
(``key=value`` everywhere).

Output layout under ``root_output_dir`` (names follow the conventions of existing
dataset packages):

- ``raw_input/`` — staged raw data (overridable via ``raw_input_dir=`` for
  pre-staged data elsewhere, typically with ``do_download=false``);
- ``MEDS_output/`` — the pipeline's output tree (``data/``, ``metadata/``);
- ``.meds_extract_run/`` — the synthesized ``pipeline.yaml`` (self-contained
  provenance: every value inlined, no env-var indirection) and the download
  child's Hydra run dir.

Exits ``0`` on full success; any failure exits non-zero (config errors exit ``1``,
child failures propagate the child's exit code) via explicit :func:`sys.exit` —
Hydra discards the task function's return value.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from MEDS_transforms.configs.utils import hydra_registered_dataclass
from omegaconf import MISSING, DictConfig, OmegaConf

from ..config import EtlConfig, effective_raw_dataset_version, read_sources_dataset_version
from .pipeline import run_command, synthesize_pipeline_config
from .registry import resolve_spec

logger = logging.getLogger(__name__)


@hydra_registered_dataclass(group=None, name="run_defaults")
class RunConfig:
    """Typed config for ``meds-extract-run``.

    Fields:
        spec: What to run — a name registered in the ``MEDS_extract.pipelines``
            entry-point group, a ``pkg://`` reference, or a path to a MESSY file.
        root_output_dir: Root directory for everything the run produces (see the
            module docstring for the layout).
        key: Which ``sources:`` bucket to stage (``dataset`` / ``demo`` / ...);
            ``common`` is always appended. Also selects the per-bucket entry of a
            mapping-form ``sources.dataset_version``.
        do_download: If ``False``, skip staging entirely and run the pipeline
            against already-present raw data.
        raw_input_dir: Where raw data is staged / read from. Defaults to
            ``<root_output_dir>/raw_input``.
        dataset_version: Explicit override for ``etl_metadata.dataset_version``.
            Default (null): the effective raw version (``sources.dataset_version``,
            falling back to ``etl.raw_dataset_version``) — suffixed with
            ``:{providing distribution's version}`` for registry-resolved specs.
        do_overwrite: If ``True``, the download stage re-fetches files even when the
            local copy matches.
    """

    spec: str = MISSING
    root_output_dir: str = MISSING
    key: str = "dataset"
    do_download: bool = True
    raw_input_dir: str | None = None
    dataset_version: str | None = None
    do_overwrite: bool = False


@hydra.main(version_base=None, config_name="run_defaults")
def main(cfg: DictConfig) -> None:
    """Entry point for the ``meds-extract-run`` console script.

    Required args (Hydra dotlist syntax):

    - ``spec=MIMIC-IV`` (registered name) / ``spec=pkg://...`` / ``spec=/path/to/messy.yaml``
    - ``root_output_dir=/path/to/output``

    See :class:`RunConfig` for the optional knobs. Exits ``0`` on success and
    non-zero on any failure.
    """

    # Hydra changes CWD by default; resolve user-relative paths against the original
    # working directory (mirroring ``meds-extract-download``).
    def _user_path(p: str) -> Path:
        return Path(hydra.utils.to_absolute_path(p)).expanduser()

    try:
        resolved = resolve_spec(str(cfg.spec), path_resolver=_user_path)
        etl = EtlConfig.load(resolved.spec_fp)
        sources_version = read_sources_dataset_version(resolved.spec_fp)
        raw_version = effective_raw_dataset_version(sources_version, etl.raw_dataset_version, cfg.key)
    except (ValueError, FileNotFoundError) as e:
        logger.error(str(e))
        sys.exit(1)
    logger.info(f"Resolved spec={cfg.spec!r} via {resolved.origin!r} to {resolved.spec_fp}")

    # Dataset name: a registry-resolved spec inherits the registered pipeline name
    # when `etl.dataset_name` is omitted (the minimal-block form); pkg:// / path
    # resolutions have no registry name to inherit, so there it is required.
    dataset_name = etl.dataset_name
    if dataset_name is None:
        if resolved.origin == "registry":
            dataset_name = str(cfg.spec)
        else:
            logger.error(
                f"The etl: block in {resolved.spec_fp} omits dataset_name, which is only "
                f"allowed when the spec is resolved via a registered MEDS_extract.pipelines "
                f"entry-point name (the dataset name then defaults to that name). For "
                f"pkg://- and path-resolved specs, add dataset_name to the etl: block."
            )
            sys.exit(1)

    # ETL provenance stamp. A registry-resolved spec knows its providing
    # distribution, so the stamp is `{raw data release}:{ETL package version}` with
    # zero code in the dataset package; pkg:// / path resolutions have no
    # distribution to ask and stamp the raw release version alone unless the user
    # passes an explicit dataset_version=.
    if cfg.dataset_version:
        dataset_version = str(cfg.dataset_version)
    elif resolved.dist_version:
        dataset_version = f"{raw_version}:{resolved.dist_version}"
    else:
        dataset_version = raw_version

    root_output_dir = _user_path(str(cfg.root_output_dir)).resolve()
    raw_input_dir = (
        _user_path(str(cfg.raw_input_dir)).resolve() if cfg.raw_input_dir else root_output_dir / "raw_input"
    )
    output_dir = root_output_dir / "MEDS_output"
    run_dir = root_output_dir / ".meds_extract_run"

    if cfg.do_download:
        try:
            rc = run_command(
                [
                    "meds-extract-download",
                    f"spec={resolved.spec_fp}",
                    f"raw_input_dir={raw_input_dir}",
                    f"key={cfg.key}",
                    f"do_overwrite={cfg.do_overwrite}",
                    # Keep the child's Hydra run dir out of the user's CWD.
                    f"hydra.run.dir={run_dir / 'hydra_download'}",
                ]
            )
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)
        if rc != 0:
            logger.error(f"meds-extract-download failed with exit code {rc}.")
            sys.exit(rc)
    else:
        logger.info("do_download=false: skipping the download stage.")

    pipeline_fp = run_dir / "pipeline.yaml"
    pipeline_fp.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        OmegaConf.create(
            synthesize_pipeline_config(
                etl,
                dataset_name=dataset_name,
                event_conversion_config_fp=resolved.spec_fp,
                input_dir=raw_input_dir,
                output_dir=output_dir,
                dataset_version=dataset_version,
            )
        ),
        pipeline_fp,
    )
    logger.info(f"Wrote synthesized pipeline config to {pipeline_fp}")

    try:
        sys.exit(run_command(["MEDS_transform-pipeline", str(pipeline_fp)]))
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
