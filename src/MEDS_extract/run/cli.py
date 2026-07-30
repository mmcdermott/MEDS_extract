"""``meds-extract-run`` — CLI entry point for the generic dataset-ETL runner.

One command runs a whole dataset ETL from its MESSY spec::

    meds-extract-run spec=MIMIC-IV root_output_dir=/data/mimic key=demo
    meds-extract-run spec=pkg://MIMIC_IV_MEDS.configs.event_configs.yaml root_output_dir=...
    meds-extract-run spec=/path/to/messy.yaml root_output_dir=... do_download=false

Flow: resolve the spec down the registered-name → ``pkg://`` → path ladder
(:mod:`.registry`); stage the selected ``sources:`` bucket **in-process** via the
download layer's :func:`~MEDS_extract.download.api.stage_sources` (unless
``do_download=false``); synthesize a fully-inlined MEDS-transforms pipeline config
from the spec's ``etl:`` block under ``<root_output_dir>/.meds_extract_run/``; and
invoke the pipeline **in-process** through ``MEDS_transforms.runner.main``
(:mod:`.pipeline`, which also heals the MEDS_transforms#398 PATH sensitivity).

Written as a Hydra entry point to match ``meds-extract-download``'s override syntax
(``key=value`` everywhere) — Hydra does not conflict with the in-process pipeline
invocation because ``MEDS_transforms.runner.main`` is plain argparse (no nested
``@hydra.main``), and each *stage* runs in its own subprocess with its own Hydra
context.

Output layout under ``root_output_dir`` (names follow the conventions of existing
dataset packages):

- ``raw_input/`` — staged raw data (overridable via ``raw_input_dir=`` for
  pre-staged data elsewhere, typically with ``do_download=false``);
- ``MEDS_output/`` — the pipeline's output tree (``data/``, ``metadata/``);
- ``.meds_extract_run/pipeline.yaml`` — the synthesized pipeline config
  (self-contained provenance: every value inlined, no env-var indirection).

Exits ``0`` on full success and ``1`` on any failure, via explicit
:func:`sys.exit` — Hydra discards the task function's return value.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from MEDS_transforms.configs.utils import hydra_registered_dataclass
from omegaconf import MISSING, DictConfig

from ..config import EtlConfig
from ..download.api import DownloadError, stage_sources
from .pipeline import run_pipeline, synthesize_pipeline_config, write_pipeline_config
from .registry import resolve_spec

logger = logging.getLogger(__name__)


@hydra_registered_dataclass(group=None, name="run_defaults")
class RunConfig:
    """Typed config for ``meds-extract-run``.

    Fields:
        spec: What to run — a name registered in the ``MEDS_extract.pipelines``
            entry-point group, a ``pkg://`` reference, or a path to a MESSY file
            (which must carry an ``etl:`` block; ``sources:`` is needed only when
            ``do_download`` is on).
        root_output_dir: Root directory for everything the run produces (see the
            module docstring for the layout).
        key: Which ``sources:`` bucket to stage (``dataset`` / ``demo`` / ...);
            ``common`` is always appended. Ignored when ``do_download=false``.
        do_download: If ``False``, skip staging entirely and run the pipeline
            against already-present raw data.
        raw_input_dir: Where raw data is staged / read from. Defaults to
            ``<root_output_dir>/raw_input``.
        dataset_version: Explicit override for ``etl_metadata.dataset_version``.
            Default (null): registry-resolved specs stamp
            ``{etl.raw_dataset_version}:{providing distribution's version}``;
            ``pkg://`` / path specs (no distribution to ask) stamp
            ``etl.raw_dataset_version`` alone.
        concurrency: Max parallel transport streams for the download stage.
        continue_on_error: Download-stage failure policy (as in
            ``meds-extract-download``).
        do_overwrite: If ``True``, the download stage re-fetches files even when the
            local copy matches.
    """

    spec: str = MISSING
    root_output_dir: str = MISSING
    key: str = "dataset"
    do_download: bool = True
    raw_input_dir: str | None = None
    dataset_version: str | None = None
    concurrency: int = 4
    continue_on_error: bool = False
    do_overwrite: bool = False


@hydra.main(version_base=None, config_name="run_defaults")
def main(cfg: DictConfig) -> None:
    """Entry point for the ``meds-extract-run`` console script.

    Required args (Hydra dotlist syntax):

    - ``spec=MIMIC-IV`` (registered name) / ``spec=pkg://...`` / ``spec=/path/to/messy.yaml``
    - ``root_output_dir=/path/to/output``

    See :class:`RunConfig` for the optional knobs. Exits ``0`` on success, ``1`` on
    any failure.
    """

    # Hydra changes CWD by default; resolve user-relative paths against the original
    # working directory (mirroring ``meds-extract-download``).
    def _user_path(p: str) -> Path:
        return Path(hydra.utils.to_absolute_path(p)).expanduser()

    try:
        resolved = resolve_spec(str(cfg.spec), path_resolver=_user_path)
        etl = EtlConfig.load(resolved.spec_fp)
    except (ValueError, FileNotFoundError) as e:
        logger.error(str(e))
        sys.exit(1)
    logger.info(f"Resolved spec={cfg.spec!r} via {resolved.origin!r} to {resolved.spec_fp}")

    root_output_dir = _user_path(str(cfg.root_output_dir)).resolve()
    raw_input_dir = (
        _user_path(str(cfg.raw_input_dir)).resolve() if cfg.raw_input_dir else root_output_dir / "raw_input"
    )
    output_dir = root_output_dir / "MEDS_output"

    if cfg.do_download:
        try:
            stage_sources(
                resolved.spec_fp,
                raw_input_dir,
                key=cfg.key,
                concurrency=cfg.concurrency,
                continue_on_error=cfg.continue_on_error,
                do_overwrite=cfg.do_overwrite,
            )
        except (TypeError, ValueError) as e:
            logger.error(str(e))
            sys.exit(1)
        except DownloadError:
            # Per-source failures were already logged (with traceback) by stage_sources.
            sys.exit(1)
    else:
        logger.info("do_download=false: skipping the download stage.")

    # ETL provenance stamp. A registry-resolved spec knows its providing
    # distribution, so the stamp is `{raw data release}:{ETL package version}` with
    # zero code in the dataset package; pkg:// / path resolutions have no
    # distribution to ask and stamp the raw release version alone unless the user
    # passes an explicit dataset_version=.
    if cfg.dataset_version:
        dataset_version = str(cfg.dataset_version)
    elif resolved.dist_version:
        dataset_version = f"{etl.raw_dataset_version}:{resolved.dist_version}"
    else:
        dataset_version = etl.raw_dataset_version

    pipeline_fp = root_output_dir / ".meds_extract_run" / "pipeline.yaml"
    write_pipeline_config(
        synthesize_pipeline_config(
            etl,
            event_conversion_config_fp=resolved.spec_fp,
            input_dir=raw_input_dir,
            output_dir=output_dir,
            dataset_version=dataset_version,
        ),
        pipeline_fp,
    )

    try:
        rc = run_pipeline(pipeline_fp)
    except Exception:
        logger.exception(f"Pipeline run failed (config: {pipeline_fp}).")
        sys.exit(1)
    sys.exit(rc)
