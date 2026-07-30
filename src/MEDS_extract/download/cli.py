"""``meds-extract-download`` — CLI entry point for the download layer.

Reads a MESSY spec's ``sources:`` block and stages the selected bucket via
:func:`~MEDS_extract.download.api.stage_sources` — the shared library orchestration
also used in-process by ``meds-extract-run``. This module owns only what is
CLI-specific: Hydra argument handling, ``pkg://`` / relative-path spec resolution,
and mapping library exceptions to log lines + exit codes.

Written as a Hydra entry point so override syntax matches the rest of the pipeline.
To re-run against a local mirror instead of the original remote, edit the spec's
``sources:`` block (or keep a second bucket — e.g. a ``mirror:`` bucket with a
``type: fsspec`` entry — and select it via ``key=mirror``); the source definitions
themselves live in the spec file, not in Hydra's config.

The config schema is a ``hydra_registered_dataclass`` (from MEDS-transforms): it
registers with Hydra's ``ConfigStore`` and types as a dataclass, so ``cfg.spec`` /
``cfg.do_overwrite`` / etc are typed attributes.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from MEDS_transforms.configs.utils import hydra_registered_dataclass
from MEDS_transforms.utils import PKG_PFX, resolve_pkg_path
from omegaconf import MISSING, DictConfig

from .api import DownloadError, stage_sources

logger = logging.getLogger(__name__)


@hydra_registered_dataclass(group=None, name="download_defaults")
class DownloadConfig:
    """Typed config for ``meds-extract-download``.

    Fields:
        spec: The MESSY spec YAML with a ``sources:`` block — a filesystem path or a
            ``pkg://`` reference to a spec bundled inside an installed package (e.g.
            ``pkg://MIMIC_IV_MEDS.configs.event_configs.yaml``, resolved via
            MEDS-transforms' ``resolve_pkg_path`` so the two CLIs share one syntax).
        raw_input_dir: Destination directory under which fetched files land.
        key: Which ``sources:`` bucket to pull. ``"common"`` is always appended.
            Must name a bucket that actually exists in the spec (guards against
            typos silently downloading nothing).
        concurrency: Max parallel transport streams across all sources (one shared pool).
        continue_on_error: If ``True``, per-file failures don't sink the run; every
            source is still attempted and the process exits non-zero at the end if
            anything failed. If ``False`` (default), the first failing source stops
            the whole run.
        do_overwrite: If ``True``, re-fetch every file even if the local copy matches.
    """

    spec: str = MISSING
    raw_input_dir: str = MISSING
    key: str = "dataset"
    concurrency: int = 4
    continue_on_error: bool = False
    do_overwrite: bool = False


def resolve_spec_path(spec: str) -> Path:
    """Resolve the ``spec=`` argument to an on-disk path (``pkg://`` or filesystem).

    ``pkg://`` references resolve through MEDS-transforms' ``resolve_pkg_path`` (the
    same helper the pipeline CLI uses for pipeline configs, so the syntax cannot
    drift): ``pkg://<pkg_name>.<dotted.relative.path>.<ext>``. Filesystem paths are
    resolved against the user's original working directory — Hydra changes CWD by
    default, so a relative ``spec=`` would otherwise be looked up under Hydra's
    output dir and silently fail with FileNotFoundError.

    Examples:
        >>> resolve_spec_path("pkg://MEDS_extract.configs._extract.yaml").name
        '_extract.yaml'
        >>> resolve_spec_path("relative/spec.yaml").is_absolute()
        True
    """
    if spec.startswith(PKG_PFX):
        return Path(str(resolve_pkg_path(spec)))
    return Path(hydra.utils.to_absolute_path(spec)).expanduser().resolve()


@hydra.main(version_base=None, config_name="download_defaults")
def main(cfg: DictConfig) -> None:
    """Entry point for the ``meds-extract-download`` console script.

    Required args (Hydra dotlist syntax):

    - ``spec=/path/to/event_configs.yaml`` — the MESSY spec with a ``sources:`` block.
      Also accepts ``pkg://`` syntax for a spec bundled inside an installed package
      (e.g. ``spec=pkg://MIMIC_IV_MEDS.configs.event_configs.yaml``).
    - ``raw_input_dir=/path/to/output`` — where the fetched files land

    Optional args:

    - ``key=dataset`` (default) / ``key=demo`` — which ``sources:`` bucket to pull.
      ``common`` is always appended. When the spec declares sources buckets, a
      ``key`` naming none of them is an error (not a silent no-op); a spec with no
      ``sources:`` block at all is a legitimately download-free ETL and warns +
      exits 0 regardless of ``key``.
    - ``concurrency=4`` (default) — max parallel transport streams across all sources.
      One :class:`~concurrent.futures.ThreadPoolExecutor` is shared by every source's
      ``download_all`` call so the bound applies globally rather than per-source.
    - ``continue_on_error=False`` (default) — if True, per-file failures don't sink the
      run and every source is attempted; if False, the first failing source stops the
      whole run.
    - ``do_overwrite=False`` (default) — if True, re-fetch every file even if the local
      copy matches the manifest.

    Exits ``0`` on full success and ``1`` on any failure, via an explicit
    :func:`sys.exit` — Hydra discards the task function's *return* value, so a
    plain ``return 1`` would not reach the process exit code.
    """
    spec_fp = resolve_spec_path(str(cfg.spec))
    raw_input_dir = Path(hydra.utils.to_absolute_path(str(cfg.raw_input_dir))).expanduser().resolve()

    try:
        stage_sources(
            spec_fp,
            raw_input_dir,
            key=cfg.key,
            concurrency=cfg.concurrency,
            continue_on_error=cfg.continue_on_error,
            do_overwrite=cfg.do_overwrite,
        )
    except (TypeError, ValueError) as e:
        # User config mistakes (unknown bucket key, malformed source entries, manifest
        # validation): one actionable log line, not a raw traceback through Hydra.
        logger.error(str(e))
        sys.exit(1)
    except DownloadError:
        # Per-source failures were already logged (with traceback) by stage_sources.
        sys.exit(1)
