"""``meds-extract-download`` — CLI entry point for the download layer.

Reads a MESSY spec's ``sources:`` block and drives a :class:`Fetcher` to a local dest
dir. Written as a Hydra entry point so override syntax matches the rest of the
pipeline — users can e.g. flip a PhysioNet source to a local fsspec source for re-runs
via ``++sources.dataset.0.type=fsspec ++sources.dataset.0.root=/scratch/mirror``.

This is deliberately not a MEDS-transforms stage (see issue #81 for the design
rationale): download's I/O contract, parallelism axis, failure model, and config scope
all differ from the sharded-parquet stage machinery. Instead the download layer sits as
a pipeline-adjacent hook — same ergonomic goals as a stage (Hydra-driven, CLI-addressable,
override-friendly) without trying to fit the stage DAG.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from .dispatch import sources_from_spec
from .fetcher import Fetcher

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_name="download_defaults")
def _main(cfg: DictConfig) -> int:
    """Entry point for the ``meds-extract-download`` console script.

    Required args (Hydra dotlist syntax):

    - ``spec=/path/to/event_configs.yaml`` — the MESSY spec with a ``sources:`` block
    - ``raw_input_dir=/path/to/output`` — where the fetched files land

    Optional args:

    - ``key=dataset`` (default) / ``key=demo`` — which ``sources:`` bucket to pull.
      ``common`` is always appended.
    - ``concurrency=4`` (default) — max parallel transport streams per fetcher.
    - ``continue_on_error=False`` (default) — if True, per-file failures don't sink the run.
    - ``do_overwrite=False`` (default) — if True, re-fetch every file even if the local
      copy matches the manifest. Forwarded to each backend's ``fetch(..., do_overwrite=)``.

    Returns:
        ``0`` on full success (all files downloaded / skipped, none failed), ``1`` otherwise.
    """
    # Hydra changes CWD by default, so resolve relative paths against the user's original
    # working directory — otherwise `meds-extract-download spec=relative.yaml` would look
    # for the spec under Hydra's output dir and silently fail with FileNotFoundError.
    spec_fp = Path(hydra.utils.to_absolute_path(str(cfg.spec))).expanduser().resolve()
    raw_input_dir = Path(hydra.utils.to_absolute_path(str(cfg.raw_input_dir))).expanduser().resolve()

    spec = OmegaConf.to_container(OmegaConf.load(spec_fp), resolve=True)
    sources = sources_from_spec(spec, key=cfg.get("key", "dataset"))

    if not sources:
        logger.warning(
            f"No sources resolved for key={cfg.get('key', 'dataset')!r} in {spec_fp}. Nothing to do."
        )
        return 0

    fetcher = Fetcher(
        dest_dir=raw_input_dir,
        max_concurrency=cfg.get("concurrency", 4),
        continue_on_error=cfg.get("continue_on_error", False),
        do_overwrite=cfg.get("do_overwrite", False),
    )
    all_ok = True
    for source in sources:
        report = fetcher.fetch_all(source)
        all_ok = all_ok and report.ok
    return 0 if all_ok else 1


def main() -> None:  # pragma: no cover — thin wrapper for the console script
    """Console-script entry point.

    Registers defaults then delegates to ``_main``.
    """
    # Inject defaults so users can run without an external config file.
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(
        name="download_defaults",
        node={
            "spec": "???",
            "raw_input_dir": "???",
            "key": "dataset",
            "concurrency": 4,
            "continue_on_error": False,
            "do_overwrite": False,
        },
    )
    sys.exit(_main())
