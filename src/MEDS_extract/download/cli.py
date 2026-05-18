"""``meds-extract-download`` — CLI entry point for the download layer.

Reads a MESSY spec's ``sources:`` block and runs each resolved source's
:meth:`~MEDS_extract.download.source.Source.download_all` in sequence. Sources are
processed one at a time; per-file fetches within a source share one
:class:`~concurrent.futures.ThreadPoolExecutor` sized to the user's ``concurrency=``
argument, so the per-file transport bound is a global cap. The pool is shut down
with ``shutdown(wait=False, cancel_futures=True)`` so a ``Ctrl+C`` mid-download
cancels queued work immediately.

Written as a Hydra entry point so override syntax matches the rest of the pipeline —
users can e.g. flip a PhysioNet source to a local fsspec source for re-runs via
``++sources.dataset.0.type=fsspec ++sources.dataset.0.root=/scratch/mirror``.

The config schema is a ``hydra_registered_dataclass`` (from MEDS-transforms): it
registers with Hydra's ``ConfigStore`` and types as a dataclass, so ``cfg.spec`` /
``cfg.do_overwrite`` / etc are typed attributes.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import hydra
from MEDS_transforms.configs.utils import hydra_registered_dataclass
from omegaconf import MISSING, DictConfig, OmegaConf

from .dispatch import sources_from_spec

logger = logging.getLogger(__name__)


@hydra_registered_dataclass(group=None, name="download_defaults")
class DownloadConfig:
    """Typed config for ``meds-extract-download``.

    Fields:
        spec: Path to the MESSY spec YAML with a ``sources:`` block.
        raw_input_dir: Destination directory under which fetched files land.
        key: Which ``sources:`` bucket to pull. ``"common"`` is always appended.
        concurrency: Max parallel transport streams across all sources (one shared pool).
        continue_on_error: If ``True``, per-file failures don't sink the run; an
            ``ExceptionGroup`` is raised at the end if any failed.
        do_overwrite: If ``True``, re-fetch every file even if the local copy matches.
    """

    spec: str = MISSING
    raw_input_dir: str = MISSING
    key: str = "dataset"
    concurrency: int = 4
    continue_on_error: bool = False
    do_overwrite: bool = False


@hydra.main(version_base=None, config_name="download_defaults")
def main(cfg: DictConfig) -> int:
    """Entry point for the ``meds-extract-download`` console script.

    Required args (Hydra dotlist syntax):

    - ``spec=/path/to/event_configs.yaml`` — the MESSY spec with a ``sources:`` block
    - ``raw_input_dir=/path/to/output`` — where the fetched files land

    Optional args:

    - ``key=dataset`` (default) / ``key=demo`` — which ``sources:`` bucket to pull.
      ``common`` is always appended.
    - ``concurrency=4`` (default) — max parallel transport streams across all sources.
      One :class:`~concurrent.futures.ThreadPoolExecutor` is shared by every source's
      ``download_all`` call so the bound applies globally rather than per-source.
    - ``continue_on_error=False`` (default) — if True, per-file failures don't sink the
      run; an ``ExceptionGroup`` is raised at the end if any failed.
    - ``do_overwrite=False`` (default) — if True, re-fetch every file even if the local
      copy matches the manifest.

    Returns:
        ``0`` on full success, ``1`` if any source raised. Hydra surfaces non-zero
        returns as the process exit code.
    """
    # Hydra changes CWD by default, so resolve relative paths against the user's original
    # working directory — otherwise `meds-extract-download spec=relative.yaml` would look
    # for the spec under Hydra's output dir and silently fail with FileNotFoundError.
    spec_fp = Path(hydra.utils.to_absolute_path(str(cfg.spec))).expanduser().resolve()
    raw_input_dir = Path(hydra.utils.to_absolute_path(str(cfg.raw_input_dir))).expanduser().resolve()

    # Resolve interpolations on ONLY the ``sources:`` subtree. Under the combined-MESSY
    # pattern (one file carrying both ``sources:`` and event-conversion entries),
    # resolving the whole document would require every ``${oc.env:...}`` in unrelated
    # event-conversion sections to be set just to run ``meds-extract-download``. This is
    # the symmetric sibling of ``MessyConfig.parse``'s "strip reserved keys before
    # resolve=True" fix — the two layers coexist without cross-polluting env requirements.
    spec_raw = OmegaConf.load(spec_fp)
    sources_node = spec_raw.get("sources")
    sources_dict = OmegaConf.to_container(sources_node, resolve=True) if sources_node is not None else {}
    sources = sources_from_spec({"sources": sources_dict}, key=cfg.key)

    if not sources:
        logger.warning(f"No sources resolved for key={cfg.key!r} in {spec_fp}. Nothing to do.")
        return 0

    # Single shared pool across all sources. ``shutdown(wait=False,
    # cancel_futures=True)`` is registered as an ExitStack callback so a Ctrl+C
    # cancels queued submissions immediately; running daemon worker threads get
    # abandoned at interpreter teardown and the OS tears down their sockets.
    # The same ExitStack closes each ``Source`` on the way out so owned
    # ``httpx.Client`` pools are released whether the loop exits normally or
    # via exception.
    with ExitStack() as stack:
        pool = ThreadPoolExecutor(max_workers=cfg.concurrency)
        stack.callback(pool.shutdown, wait=False, cancel_futures=True)
        for source in sources:
            stack.enter_context(source)
        all_ok = True
        for source in sources:
            try:
                source.download_all(
                    raw_input_dir,
                    pool=pool,
                    continue_on_error=cfg.continue_on_error,
                    do_overwrite=cfg.do_overwrite,
                )
            except Exception:
                logger.exception(f"download_all failed for {type(source).__name__}")
                all_ok = False
        return 0 if all_ok else 1
