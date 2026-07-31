"""``meds-extract-download`` — CLI entry point for the download layer.

Reads a MESSY spec's ``sources:`` block and runs each resolved source's
:meth:`~MEDS_extract.download.source.Source.download_all` in sequence. Sources are
processed one at a time; per-file fetches within a source share one
:class:`~concurrent.futures.ThreadPoolExecutor` sized to the user's ``concurrency=``
argument, so the per-file transport bound is a global cap.

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
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import hydra
from MEDS_transforms.configs.utils import hydra_registered_dataclass
from omegaconf import MISSING, DictConfig

from ..config import MessyConfig
from .source import validate_unique_destinations

logger = logging.getLogger(__name__)


@hydra_registered_dataclass(group=None, name="download_defaults")
class DownloadConfig:
    """Typed config for ``meds-extract-download``.

    Fields:
        spec: Path or ``pkg://`` reference to the MESSY spec YAML with a
            ``sources:`` block.
        output_dir: Destination directory under which fetched files land.
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
    output_dir: str = MISSING
    key: str = "dataset"
    concurrency: int = 4
    continue_on_error: bool = False
    do_overwrite: bool = False


@hydra.main(version_base=None, config_name="download_defaults")
def main(cfg: DictConfig) -> None:
    """Entry point for the ``meds-extract-download`` console script.

    Required args (Hydra dotlist syntax):

    - ``spec=/path/to/event_configs.yaml`` — the MESSY spec with a ``sources:`` block
      (a ``pkg://`` reference also works).
    - ``output_dir=/path/to/output`` — where the fetched files land

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
    # ``pkg://`` specs resolve through the shared helper; filesystem specs resolve
    # against the user's original working directory — Hydra changes CWD by default,
    # so a relative `spec=` would otherwise be looked up under Hydra's output dir
    # and silently fail with FileNotFoundError.
    output_dir = Path(hydra.utils.to_absolute_path(str(cfg.output_dir))).expanduser().resolve()

    # One load of the one config object, then everything comes off it. Errors —
    # spec resolution, document validation, bad bucket key, malformed source
    # entries — are user config mistakes: log them and exit 1, mirroring the
    # manifest-validation handling below, rather than dumping a raw traceback
    # through Hydra. ``path_resolver`` maps relative specs against the user's
    # original working directory (Hydra changes CWD by default).
    def _user_path(p: str) -> Path:
        return Path(hydra.utils.to_absolute_path(p)).expanduser().resolve()

    try:
        messy = MessyConfig.load(str(cfg.spec), path_resolver=_user_path)
        sources = messy.selected_sources(key=cfg.key)
    except (TypeError, ValueError, FileNotFoundError) as e:
        logger.error(f"Could not construct sources from the spec: {e}")
        sys.exit(1)

    if not sources:
        logger.warning(f"No sources resolved for key={cfg.key!r} in {messy.source_fp}. Nothing to do.")
        return

    # Teardown notes:
    #
    # - ``shutdown(wait=False, cancel_futures=True)`` cancels *queued* futures
    #   immediately. Worker threads are NOT daemon threads (since Python 3.9,
    #   bpo-39812), so the interpreter joins any still-running workers at exit —
    #   in-flight transfers finish (or die when their transport is torn down)
    #   before the process can exit.
    # - The ExitStack closes sources LIFO *before* the pool-shutdown callback runs,
    #   so on any exit path each owned ``httpx.Client`` is closed while workers may
    #   still be streaming — those in-flight HTTP transfers fail fast rather than
    #   draining, which is what keeps Ctrl+C reasonably prompt for HTTP sources.
    #   Fsspec copies have no equivalent abort path and run to completion.
    with ExitStack() as stack:
        pool = ThreadPoolExecutor(max_workers=cfg.concurrency)
        stack.callback(pool.shutdown, wait=False, cancel_futures=True)
        for source in sources:
            stack.enter_context(source)

        # Materializes every source's manifest up-front (cached for the fetch loop
        # below) and fails before any fetch into output_dir if a manifest row is
        # malformed or two sources would write the same file. (Manifest listing
        # itself may do network I/O — e.g. PhysioNet's SHA256SUMS.txt GET, fsspec
        # source-side hashing.)
        try:
            validate_unique_destinations(sources)
        except ValueError as e:
            logger.error(f"Source manifests failed validation: {e}")
            sys.exit(1)

        all_ok = True
        for source in sources:
            try:
                source.download_all(
                    output_dir,
                    pool=pool,
                    continue_on_error=cfg.continue_on_error,
                    do_overwrite=cfg.do_overwrite,
                )
            except Exception:
                logger.exception(f"download_all failed for {type(source).__name__}")
                all_ok = False
                if not cfg.continue_on_error:
                    # Fail fast applies across sources too: don't start source N+1
                    # after source N has already sunk the run.
                    break
        if not all_ok:
            sys.exit(1)
