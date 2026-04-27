"""``meds-extract-download`` — CLI entry point for the download layer.

Reads a MESSY spec's ``sources:`` block, builds one :class:`ThreadPoolExecutor`
+ one :class:`DownloadPolicy` from CLI flags, and runs ``download_all`` on each
source under the shared pool. Written as a Hydra entry point so override syntax
matches the rest of the pipeline — users can e.g. flip a PhysioNet source to a
local fsspec source for re-runs via
``++sources.dataset.0.type=fsspec ++sources.dataset.0.root=/scratch/mirror``.

This is deliberately not a MEDS-transforms stage (see issue #81 for the design
rationale): download's I/O contract, parallelism axis, failure model, and config
scope all differ from the sharded-parquet stage machinery. Instead the download
layer sits as a pipeline-adjacent hook — same ergonomic goals as a stage
(Hydra-driven, CLI-addressable, override-friendly) without trying to fit the
stage DAG.

The config schema is defined as a ``hydra_registered_dataclass`` (from MEDS-transforms)
which both registers it with Hydra's ``ConfigStore`` and types it as a dataclass — so
``cfg.spec`` / ``cfg.do_overwrite`` / etc are typed attributes, not dict-style lookups.
"""

from __future__ import annotations

import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import hydra
from MEDS_transforms.configs.utils import hydra_registered_dataclass
from omegaconf import MISSING, DictConfig, OmegaConf

from .dispatch import sources_from_spec
from .source import DownloadPolicy

logger = logging.getLogger(__name__)


@hydra_registered_dataclass(group=None, name="download_defaults")
class DownloadConfig:
    """Typed config for ``meds-extract-download``.

    Fields:
        spec: Path to the MESSY spec YAML with a ``sources:`` block.
        raw_input_dir: Destination directory under which fetched files land.
        key: Which ``sources:`` bucket to pull. ``"common"`` is always appended.
        concurrency: Max parallel transport streams per fetcher.
        continue_on_error: If ``True``, per-file failures don't sink the run.
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

    policy = DownloadPolicy(
        continue_on_error=cfg.continue_on_error,
        do_overwrite=cfg.do_overwrite,
    )
    # Register every source with an ExitStack so each one's ``__exit__`` → ``close()``
    # fires on the way out, whether the loop completes normally or raises. Owned
    # ``httpx.Client`` connections / pools get released without a hand-rolled try/finally.
    # The ``ThreadPoolExecutor`` lives on the same stack — ``download_all`` borrows it,
    # never owns it, so a single pool drives every source in this CLI invocation.
    with ExitStack() as stack:
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=cfg.concurrency))
        for source in sources:
            stack.enter_context(source)
        all_ok = True
        for source in sources:
            report = source.download_all(raw_input_dir, pool=pool, policy=policy)
            all_ok = all_ok and report.ok
        return 0 if all_ok else 1


if __name__ == "__main__":  # pragma: no cover — exercised by subprocess integration test
    # Setuptools wraps console-script entry points in ``sys.exit(func())`` automatically,
    # so the ``meds-extract-download`` binary propagates the Hydra-decorated ``main``'s
    # return value (0 on full success, 1 on any per-file failure) as the process exit
    # code. ``python -m MEDS_extract.download.cli`` does NOT get that wrapping, so we
    # have to do it here manually — otherwise a partial-failure return value of 1 would
    # be discarded and the subprocess would falsely report success.
    sys.exit(main())
