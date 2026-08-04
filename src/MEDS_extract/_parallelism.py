"""Guard for the one configuration where parallel workers cannot safely share work.

Every map stage divides work the same way: all workers walk the same list of
``(input -> output)`` pairs, and :func:`~MEDS_transforms.mapreduce.rwlock.rwlock_wrap`
makes the first worker to produce an output the only one that computes it — the rest
hit the "output exists, return" branch and move on.

``do_overwrite=True`` disables exactly that branch. Instead of skipping, every worker
unlinks the finished output and recomputes it, so N workers do somewhere between 1x
and Nx the work rather than 1/N of it each. (The ``FileLock`` still prevents two
workers computing the same output *simultaneously* — the second gets a ``Timeout``
and moves on — so duplication happens whenever a worker arrives after another has
finished, which is common given each worker shuffles its list independently.)

In a pure map stage that is only waste. In ``extract_code_metadata`` it is also a
crash: that stage reduces its own map outputs in the same invocation, and
``rwlock_wrap`` unlinks *before* taking its lock, so a straggler can delete a partial
the reducer has already validated (#194).

Rather than let either happen, a run that sets ``do_overwrite`` runs single-worker.
That is a real cost — the caller asked for parallelism and does not get it — so the
warning names the alternative that does parallelize.

The proper fix belongs upstream: if ``rwlock_wrap``'s cache could distinguish "this
output was produced during THIS run" from "left over from a previous one", every
worker could safely skip a sibling's fresh output while still discarding stale ones,
and ``do_overwrite`` would parallelize normally. Nothing available to a stage
identifies a run — hydra's ``sweep.dir`` is overridden to a fixed per-stage log
directory here, and ``job.num``/``job.id`` are per-worker — so this cannot be fixed
from inside a stage. Tracked upstream in mmcdermott/MEDS_transforms.

Note this is a property of the cache-based work division, NOT of parallelism itself:
work partitioned by worker index would be perfectly safe to overwrite. That would
need the worker *count*, which no stage config exposes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def exit_for_overwrite(cfg: DictConfig) -> bool:
    """Should this worker stop immediately because ``do_overwrite`` disables sharing?

    ``True`` for every worker except worker 0 when ``do_overwrite`` is set. Worker 0
    then performs the whole stage alone, which is correct but serial. Returns ``False``
    for any ordinary run, so a serial run (``worker`` defaults to 0) and every
    ``do_overwrite=False`` run are untouched.

    Args:
        cfg: The stage config. Reads ``do_overwrite`` and ``worker``, both of which
            MEDS-transforms always supplies.

    Examples:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.create({"do_overwrite": False, "worker": 0})
        >>> exit_for_overwrite(cfg)
        False

        A parallel run WITHOUT ``do_overwrite`` is the normal case — every worker
        participates:

        >>> exit_for_overwrite(OmegaConf.create({"do_overwrite": False, "worker": 3}))
        False

        Worker 0 always participates, so the stage still runs to completion:

        >>> exit_for_overwrite(OmegaConf.create({"do_overwrite": True, "worker": 0}))
        False

        Any other worker stands down when ``do_overwrite`` is set:

        >>> exit_for_overwrite(OmegaConf.create({"do_overwrite": True, "worker": 1}))
        True
    """
    if not (cfg.get("do_overwrite", False) and cfg.get("worker", 0) != 0):
        return False

    logger.warning(
        f"do_overwrite=True disables parallel work-sharing, so worker {cfg.worker} is standing "
        "down and worker 0 will run this stage alone. Workers divide work by skipping outputs a "
        "sibling already produced, which do_overwrite switches off — leaving every worker to "
        "redo every output (and, in extract_code_metadata, to delete partials the reducer is "
        "reading). To re-extract from scratch WITH parallelism, delete the stage's output "
        "directory and run without do_overwrite instead."
    )
    return True
