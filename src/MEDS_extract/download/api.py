"""Library surface for staging a MESSY spec's ``sources:`` block into a local directory.

:func:`stage_sources` is the download layer's whole-spec orchestration: bucket
selection (+ the always-appended ``common``), per-bucket interpolation resolution,
source construction, cross-source destination validation, and the pooled
``download_all`` loop. It is the single implementation behind both CLIs —
``meds-extract-download`` (:mod:`.cli`) wraps it with Hydra argument handling and
exit-code mapping, and ``meds-extract-run`` calls it in-process before invoking the
pipeline — so the two cannot drift in semantics.

Error signaling is exception-based (this is a library, not a process): user config
mistakes raise :class:`ValueError`/:class:`TypeError` with a message naming the spec
file, and transport-level failures raise :class:`DownloadError` after per-source
logging. Callers map those to exit codes.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

from .source import validate_unique_destinations
from .spec import sources_from_spec

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class DownloadError(Exception):
    """At least one source failed while fetching (the spec was valid; a transfer was not).

    Raised by :func:`stage_sources` after the per-source failure has already been
    logged (with traceback) — callers should treat it as "downloads failed, details
    are in the log" and exit non-zero without re-raising a wall of text.
    """


def stage_sources(
    spec_fp: Path,
    raw_input_dir: Path,
    *,
    key: str = "dataset",
    concurrency: int = 4,
    continue_on_error: bool = False,
    do_overwrite: bool = False,
) -> None:
    """Stage every file from the spec's selected ``sources:`` bucket under ``raw_input_dir``.

    Reads the MESSY spec at ``spec_fp``, selects the ``key`` bucket plus the
    always-appended ``common`` bucket, constructs the configured sources, and runs
    each source's :meth:`~MEDS_extract.download.source.Source.download_all` in
    sequence over one shared :class:`~concurrent.futures.ThreadPoolExecutor` of
    ``concurrency`` workers (a global per-file transport cap, not per-source).

    Interpolations are resolved on ONLY the selected buckets. Under the
    combined-MESSY pattern (one file carrying ``sources:``, ``etl:``, and
    event-conversion entries), resolving the whole document would require every
    ``${oc.env:...}`` in unrelated sections to be set just to download — and
    resolving all of ``sources:`` would likewise require unselected buckets'
    credentials (e.g. a credentialed ``dataset`` bucket's env vars just to pull
    ``key=demo``). This is the symmetric sibling of ``MessyConfig.parse``'s "strip
    reserved keys before resolve=True" fix — the layers coexist without
    cross-polluting env requirements. Each selected bucket is resolved while still
    ATTACHED to the loaded document, so document-relative interpolations (e.g.
    ``${sources.demo.0.root}``) keep resolving; detaching the bucket first would
    break them.

    Args:
        spec_fp: Path to the MESSY spec YAML with a ``sources:`` block.
        raw_input_dir: Destination directory under which fetched files land.
        key: Which ``sources:`` bucket to pull; ``"common"`` is always appended.
        concurrency: Max parallel transport streams across all sources.
        continue_on_error: If ``True``, per-file failures don't sink the run; every
            source is still attempted and :class:`DownloadError` is raised at the end
            if anything failed. If ``False``, the first failing source stops the run.
        do_overwrite: If ``True``, re-fetch every file even if the local copy matches.

    Raises:
        ValueError: On user config mistakes — a ``key`` naming no bucket, a malformed
            source entry, or manifest-validation failures (duplicate destinations,
            malformed rows). Also ``TypeError`` for backend-kwarg type errors.
        DownloadError: If sources constructed fine but at least one download failed
            (the per-source failure is logged with traceback before raising).

    Examples:
        A local ``fsspec`` mirror stages end-to-end (no network, no extras):

        >>> with yaml_disk('''
        ... mirror:
        ...   patients.csv: "patient_id,dob\\\\n1,2000-01-01\\\\n"
        ... ''') as d:
        ...     spec_fp = Path(d) / "spec.yaml"
        ...     src_yaml = f"sources:\\n  dataset:\\n    - type: fsspec\\n      root: {d}/mirror\\n"
        ...     _ = spec_fp.write_text(src_yaml)
        ...     stage_sources(spec_fp, Path(d) / "raw")
        ...     (Path(d) / "raw" / "patients.csv").read_text()
        'patient_id,dob\\n1,2000-01-01'

        A ``key`` naming no declared bucket is a config error (likely a typo), not an
        empty download — because ``common`` is always appended, a typo'd key would
        otherwise quietly fetch only the common bucket (or nothing) and "succeed":

        >>> with yaml_disk("mirror: {}") as d:
        ...     spec_fp = Path(d) / "spec.yaml"
        ...     src_yaml = f"sources:\\n  dataset:\\n    - type: fsspec\\n      root: {d}/mirror\\n"
        ...     _ = spec_fp.write_text(src_yaml)
        ...     stage_sources(spec_fp, Path(d) / "raw", key="dataste")
        Traceback (most recent call last):
            ...
        ValueError: key='dataste' does not name a sources bucket in ...spec.yaml. Available
        buckets: ['dataset'].

        A spec with no ``sources:`` block at all is a legitimately download-free ETL:
        it warns and returns (nothing staged, no error), regardless of ``key``:

        >>> with yaml_disk("spec.yaml: 'patients: {dob: {code: BIRTH, time: null}}'") as d:
        ...     stage_sources(Path(d) / "spec.yaml", Path(d) / "raw")
        ...     (Path(d) / "raw").exists()
        False
    """
    spec_raw = OmegaConf.load(spec_fp)
    sources_node = spec_raw.get("sources")

    # A key that names no bucket is a config error (likely a typo), not an empty
    # download. Bucket names come from the UNRESOLVED node — listing them must not
    # require any interpolation (in any bucket) to be resolvable.
    if sources_node and key not in sources_node:
        raise ValueError(
            f"key={key!r} does not name a sources bucket in {spec_fp}. "
            f"Available buckets: {sorted(sources_node)}."
        )

    sources_dict = {}
    if sources_node is not None:
        for bucket in dict.fromkeys((key, "common")):  # de-dupe when key="common"
            bucket_node = sources_node.get(bucket)
            if bucket_node is not None:
                sources_dict[bucket] = OmegaConf.to_container(bucket_node, resolve=True)

    # Spec-shape errors (missing/unknown ``type:``, bad backend kwargs) are user
    # config mistakes: re-raise with the spec path named so callers can surface one
    # actionable line instead of a raw traceback.
    try:
        sources = sources_from_spec({"sources": sources_dict}, key=key)
    except (TypeError, ValueError) as e:
        raise type(e)(f"Could not construct sources from the spec at {spec_fp}: {e}") from e

    if not sources:
        logger.warning(f"No sources resolved for key={key!r} in {spec_fp}. Nothing to do.")
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
        pool = ThreadPoolExecutor(max_workers=concurrency)
        stack.callback(pool.shutdown, wait=False, cancel_futures=True)
        for source in sources:
            stack.enter_context(source)

        # Materializes every source's manifest up-front (cached for the fetch loop
        # below) and fails before any fetch into raw_input_dir if a manifest row is
        # malformed or two sources would write the same file. (Manifest listing
        # itself may do network I/O — e.g. PhysioNet's SHA256SUMS.txt GET, fsspec
        # source-side hashing.)
        try:
            validate_unique_destinations(sources)
        except ValueError as e:
            raise ValueError(f"Source manifests failed validation: {e}") from e

        all_ok = True
        for source in sources:
            try:
                source.download_all(
                    raw_input_dir,
                    pool=pool,
                    continue_on_error=continue_on_error,
                    do_overwrite=do_overwrite,
                )
            except Exception:
                logger.exception(f"download_all failed for {type(source).__name__}")
                all_ok = False
                if not continue_on_error:
                    # Fail fast applies across sources too: don't start source N+1
                    # after source N has already sunk the run.
                    break
        if not all_ok:
            raise DownloadError(f"At least one source failed downloading into {raw_input_dir}.")
