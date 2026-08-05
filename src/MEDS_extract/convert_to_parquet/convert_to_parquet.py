"""Stage: normalize every needed raw source table into projected parquet.

This is the pipeline's single raw-data ingress. It reads ``input_dir`` — the only
stage that touches user-supplied files — and writes one parquet per input file into
the stage output, preserving the source's relative layout. Everything downstream
reads that output through the ordinary stage chain, so no later stage needs to know
where the raw data lived or what format it was in.

It replaces ``shard_events``, which split each table into row-range chunks. That
chunking existed because an older polars could not subject-shard a very large table
directly; it is now pure cost. Chunk *k* of a csv source had to re-parse from byte 0
(a slice cannot be pushed into a CSV, still less a gzip stream), so an N-chunk table
paid N full parses — each one materializing the file, which is what OOM-killed
production runs on ``chartevents`` (#102). Nothing downstream ever used the chunk
boundaries: ``convert_to_subject_sharded`` reads every file of a table for every
subject shard regardless.

Format handling, per input file:

- **csv / csv.gz** — converted by :func:`~MEDS_extract.io.convert_csv_to_parquet`,
  whose three passes get full-file-accurate type inference without materializing the
  file. Dtypes match what ``shard_events`` produced, so extracted output is unchanged.
- **parquet / par** — hardlinked when the file carries only columns the config reads
  (copied across filesystems if links are unavailable), since downstream scans push
  projection into the parquet reader themselves. A source with extra columns is
  rewritten projected instead: ``convert_to_subject_sharded`` does not project, so an
  unpruned column is copied into its output too.

Only tables the MESSY config actually needs are converted — event tables and their
join targets. Metadata tables (``_metadata`` blocks) are deliberately NOT converted:
``extract_code_metadata`` reads them from ``input_dir`` with ``infer_schema=False``
on purpose, so that vocabulary keys keep their exact source text (``"01.90"`` must
not become ``1.9``). Converting them would either break that or need a second,
all-String artifact for the same file.
"""

from __future__ import annotations

import logging
import os
import random
import shutil
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from MEDS_transforms.mapreduce.rwlock import rwlock_wrap
from MEDS_transforms.stages import Stage
from upath import UPath

from .._stage_example import MEDSExtractStageExample
from ..config import MessyConfig
from ..io import _format_family, convert_csv_to_parquet, resolve_source_files

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def link_or_copy(src: Path, dest: Path) -> str:
    """Hardlink ``src`` to ``dest``, falling back to a copy across filesystems.

    A parquet source needs no conversion, so the cheap thing is to not move its bytes
    at all. A hardlink makes the file appear in the stage output for free; it falls
    back to a copy when the two paths live on different devices (or the filesystem
    refuses links).

    Returns ``"linked"`` or ``"copied"``, for the caller's log line.

    Examples:
        >>> with yaml_disk('src.parquet:\\n  a: [1, 2]') as d:
        ...     out = Path(d) / "nested" / "dest.parquet"
        ...     how = link_or_copy(Path(d) / "src.parquet", out)
        ...     (how, out.is_file(), pl.read_parquet(out)["a"].to_list())
        ('linked', True, [1, 2])

        An existing destination is replaced, so a re-run is not blocked by its own
        previous output:

        >>> with yaml_disk('''
        ... src.parquet:
        ...   a: [3]
        ... dest.parquet:
        ...   a: [999]
        ... ''') as d:
        ...     _ = link_or_copy(Path(d) / "src.parquet", Path(d) / "dest.parquet")
        ...     pl.read_parquet(Path(d) / "dest.parquet")["a"].to_list()
        [3]
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    try:
        os.link(src, dest)
        return "linked"
    except OSError:
        # Cross-device, or a filesystem without hardlinks.
        shutil.copy2(src, dest)
        return "copied"


def _convert_one(src: Path | UPath, dest: Path, *, prefix: str, columns: list[str] | None) -> None:
    """Materialize one source file at ``dest`` as parquet — the ``rwlock_wrap`` write step.

    csv-family sources go through the three-pass streaming conversion. A parquet source
    is *linked* when there is nothing to prune, and rewritten with a projection when it
    carries columns the config never reads.

    Downstream *reads* are unaffected either way — parquet is columnar, so a scan only
    touches the columns it projects. The reason to prune is downstream *writes*:
    ``convert_to_subject_sharded`` does not project, so every unread column is copied
    into its output. Each row lands in exactly one subject shard, so that is a one-time
    cost rather than a per-shard one, but for a wide source (MIMIC-IV ``chartevents``
    is 36 columns where a config typically reads ~5) it still means an intermediate
    several times larger than it needs to be, and a correspondingly wider frame in the
    stage already known to be the pipeline's memory hot spot (#76).

    One bounded streaming rewrite here is the cheaper side of that trade, and it keeps
    the column set identical to what ``shard_events`` produced.
    """
    if _format_family(src) != "parquet":
        schema = convert_csv_to_parquet(src, dest, columns)
        logger.info(f"{prefix}: converted {src} -> {dest} ({len(schema)} columns).")
        return

    lf = pl.scan_parquet(src, glob=False)
    have = lf.collect_schema().names()
    if columns is not None:
        missing = [c for c in columns if c not in have]
        if missing:
            raise ValueError(
                f"{src} is missing requested column(s) {sorted(missing)}. It has: {sorted(have)}."
            )

    if columns is None or not set(have) - set(columns):
        # Nothing to prune, so nothing to gain by rewriting: downstream scans push
        # projection into the parquet reader themselves.
        how = link_or_copy(Path(src), dest)
        logger.info(f"{prefix}: {how} {src} -> {dest} (already parquet, no columns to prune).")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        lf.select(columns).sink_parquet(dest)
        logger.info(f"{prefix}: projected {src} -> {dest} ({len(have)} -> {len(columns)} columns).")


@Stage.register(is_metadata=False, example_class=MEDSExtractStageExample)
def main(cfg: DictConfig):
    """Normalize every needed raw source table into the stage output as parquet.

    Reads ``cfg.input_dir`` directly rather than deriving it from
    ``stage_cfg.data_input_dir``. As the pipeline's first stage its ``data_input_dir``
    is ``<input_dir>/data``, which is where the *cohort* lives, not where the user's
    raw files are — ``shard_events`` compensated with a ``.parent`` walk-up, and that
    sleight of hand is exactly what made the pipeline unrunnable without it (#186).
    Naming ``input_dir`` explicitly keeps raw-data resolution in one honest place.
    """
    input_dir = UPath(cfg.input_dir)
    out_dir = UPath(cfg.stage_cfg.output_dir)

    messy_cfg = MessyConfig.load(cfg.MESSY_config_fp)
    prefix_to_columns = messy_cfg.needed_source_columns()

    # Resolve every needed prefix before converting anything, so a missing table fails
    # before we have written half an output tree.
    work: list[tuple[str, Path | UPath]] = []
    for prefix in prefix_to_columns:
        try:
            fps = resolve_source_files(input_dir, prefix)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"No raw source file found for prefix '{prefix}' under {input_dir}."
            ) from e
        work.extend((prefix, fp) for fp in fps)

    # Shuffled so parallel workers spread across tables instead of contending on the
    # first one; each output is locked independently by ``rwlock_wrap`` semantics.
    random.shuffle(work)
    logger.info(f"Converting {len(work)} source file(s) from {input_dir} to parquet.")

    start = datetime.now(tz=UTC)
    for prefix, fp in work:
        # One output per input file, at the same relative position, so a table that
        # arrives pre-sharded across many files stays sharded exactly as it was.
        rel = fp.relative_to(input_dir).with_suffix("")
        if str(rel).endswith(".csv"):  # ``.csv.gz`` sheds only ``.gz`` above
            rel = rel.with_suffix("")
        dest = Path(out_dir / f"{rel}.parquet")

        # ``rwlock_wrap`` gives the caching + cross-worker locking every other stage
        # relies on. Passing the PATH through ``read_fn``/``compute_fn`` (rather than a
        # dataframe) is what keeps the conversion streaming: nothing is materialized to
        # satisfy the wrapper, and ``write_fn`` does the real work.
        rwlock_wrap(
            fp,
            dest,
            read_fn=lambda src: src,
            write_fn=partial(_convert_one, prefix=prefix, columns=prefix_to_columns[prefix] or None),
            compute_fn=lambda src: src,
            do_overwrite=cfg.do_overwrite,
        )

    logger.info(f"Raw-table conversion completed in {datetime.now(tz=UTC) - start}")
