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
  whose three passes reproduce polars' full-file type inference without materializing
  the file (round-trip parity is property-tested in ``tests/test_convert_to_parquet.py``;
  the one lenience — an integer column overflowing Int64 degrades to Float64 where
  polars' full read hard-errors — is documented there). Dtypes therefore match what
  ``shard_events`` produced, so extracted output is unchanged.
- **parquet / par** — hardlinked when the file carries only columns the config reads
  (copied across filesystems if links are unavailable), since downstream scans push
  projection into the parquet reader themselves. A source with extra columns is
  rewritten projected instead: ``convert_to_subject_sharded`` does not project, so an
  unpruned column is copied into its output too.

Staged input (``input_overlay_dirs``): when the spec declares ``etl.pre_stages``,
``meds-extract-run`` passes their output dirs here, and each source prefix is read
from the first of those dirs that holds it, falling back to ``input_dir`` (see
:func:`~MEDS_extract.io.resolve_raw_source_files`). A pre-stage therefore writes only
the tables it adds or reshapes; the raw tree is never copied or modified.

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
from collections import defaultdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from MEDS_transforms.mapreduce.rwlock import run_marker_dir, rwlock_wrap
from MEDS_transforms.stages import Stage
from upath import UPath

from .._stage_example import MEDSExtractStageExample
from ..config import MessyConfig
from ..encoding import AUTO_ENCODING, source_encoding
from ..io import (
    DEFAULT_CSV_SEPARATOR,
    _format_family,
    convert_csv_to_parquet,
    resolve_raw_source_files,
    source_separator,
)

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


def _convert_one(
    src: Path | UPath,
    dest: Path,
    *,
    prefix: str,
    columns: list[str] | None,
    encoding: str,
    separator: str = DEFAULT_CSV_SEPARATOR,
) -> None:
    """Materialize one source file at ``dest`` as parquet — the ``rwlock_wrap`` write step.

    csv-family sources go through the three-pass streaming conversion, decoded as
    ``encoding`` and split on ``separator`` (``prefix``'s codec and field separator under
    the ``etl.encoding`` and ``etl.csv_separator`` options; parquet carries neither, so
    they do not apply there). A parquet source is
    *linked* when there is nothing to prune, and rewritten with a projection when it
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
        schema = convert_csv_to_parquet(src, dest, columns, encoding=encoding, separator=separator)
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


def _dest_for(fp: Path | UPath, input_dir: Path | UPath, out_dir: Path | UPath) -> Path:
    """Destination parquet path for a source file: same relative position, format suffixes shed.

    One output per input file, so a table that arrives pre-sharded across many
    files stays sharded exactly as it was.
    """
    rel = fp.relative_to(input_dir).with_suffix("")
    if str(rel).endswith(".csv"):  # ``.csv.gz`` sheds only ``.gz`` above
        rel = rel.with_suffix("")
    return Path(out_dir / f"{rel}.parquet")


def _reject_dest_collisions(fps: list[Path | UPath], input_dir: Path | UPath, out_dir: Path | UPath) -> None:
    """Raise when two source files map to the same converted output.

    Suffix-stripping maps same-stem sources in different encodings (``a.csv`` +
    ``a.csv.gz``) onto one destination; whichever converted first would win and
    the other chunk's rows would silently vanish as a cache hit on the existing
    output, so collisions are an error naming both sources.

    Examples:
        >>> _reject_dest_collisions([Path("/in/t/a.csv"), Path("/in/t/b.csv")], Path("/in"), Path("/out"))
        >>> _reject_dest_collisions(
        ...     [Path("/in/t/a.csv"), Path("/in/t/a.csv.gz")], Path("/in"), Path("/out")
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Multiple source files map to the same converted output:
        ['/in/t/a.csv', '/in/t/a.csv.gz'] -> /out/t/a.parquet. Whichever converted first would
        silently shadow the other's rows. Rename or remove the duplicate source files.
    """
    by_dest: dict[Path, list[Path | UPath]] = defaultdict(list)
    for fp in fps:
        by_dest[_dest_for(fp, input_dir, out_dir)].append(fp)
    collisions = {d: v for d, v in by_dest.items() if len(v) > 1}
    if collisions:
        desc = "; ".join(f"{sorted(str(f) for f in v)} -> {d}" for d, v in sorted(collisions.items()))
        raise ValueError(
            f"Multiple source files map to the same converted output: {desc}. Whichever "
            f"converted first would silently shadow the other's rows. Rename or remove the "
            f"duplicate source files."
        )


@Stage.register(is_metadata=False, example_class=MEDSExtractStageExample)
def main(cfg: DictConfig):
    """Normalize every needed raw source table into the stage output as parquet.

    Reads ``cfg.input_dir`` directly rather than deriving it from
    ``stage_cfg.data_input_dir``. As the pipeline's first stage its ``data_input_dir``
    is ``<input_dir>/data``, which is where the *cohort* lives, not where the user's
    raw files are — ``shard_events`` compensated with a ``.parent`` walk-up, and that
    sleight of hand is exactly what made the pipeline unrunnable without it (#186).
    Naming ``input_dir`` explicitly keeps raw-data resolution in one honest place.

    Args:
        stage_cfg.encoding: The text encoding of csv sources: ``auto`` (the default;
            detected per file), one Python codec name, or a mapping of source prefix to
            either, under which unlisted prefixes are ``auto``.
        stage_cfg.csv_separator: The field separator of csv sources: ``","`` (the default),
            one other single-byte character such as ``";"`` or ``"\\t"``, or a mapping of
            source prefix to one, under which unlisted prefixes are ``","``.
        stage_cfg.input_overlay_dirs: Staged-input directories searched, in order, before
            ``input_dir`` for each source prefix (default: none). ``meds-extract-run``
            sets it to the output dirs of the spec's ``etl.pre_stages``, last stage first.
    """
    input_dir = UPath(cfg.input_dir)
    out_dir = UPath(cfg.stage_cfg.output_dir)
    overlay_dirs = [UPath(d) for d in cfg.stage_cfg.get("input_overlay_dirs") or ()]

    messy_cfg = MessyConfig.load(cfg.MESSY_config_fp)
    prefix_to_columns = messy_cfg.needed_source_columns()

    # Resolve every needed prefix before converting anything, so a missing table fails
    # before we have written half an output tree. Each prefix resolves in one directory
    # (a staged-input overlay, else ``input_dir``), and its outputs are placed relative
    # to that directory.
    work: list[tuple[str, Path | UPath, Path | UPath]] = []
    for prefix in prefix_to_columns:
        try:
            root, fps = resolve_raw_source_files(input_dir, prefix, overlay_dirs)
        except FileNotFoundError as e:
            staged = f" or the staged input dirs {[str(d) for d in overlay_dirs]}" if overlay_dirs else ""
            raise FileNotFoundError(
                f"No raw source file found for prefix '{prefix}' under {input_dir}{staged}."
            ) from e
        work.extend((prefix, root, fp) for fp in fps)

    by_root: dict[Path | UPath, list[Path | UPath]] = defaultdict(list)
    for _, root, fp in work:
        by_root[root].append(fp)
    for root, fps in by_root.items():
        _reject_dest_collisions(fps, root, out_dir)

    # Shuffled so parallel workers spread across tables instead of contending on the
    # first one; each output is locked independently by ``rwlock_wrap`` semantics.
    random.shuffle(work)
    logger.info(f"Converting {len(work)} source file(s) from {sorted(map(str, by_root))} to parquet.")

    start = datetime.now(tz=UTC)
    for prefix, root, fp in work:
        dest = _dest_for(fp, root, out_dir)

        # ``rwlock_wrap`` gives the caching + cross-worker locking every other stage
        # relies on. Passing the PATH through ``read_fn``/``compute_fn`` (rather than a
        # dataframe) is what keeps the conversion streaming: nothing is materialized to
        # satisfy the wrapper, and ``write_fn`` does the real work.
        rwlock_wrap(
            fp,
            dest,
            read_fn=lambda src: src,
            write_fn=partial(
                _convert_one,
                prefix=prefix,
                columns=prefix_to_columns[prefix] or None,
                encoding=source_encoding(cfg.stage_cfg.get("encoding", AUTO_ENCODING), prefix),
                separator=source_separator(cfg.stage_cfg.get("csv_separator", DEFAULT_CSV_SEPARATOR), prefix),
            ),
            compute_fn=lambda src: src,
            do_overwrite=cfg.do_overwrite,
            # Run-scoped do_overwrite (MT 0.7.0): without the marker dir, parallel
            # workers treat each other's fresh outputs as stale and redo the work.
            marker_dir=run_marker_dir(cfg),
        )

    logger.info(f"Raw-table conversion completed in {datetime.now(tz=UTC) - start}")
