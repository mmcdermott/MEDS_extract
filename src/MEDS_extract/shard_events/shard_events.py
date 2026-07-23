from __future__ import annotations

import logging
import random
from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING

import polars as pl
from MEDS_transforms.dataframe import write_df
from MEDS_transforms.mapreduce.rwlock import rwlock_wrap
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig, OmegaConf
from upath import UPath

from .._stage_example import MEDSExtractStageExample
from ..config import MessyConfig
from ..io import ROW_IDX_NAME, SOURCE_FILE_COL, resolve_source_files, scan_source

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def filter_to_row_chunk(df: pl.LazyFrame, start: int, end: int) -> pl.LazyFrame:
    """Filters the input LazyFrame to a specific row chunk.

    This function is a simple helper designed to make other code clearer. The lazyframe must have a row index
    column named `ROW_IDX_NAME`. That column is *kept* in the output — together with the ``SOURCE_FILE_COL``
    literal stamped at read time, it is the row-provenance anchor that downstream stages rely on.

    Args:
        df: The input LazyFrame.
        start: The starting row index (inclusive).
        end: The ending row index (exclusive).

    Returns:
        The dataframe with only the rows in the range [`start`, `end`), with the row index column retained.

    Examples:
        >>> df = pl.DataFrame({ROW_IDX_NAME: [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]})
        >>> filter_to_row_chunk(df.lazy(), 1, 3).collect()
        shape: (2, 2)
        ┌─────────────┬─────┐
        │ __row_idx__ ┆ b   │
        │ ---         ┆ --- │
        │ i64         ┆ i64 │
        ╞═════════════╪═════╡
        │ 1           ┆ 6   │
        │ 2           ┆ 7   │
        └─────────────┴─────┘
        >>> filter_to_row_chunk(df.lazy(), 100, 300).collect()
        shape: (0, 2)
        ┌─────────────┬─────┐
        │ __row_idx__ ┆ b   │
        │ ---         ┆ --- │
        │ i64         ┆ i64 │
        ╞═════════════╪═════╡
        └─────────────┴─────┘
    """

    return df.filter(pl.col(ROW_IDX_NAME).is_between(start, end, closed="left"))


@Stage.register(is_metadata=False, example_class=MEDSExtractStageExample)
def main(cfg: DictConfig):
    """Runs the input data re-sharding process. Can be parallelized across output shards.

    This stage takes the raw input files and splits them into smaller files by taking consecutive chunks of
    rows and writing them out to new files. This is useful for parallelizing the processing of the input data.
    There is no randomization or re-ordering of the input data, and furthermore read contention on the input
    files being split may render additional parallelism beyond one worker per input file ineffective.

    Every output sub-shard carries two provenance anchor columns: ``ROW_IDX_NAME`` (the 0-based row index
    within the original source file) and ``SOURCE_FILE_COL`` (the input-dir-relative path of that file).
    These are intermediate-only annotations — ``convert_to_MEDS_events`` strips them (or folds them into a
    ``provenance`` column when ``do_track_provenance`` is enabled).

    All arguments are specified through the command line into the `cfg` object through Hydra.

    The `cfg.stage_cfg` object is a special key that is imputed by OmegaConf to contain the stage-specific
    configuration arguments based on the global, pipeline-level configuration file.

    Args:
        row_chunksize: The number of rows to read in at a time.
        infer_schema_length: The number of rows to read in to infer the
            schema (only used if the source files are csvs).
    """

    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )

    raw_cohort_dir = UPath(cfg.stage_cfg.data_input_dir).parent

    row_chunksize = cfg.stage_cfg.row_chunksize

    messy_cfg = MessyConfig.load(cfg.event_conversion_config_fp)
    prefix_to_columns = messy_cfg.needed_source_columns()

    # The anchor column names are unconditionally stamped onto every sub-shard below, so a real
    # source column with either name would be silently clobbered (or collide at scan time).
    for prefix, columns in prefix_to_columns.items():
        reserved = sorted({ROW_IDX_NAME, SOURCE_FILE_COL} & set(columns))
        if reserved:
            raise ValueError(
                f"Source table '{prefix}' uses reserved column name(s) {reserved}. These names are "
                f"reserved for the provenance anchor columns added by shard_events; rename the source "
                f"column(s) or adjust the event conversion config."
            )

    # Resolve each prefix to its source file(s). A prefix may resolve to multiple
    # files (sub-sharded directory layout). All chunks for a prefix land in the
    # same `{output}/{prefix}/` directory — for the multi-file case we prefix the
    # chunk filename with the source stem so distinct source files don't collide
    # on `[{start}-{end}).parquet`. The output stays flat under `{prefix}/` so
    # downstream `resolve_source_files` (which globs `{prefix}/*.parquet`
    # non-recursively) can find everything uniformly.
    files_to_process: list[tuple[str, Path | UPath, str]] = []
    for prefix in prefix_to_columns:
        try:
            fps = resolve_source_files(raw_cohort_dir, prefix)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"No raw source file found for prefix '{prefix}' under {raw_cohort_dir.resolve()!s}."
            ) from e
        # Only multi-file prefixes need per-file chunk-name disambiguation.
        needs_stem_prefix = len(fps) > 1
        for fp in fps:
            stem_prefix = f"{fp.stem}_" if needs_stem_prefix else ""
            files_to_process.append((prefix, fp, stem_prefix))

    random.shuffle(files_to_process)

    subsharding_files_strs = "\n".join([f"  * {fp.resolve()!s}" for _, fp, _ in files_to_process])
    logger.info(
        f"Starting event sub-sharding. Sub-sharding {len(files_to_process)} files:\n{subsharding_files_strs}"
    )

    raw_opts = cfg.get("cloud_io_storage_options", {})
    cloud_io_storage_options = OmegaConf.to_container(raw_opts) if OmegaConf.is_config(raw_opts) else raw_opts

    start = datetime.now(tz=UTC)
    for prefix, input_file, chunk_name_prefix in files_to_process:
        columns = prefix_to_columns[prefix]

        out_dir = UPath(cfg.stage_cfg.output_dir) / prefix
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Processing {input_file} to {out_dir}.")

        scan_kwargs = {
            "row_index_name": ROW_IDX_NAME,
            "infer_schema_length": cfg.stage_cfg.infer_schema_length,
            "storage_options": cloud_io_storage_options,
        }

        # The source-file anchor is the *input-dir-relative* path, so provenance stays
        # meaningful when the cohort directory moves (or lives on a cloud store).
        source_file_rel = str(input_file.relative_to(raw_cohort_dir))

        def _read_with_row_idx(fp, _columns=columns, _kwargs=scan_kwargs, _source_file=source_file_rel):
            df = scan_source(fp, **_kwargs)
            if _columns:
                df = df.select([ROW_IDX_NAME, *_columns])
            return df.with_columns(pl.lit(_source_file).alias(SOURCE_FILE_COL))

        df = _read_with_row_idx(input_file)
        row_count = df.select(pl.len()).collect().item()

        if row_count == 0:
            raise ValueError(
                f"File {input_file.resolve()!s} has no rows! If this is not an error, exclude it from "
                f"the event conversion configuration at {cfg.event_conversion_config_fp}."
            )

        logger.info(f"Read {row_count} rows from {input_file.resolve()!s}.")

        row_shards = list(range(0, row_count, row_chunksize))
        random.shuffle(row_shards)
        logger.info(f"Splitting {input_file} into {len(row_shards)} row-chunks of size {row_chunksize}.")

        for i, st in enumerate(row_shards):
            end = min(st + row_chunksize, row_count)
            out_fp = out_dir / f"{chunk_name_prefix}[{st}-{end}).parquet"

            compute_fn = partial(filter_to_row_chunk, start=st, end=end)
            logger.info(
                f"Writing file {i + 1}/{len(row_shards)}: {input_file} row-chunk [{st}-{end}) to {out_fp}."
            )
            rwlock_wrap(
                input_file,
                out_fp,
                _read_with_row_idx,
                write_df,
                compute_fn,
                do_overwrite=cfg.do_overwrite,
            )
    end = datetime.now(tz=UTC)
    logger.info(f"Sub-sharding completed in {datetime.now(tz=UTC) - start}")
