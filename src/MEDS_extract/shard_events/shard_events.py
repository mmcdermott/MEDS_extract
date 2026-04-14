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

from ..config import MessyConfig
from ..io import resolve_source_files, scan_source

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

ROW_IDX_NAME = "__row_idx__"


def filter_to_row_chunk(df: pl.LazyFrame, start: int, end: int) -> pl.LazyFrame:
    """Filters the input LazyFrame to a specific row chunk.

    This function is a simple helper designed to make other code clearer. The lazyframe must have a row index
    column named `ROW_IDX_NAME`.

    Args:
        df: The input LazyFrame.
        start: The starting row index (inclusive).
        end: The ending row index (exclusive).

    Returns:
        The dataframe with only the rows in the range [`start`, `end`), and with the row index column dropped.

    Examples:
        >>> df = pl.DataFrame({ROW_IDX_NAME: [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]})
        >>> filter_to_row_chunk(df.lazy(), 1, 3).collect()
        shape: (2, 1)
        ┌─────┐
        │ b   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 6   │
        │ 7   │
        └─────┘
        >>> filter_to_row_chunk(df.lazy(), 100, 300).collect()
        shape: (0, 1)
        ┌─────┐
        │ b   │
        │ --- │
        │ i64 │
        ╞═════╡
        └─────┘
    """

    return df.filter(pl.col(ROW_IDX_NAME).is_between(start, end, closed="left")).drop(ROW_IDX_NAME)


@Stage.register(is_metadata=False)
def main(cfg: DictConfig):
    """Runs the input data re-sharding process. Can be parallelized across output shards.

    This stage takes the raw input files and splits them into smaller files by taking consecutive chunks of
    rows and writing them out to new files. This is useful for parallelizing the processing of the input data.
    There is no randomization or re-ordering of the input data, and furthermore read contention on the input
    files being split may render additional parallelism beyond one worker per input file ineffective.

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

    # Resolve each prefix to its source file(s). A prefix may resolve to multiple
    # files (sub-sharded directory layout); each file gets a derived output prefix
    # that includes the file stem so per-file row-chunk filenames don't collide
    # under the shared output directory.
    # Tuples are (output_prefix, source_prefix, input_file) — output_prefix tells
    # us where to write, source_prefix tells us which column set to project.
    files_to_process: list[tuple[str, str, Path | UPath]] = []
    for prefix in prefix_to_columns:
        try:
            fps = resolve_source_files(raw_cohort_dir, prefix)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"No raw source file found for prefix '{prefix}' under {raw_cohort_dir.resolve()!s}."
            ) from e
        if len(fps) == 1:
            files_to_process.append((prefix, prefix, fps[0]))
        else:
            for fp in fps:
                # Derive a per-file output prefix from the file stem so chunks
                # land at {output}/{prefix}/{stem}/[{start}-{end}).parquet rather
                # than colliding at {output}/{prefix}/[{start}-{end}).parquet.
                files_to_process.append((f"{prefix}/{fp.stem}", prefix, fp))

    random.shuffle(files_to_process)

    subsharding_files_strs = "\n".join([f"  * {fp.resolve()!s}" for _, _, fp in files_to_process])
    logger.info(
        f"Starting event sub-sharding. Sub-sharding {len(files_to_process)} files:\n{subsharding_files_strs}"
    )

    raw_opts = cfg.get("cloud_io_storage_options", {})
    cloud_io_storage_options = OmegaConf.to_container(raw_opts) if OmegaConf.is_config(raw_opts) else raw_opts

    start = datetime.now(tz=UTC)
    for output_prefix, source_prefix, input_file in files_to_process:
        columns = prefix_to_columns[source_prefix]

        out_dir = UPath(cfg.stage_cfg.output_dir) / output_prefix
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Processing {input_file} to {out_dir}.")

        scan_kwargs = {
            "row_index_name": ROW_IDX_NAME,
            "infer_schema_length": cfg.stage_cfg.infer_schema_length,
            "storage_options": cloud_io_storage_options,
        }

        def _read_with_row_idx(fp, _columns=columns, _kwargs=scan_kwargs):
            df = scan_source(fp, **_kwargs)
            if _columns:
                df = df.select([ROW_IDX_NAME, *_columns])
            return df

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
            out_fp = out_dir / f"[{st}-{end}).parquet"

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
