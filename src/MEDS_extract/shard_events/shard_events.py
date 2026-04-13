import logging
import random
from collections.abc import Sequence
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import polars as pl
from MEDS_transforms.dataframe import write_df
from MEDS_transforms.mapreduce.rwlock import rwlock_wrap
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig, OmegaConf
from upath import UPath

from ..config import MessyConfig, _resolve_source_files, _scan_file

logger = logging.getLogger(__name__)

ROW_IDX_NAME = "__row_idx__"


def get_shard_prefix(base_path: Path | UPath, fp: Path | UPath) -> str:
    """Extracts the shard prefix from a file path by removing the raw_cohort_dir.

    Args:
        base_path: The base path to remove.
        fp: The file path to extract the shard prefix from.

    Returns:
        The shard prefix (the file path relative to the base path with the suffix removed).

    Examples:
        >>> get_shard_prefix(Path("/a/b/c"), Path("/a/b/c/d.parquet"))
        'd'
        >>> get_shard_prefix(Path("/a/b/c"), Path("/a/b/c/d/e.csv.gz"))
        'd/e'
    """

    relative_path = fp.relative_to(base_path)
    relative_parent = relative_path.parent
    file_name = relative_path.name.split(".")[0]

    return str(relative_parent / file_name)


def kwargs_strs(kwargs: dict) -> str:
    """Returns a string representation of the kwargs dictionary for logging.

    Args:
        kwargs: A dictionary of keyword arguments.

    Returns: A string with each key-value pair in the dictionary formatted as a bullet point,
        newline-separated. The order of the key-value pairs is the order of the dictionary.

    Examples:
        >>> print(kwargs_strs({"a": 1, "b": "two", "c": 3.0}))
          * a=1
          * b=two
          * c=3.0
        >>> print(kwargs_strs({}))
        <BLANKLINE>
    """
    return "\n".join([f"  * {k}={v}" for k, v in kwargs.items()])


def scan_with_row_idx(fp: Path, columns: Sequence[str], **scan_kwargs) -> pl.LazyFrame:
    """Scan a source file and add a row-index column named ``ROW_IDX_NAME``.

    Thin wrapper around :func:`MEDS_extract.config._scan_file` that threads in
    the row-index kwarg and projects down to ``columns``. Format dispatch
    (parquet vs csv vs csv.gz) lives in ``_scan_file``.

    Examples:
        >>> from tempfile import TemporaryDirectory
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, schema={"a": pl.UInt8, "b": pl.Int64})
        >>> with TemporaryDirectory() as tmpdir:
        ...     fp = Path(tmpdir) / "test.parquet"
        ...     df.write_parquet(fp)
        ...     scan_with_row_idx(fp, columns=["a", "b"]).collect()
        shape: (3, 3)
        ┌─────────────┬─────┬─────┐
        │ __row_idx__ ┆ a   ┆ b   │
        │ ---         ┆ --- ┆ --- │
        │ u32         ┆ u8  ┆ i64 │
        ╞═════════════╪═════╪═════╡
        │ 0           ┆ 1   ┆ 4   │
        │ 1           ┆ 2   ┆ 5   │
        │ 2           ┆ 3   ┆ 6   │
        └─────────────┴─────┴─────┘
    """
    df = _scan_file(fp, row_index_name=ROW_IDX_NAME, **scan_kwargs)
    if columns:
        df = df.select([ROW_IDX_NAME, *columns])
    return df


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

    # Resolve each prefix in the config to its source file via the unified reader,
    # iterating tables and join targets alike. A missing source is a hard error —
    # shard_events is the raw-data entry point, so every prefix in the config has
    # to have a backing file.
    prefixes_to_process: list[tuple[str, Path | UPath]] = []
    for prefix in prefix_to_columns:
        try:
            fps = _resolve_source_files(raw_cohort_dir, prefix)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"No raw source file found for prefix '{prefix}' under {raw_cohort_dir.resolve()!s}."
            ) from e
        if len(fps) != 1:
            raise ValueError(
                f"shard_events expects exactly one source file per prefix, but found {len(fps)} "
                f"for '{prefix}' under {raw_cohort_dir.resolve()!s}. Pre-subsharded inputs should "
                f"skip this stage and enter at split_and_shard_subjects."
            )
        prefixes_to_process.append((prefix, fps[0]))

    random.shuffle(prefixes_to_process)

    subsharding_files_strs = "\n".join([f"  * {fp.resolve()!s}" for _, fp in prefixes_to_process])
    logger.info(
        f"Starting event sub-sharding. Sub-sharding {len(prefixes_to_process)} files:\n"
        f"{subsharding_files_strs}"
    )

    raw_opts = cfg.get("cloud_io_storage_options", {})
    cloud_io_storage_options = OmegaConf.to_container(raw_opts) if OmegaConf.is_config(raw_opts) else raw_opts

    start = datetime.now(tz=UTC)
    for prefix, input_file in prefixes_to_process:
        columns = prefix_to_columns[prefix]

        out_dir = UPath(cfg.stage_cfg.output_dir) / prefix
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Processing {input_file} to {out_dir}.")

        logger.info(f"Performing preliminary read of {input_file.resolve()!s} to determine row count.")

        scan_kwargs = {
            "columns": columns,
            "infer_schema_length": cfg.stage_cfg.infer_schema_length,
            "storage_options": cloud_io_storage_options,
        }

        df = scan_with_row_idx(input_file, **scan_kwargs)

        row_count = df.select(pl.len()).collect().item()

        if row_count == 0:
            logger.warning(
                f"File {input_file.resolve()!s} reports "
                f"`df.select(pl.len()).collect().item()={row_count}`. Trying to debug"
            )
            logger.warning(f"Columns: {', '.join(df.columns)}")
            logger.warning(f"First 10 rows:\n{df.head(10).collect()}")
            logger.warning(f"Last 10 rows:\n{df.tail(10).collect()}")
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
                partial(scan_with_row_idx, **scan_kwargs),
                write_df,
                compute_fn,
                do_overwrite=cfg.do_overwrite,
            )
    end = datetime.now(tz=UTC)
    logger.info(f"Sub-sharding completed in {datetime.now(tz=UTC) - start}")
