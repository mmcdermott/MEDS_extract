import copy
import gzip
import logging
import random
import warnings
from collections.abc import Sequence
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import polars as pl
from meds import DataSchema
from MEDS_transforms.dataframe import write_df
from MEDS_transforms.mapreduce.rwlock import rwlock_wrap
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig, OmegaConf
from upath import UPath

from ..parsing import is_col_field, parse_col_field

logger = logging.getLogger(__name__)

ROW_IDX_NAME = "__row_idx__"
# Keys in the event configuration that do not correspond to event columns.
META_KEYS = {"time_format", "_metadata", "join"}


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
    """Scans a file into a polars lazyframe and adds a row index with name `ROW_IDX_NAME`.

    Args:
        fp: The file path to read. Must be either a ".csv", ".csv.gz", or ".parquet" file.
        columns: A list of column names to read from the file.
        scan_kwargs: Additional keyword arguments to pass to the scan function. The `infer_schema_length`
            kwarg is removed for reading parquet files as it is not used for such files.

    Raises:
        ValueError: If the file type is not supported.

    Returns:
        A LazyFrame with the row index column added.

    Examples:
        >>> from tempfile import TemporaryDirectory
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, schema={"a": pl.UInt8, "b": pl.Int64})
        >>> with TemporaryDirectory() as tmpdir:
        ...     fp = Path(tmpdir) / "test.csv"
        ...     df.write_csv(fp)
        ...     scan_with_row_idx(fp, columns=["a"], infer_schema_length=40).collect()
        shape: (3, 2)
        ┌─────────────┬─────┐
        │ __row_idx__ ┆ a   │
        │ ---         ┆ --- │
        │ u32         ┆ i64 │
        ╞═════════════╪═════╡
        │ 0           ┆ 1   │
        │ 1           ┆ 2   │
        │ 2           ┆ 3   │
        └─────────────┴─────┘
        >>> with TemporaryDirectory() as tmpdir:
        ...     fp = Path(tmpdir) / "test.parquet"
        ...     df.write_parquet(fp)
        ...     scan_with_row_idx(fp, columns=["a", "b"], infer_schema_length=40).collect()
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
        >>> import gzip
        >>> with TemporaryDirectory() as tmpdir:
        ...     fp = Path(tmpdir) / "test.csv.gz"
        ...     with gzip.open(fp, mode="wb") as f:
        ...         with warnings.catch_warnings():
        ...             warnings.simplefilter("ignore", category=UserWarning)
        ...             df.write_csv(f)
        ...     scan_with_row_idx(fp, columns=["b"]).collect()
        shape: (3, 2)
        ┌─────────────┬─────┐
        │ __row_idx__ ┆ b   │
        │ ---         ┆ --- │
        │ u32         ┆ i64 │
        ╞═════════════╪═════╡
        │ 0           ┆ 4   │
        │ 1           ┆ 5   │
        │ 2           ┆ 6   │
        └─────────────┴─────┘
        >>> with TemporaryDirectory() as tmpdir:
        ...     fp = Path(tmpdir) / "test.json"
        ...     df.write_json(fp)
        ...     scan_with_row_idx(fp, columns=["a", "b"])
        Traceback (most recent call last):
            ...
        ValueError: Unsupported file type: .json
    """

    kwargs = {
        **scan_kwargs,
        "row_index_name": ROW_IDX_NAME,
    }
    match "".join(fp.suffixes).lower():
        case ".csv.gz":
            if columns:
                kwargs["columns"] = columns

            logger.debug(f"Reading {fp.resolve()!s} as compressed CSV with kwargs:\n{kwargs_strs(kwargs)}.")
            logger.warning("Reading compressed CSV files may be slow and limit parallelizability.")
            with gzip.open(fp, mode="rb") as f, warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                return pl.read_csv(f, **kwargs).lazy()
        case ".csv":
            logger.debug(f"Reading {fp.resolve()!s} as CSV with kwargs:\n{kwargs_strs(kwargs)}.")
            df = pl.scan_csv(fp, **kwargs)
        case ".parquet" | ".par":
            if "infer_schema_length" in kwargs:
                infer_schema_length = kwargs.pop("infer_schema_length")
                logger.info(f"Ignoring infer_schema_length={infer_schema_length} for Parquet files.")

            logger.debug(f"Reading {fp.resolve()!s} as Parquet with kwargs:\n{kwargs_strs(kwargs)}.")
            df = pl.scan_parquet(fp, **kwargs)
        case _:
            raise ValueError(f"Unsupported file type: {fp.suffix}")

    if columns:
        columns = [ROW_IDX_NAME, *columns]
        logger.debug(f"Selecting columns: {columns}")
        df = df.select(columns)

    logger.debug(f"Returning df with columns: {', '.join(df.collect_schema().names())}")
    return df


def retrieve_columns(event_conversion_cfg: DictConfig) -> dict[str, list[str]]:
    """Extracts and organizes column names from configuration for a list of files.

    This function processes each file specified in the 'files' list, reading the
    event conversion configurations that are specific to each file based on its
    stem (filename without the extension). It compiles a list of column names
    needed for each file from the configuration, which includes both general
    columns like row index and subject ID, as well as specific columns defined
    for medical events and times formatted in a special 'col(column_name)' syntax.

    Args:
        event_conversion_cfg (DictConfig): A dictionary configuration where
            each key is a filename stem and each value is a dictionary containing
            configuration details for different codes or events, specifying
            which columns to retrieve for each file.

    Returns:
        dict[Path, list[str]]: A dictionary mapping each file path to a list
        of unique column names necessary for processing the file.
        The list of columns includes generic columns and those specified in the 'event_conversion_cfg'.

    Examples:
        >>> cfg = DictConfig({
        ...     "subject_id_col": "subject_id_global",
        ...     "hosp/patients": {
        ...         "eye_color": {
        ...             "code": ["EYE_COLOR", "col(eye_color)"], "time": None, "mod": "mod_col"
        ...         },
        ...         "height": {
        ...             "code": "HEIGHT", "time": None, "numeric_value": "height"
        ...         }
        ...     },
        ...     "icu/chartevents": {
        ...         "subject_id_col": "subject_id_icu",
        ...         "heart_rate": {
        ...             "code": "HEART_RATE", "time": "charttime", "numeric_value": "HR"
        ...         },
        ...         "lab": {
        ...             "code": ["col(itemid)", "col(valueuom)"],
        ...             "time": "charttime",
        ...             "numeric_value": "valuenum",
        ...             "text_value": "value",
        ...             "mod": "mod_lab",
        ...         }
        ...     },
        ...     "icu/meds": {
        ...         "med": {"code": "col(medication)", "time": "medtime"}
        ...     }
        ... })
        >>> retrieve_columns(cfg)
        {'hosp/patients': ['eye_color', 'height', 'mod_col', 'subject_id_global'],
         'icu/chartevents': ['HR', 'charttime', 'itemid', 'mod_lab', 'subject_id_icu', 'value', 'valuenum',
                             'valueuom'],
         'icu/meds': ['medication', 'medtime', 'subject_id_global']}
        >>> cfg = DictConfig({
        ...     "subjects": {
        ...         "subject_id_col": "MRN",
        ...         "eye_color": {"code": ["col(eye_color)"], "time": None},
        ...     },
        ...     "labs": {"lab": {"code": "col(labtest)", "time": "charttime"}},
        ... })
        >>> retrieve_columns(cfg)
        {'subjects': ['MRN', 'eye_color'], 'labs': ['charttime', 'labtest', 'subject_id']}
    """

    event_conversion_cfg = copy.deepcopy(event_conversion_cfg)

    # Initialize a dictionary to store file paths as keys and lists of column names as values.
    prefix_to_columns = {}

    default_subject_id_col = event_conversion_cfg.pop("subject_id_col", DataSchema.subject_id_name)
    for input_prefix, event_cfgs in event_conversion_cfg.items():
        input_subject_id_column = event_cfgs.get("subject_id_col", default_subject_id_col)

        prefix_to_columns.setdefault(input_prefix, set()).add(input_subject_id_column)

        join_cfg = event_cfgs.get("join")
        if join_cfg is not None:
            join_prefix = join_cfg["input_prefix"]
            prefix_to_columns[input_prefix].add(join_cfg["left_on"])
            prefix_to_columns.setdefault(join_prefix, set()).add(join_cfg["right_on"])
            for col in join_cfg.get("columns_from_right", []):
                prefix_to_columns.setdefault(join_prefix, set()).add(col)
                if col in prefix_to_columns[input_prefix]:
                    logger.debug(f"Removing {col} from {input_prefix} extraction plan as it is joined in")
                    prefix_to_columns[input_prefix].remove(col)

        for event_name, event_cfg in event_cfgs.items():
            if event_name in {"subject_id_col", "join"}:
                continue
            # If the config has a 'code' key and it contains column fields, parse and add them.
            for key, value in event_cfg.items():
                if key in META_KEYS:
                    continue

                if value is None:
                    # None can be used to indicate a null time, which has no associated column.
                    continue

                if isinstance(value, str):
                    value = [value]

                for field in value:
                    if is_col_field(field):
                        prefix_to_columns[input_prefix].add(parse_col_field(field))
                    elif key == "code":
                        # strings in the "code" fields are literals, not columns
                        continue
                    else:
                        prefix_to_columns[input_prefix].add(field)

    # Return things in sorted order for determinism.
    return {k: sorted(v) for k, v in prefix_to_columns.items()}


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

    event_conversion_cfg_fp = Path(cfg.event_conversion_config_fp)
    if not event_conversion_cfg_fp.exists():
        raise FileNotFoundError(f"Event conversion config file not found: {event_conversion_cfg_fp}")
    logger.info(f"Reading event conversion config from {event_conversion_cfg_fp} to identify needed columns.")
    event_conversion_cfg = OmegaConf.load(event_conversion_cfg_fp)

    prefix_to_columns = retrieve_columns(event_conversion_cfg)

    seen_files = set()
    input_files_to_subshard = []
    for fmt in ["parquet", "par", "csv", "csv.gz"]:
        files_in_fmt = set(raw_cohort_dir.rglob(f"*.{fmt}"))
        for f in files_in_fmt:
            if get_shard_prefix(raw_cohort_dir, f) in seen_files:
                logger.warning(f"Skipping {f} as it has already been added in a preferred format.")
                continue
            elif get_shard_prefix(raw_cohort_dir, f) not in prefix_to_columns:
                logger.warning(f"Skipping {f} as it is not specified in the event conversion configuration.")
                continue
            else:
                input_files_to_subshard.append(f)
                seen_files.add(get_shard_prefix(raw_cohort_dir, f))

    if not input_files_to_subshard:
        raise FileNotFoundError(f"Can't find any files in {raw_cohort_dir.resolve()!s} to sub-shard!")

    random.shuffle(input_files_to_subshard)

    subsharding_files_strs = "\n".join([f"  * {fp.resolve()!s}" for fp in input_files_to_subshard])
    logger.info(
        f"Starting event sub-sharding. Sub-sharding {len(input_files_to_subshard)} files:\n"
        f"{subsharding_files_strs}"
    )
    logger.info(
        f"Will read raw data from {raw_cohort_dir.resolve()!s}/$IN_FILE.parquet and write sub-sharded "
        f"data to {cfg.stage_cfg.output_dir}/$IN_FILE/$ROW_START-$ROW_END.parquet"
    )

    cloud_io_storage_options = cfg.get("cloud_io_storage_options", {})

    start = datetime.now(tz=UTC)
    for input_file in input_files_to_subshard:
        columns = prefix_to_columns[get_shard_prefix(raw_cohort_dir, input_file)]

        out_dir = UPath(cfg.stage_cfg.output_dir) / get_shard_prefix(raw_cohort_dir, input_file)
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
                f"the event conversion configuration in {event_conversion_cfg_fp.resolve()!s}."
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
