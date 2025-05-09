import json
import logging
from functools import partial
from pathlib import Path

import polars as pl
from MEDS_transforms.compute_modes.compute_fn import identity_fn
from MEDS_transforms.mapreduce import map_stage
from MEDS_transforms.mapreduce.shard_iteration import shuffle_shards
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def shard_iterator_by_shard_map(cfg: DictConfig) -> tuple[list[str], bool]:
    """Returns an iterator over shard paths and output paths based on a shard map file, not files on disk.

    Args:
        cfg: The configuration dictionary for the overall pipeline. Should contain the following keys:
            - `shards_map_fp` (mandatory): The file path to the shards map file.
            - `stage_cfg.data_input_dir` (mandatory): The directory containing the input data.
            - `stage_cfg.output_dir` (mandatory): The directory to write the output data.
            - `worker` (optional): The worker ID for the MR worker; this is also used to seed the

    Returns:
        A list of pairs of input and output file paths for each shard, as well as a boolean indicating
        whether the shards are only train shards.

    Raises:
        ValueError: If the `shards_map_fp` key is not present in the configuration.
        FileNotFoundError: If the shard map file is not found at the path specified in the configuration.
        ValueError: If the `train_only` key is present in the configuration.

    Examples:
        >>> shard_iterator_by_shard_map(DictConfig({}))
        Traceback (most recent call last):
            ...
        ValueError: shards_map_fp must be present in the configuration for a map-based shard iterator.
        >>> with tempfile.NamedTemporaryFile() as tmp:
        ...     cfg = DictConfig({"shards_map_fp": tmp.name, "stage_cfg": {"train_only": True}})
        ...     shard_iterator_by_shard_map(cfg)
        Traceback (most recent call last):
            ...
        ValueError: train_only is not supported for this stage.
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     tmp = Path(tmp)
        ...     shards_map_fp = tmp / "shards_map.json"
        ...     cfg = DictConfig({"shards_map_fp": shards_map_fp, "stage_cfg": {"train_only": False}})
        ...     shard_iterator_by_shard_map(cfg)
        Traceback (most recent call last):
            ...
        FileNotFoundError: Shard map file not found at ...shards_map.json
        >>> shards = {"train/0": [1, 2, 3, 4], "train/1": [5, 6, 7], "tuning": [8], "held_out": [9]}
        >>> with tempfile.NamedTemporaryFile() as tmp:
        ...     _ = Path(tmp.name).write_text(json.dumps(shards))
        ...     cfg = DictConfig({
        ...         "shards_map_fp": tmp.name,
        ...         "worker": 1,
        ...         "stage_cfg": {"data_input_dir": "data", "output_dir": "output"},
        ...     })
        ...     fps, includes_only_train = shard_iterator_by_shard_map(cfg)
        >>> fps
        [(PosixPath('data/train/1'),  PosixPath('output/train/1.parquet')),
         (PosixPath('data/held_out'), PosixPath('output/held_out.parquet')),
         (PosixPath('data/tuning'),   PosixPath('output/tuning.parquet')),
         (PosixPath('data/train/0'),  PosixPath('output/train/0.parquet'))]
        >>> includes_only_train
        False
    """

    if "shards_map_fp" not in cfg:
        raise ValueError("shards_map_fp must be present in the configuration for a map-based shard iterator.")

    if cfg.stage_cfg.get("train_only", None):
        raise ValueError("train_only is not supported for this stage.")

    shard_map_fp = Path(cfg.shards_map_fp)
    if not shard_map_fp.exists():
        raise FileNotFoundError(f"Shard map file not found at {shard_map_fp.resolve()!s}")

    shards = list(json.loads(shard_map_fp.read_text()).keys())

    input_dir = Path(cfg.stage_cfg.data_input_dir)
    output_dir = Path(cfg.stage_cfg.output_dir)

    shards = shuffle_shards(shards, cfg)

    logger.info(f"Mapping computation over a maximum of {len(shards)} shards")

    out = []
    for sh in shards:
        in_fp = input_dir / sh
        out_fp = output_dir / f"{sh}.parquet"
        out.append((in_fp, out_fp))

    return out, False


def merge_subdirs_and_sort(
    sp_dir: Path,
    event_subsets: list[str],
    unique_by: list[str] | str | None,
    additional_sort_by: list[str] | None = None,
) -> pl.LazyFrame:
    """This function reads all parquet files in subdirs of `sp_dir` and merges them into a single dataframe.

    Args:
        sp_dir: The directory containing the subdirs with parquet files to be merged.
        event_subsets: The list of event table paths passed to maintain the order in event_configs.yaml
            while merging the events.
        unique_by: The list of columns that should be ensured to be unique after the dataframes are merged. If
            `None`, this is ignored. If `*`, all columns are used. If a list of strings, only the columns in
            the list are used. If a column is not found in the dataframe, it is omitted from the unique-by, a
            warning is logged, but an error is *not* raised. Which rows are retained if the uniqeu-by columns
            are not all columns is not guaranteed, but is also *not* random, so this may have statistical
            implications.
        additional_sort_by: Additional columns to sort by, in addition to the default sorting by subject ID
            and time. If `None`, only subject ID and time are used. If a list of strings, these
            columns are used in addition to the default sorting. If a column is not found in the dataframe, it
            is omitted from the sort-by, a warning is logged, but an error is *not* raised. This functionality
            is useful both for deterministic testing and in cases where a data owner wants to impose
            intra-event measurement ordering in the data, though this is not recommended in general.

    Returns:
        A single dataframe containing all the data from the parquet files in the subdirs of `sp_dir`. These
        files will be concatenated diagonally, taking the union of all rows in all dataframes and all unique
        columns in all dataframes to form the merged output. The returned dataframe will be made unique by the
        columns specified in `unique_by` and sorted by first subject ID, then time, then all columns in
        `additional_sort_by`, if any.

    Raises:
        FileNotFoundError: If no parquet files are found in the subdirs of `sp_dir`.
        ValueError: If `unique_by` is not `None`, `*`, or a list of strings

    Examples:
        >>> from tempfile import TemporaryDirectory
        >>> df1 = pl.DataFrame({"subject_id": [1, 2], "time": [10, 20], "code": ["A", "B"]})
        >>> df2 = pl.DataFrame({
        ...     "subject_id":      [1,   1,    3],
        ...     "time":       [2,   1,    8],
        ...     "code":            ["C", "D",  "E"],
        ...     "numeric_value": [None, 2.0, None],
        ... })
        >>> df3 = pl.DataFrame({
        ...     "subject_id":      [1,   1,    3],
        ...     "time":       [2,   2,    8],
        ...     "code":            ["C", "D",  "E"],
        ...     "numeric_value": [6.2, 2.0, None],
        ... })
        >>> with TemporaryDirectory() as tmpdir:
        ...     sp_dir = Path(tmpdir)
        ...     merge_subdirs_and_sort(sp_dir, event_subsets=[], unique_by=None)
        Traceback (most recent call last):
            ...
        FileNotFoundError: No parquet files found in ...
        >>> with TemporaryDirectory() as tmpdir:
        ...     sp_dir = Path(tmpdir)
        ...     df1.write_parquet(sp_dir / "file1.parquet")
        ...     df2.write_parquet(sp_dir / "file2.parquet")
        ...     df3.write_parquet(sp_dir / "df.parquet")
        ...     merge_subdirs_and_sort(
        ...         sp_dir,
        ...         event_subsets=["file1", "file2", "df"],
        ...         unique_by=None,
        ...         additional_sort_by=["code", "numeric_value", "missing_col_will_not_error"]
        ...     ).collect()
        shape: (8, 4)
        ┌────────────┬──────┬──────┬───────────────┐
        │ subject_id ┆ time ┆ code ┆ numeric_value │
        │ ---        ┆ ---  ┆ ---  ┆ ---           │
        │ i64        ┆ i64  ┆ str  ┆ f64           │
        ╞════════════╪══════╪══════╪═══════════════╡
        │ 1          ┆ 1    ┆ D    ┆ 2.0           │
        │ 1          ┆ 2    ┆ C    ┆ null          │
        │ 1          ┆ 2    ┆ C    ┆ 6.2           │
        │ 1          ┆ 2    ┆ D    ┆ 2.0           │
        │ 1          ┆ 10   ┆ A    ┆ null          │
        │ 2          ┆ 20   ┆ B    ┆ null          │
        │ 3          ┆ 8    ┆ E    ┆ null          │
        │ 3          ┆ 8    ┆ E    ┆ null          │
        └────────────┴──────┴──────┴───────────────┘
        >>> with TemporaryDirectory() as tmpdir:
        ...     sp_dir = Path(tmpdir)
        ...     df1.write_parquet(sp_dir / "file1.parquet")
        ...     df2.write_parquet(sp_dir / "file2.parquet")
        ...     df3.write_parquet(sp_dir / "df.parquet")
        ...     merge_subdirs_and_sort(
        ...         sp_dir,
        ...         event_subsets=["file1", "file2", "df"],
        ...         unique_by="*",
        ...         additional_sort_by=["code", "numeric_value"]
        ...     ).collect()
        shape: (7, 4)
        ┌────────────┬──────┬──────┬───────────────┐
        │ subject_id ┆ time ┆ code ┆ numeric_value │
        │ ---        ┆ ---  ┆ ---  ┆ ---           │
        │ i64        ┆ i64  ┆ str  ┆ f64           │
        ╞════════════╪══════╪══════╪═══════════════╡
        │ 1          ┆ 1    ┆ D    ┆ 2.0           │
        │ 1          ┆ 2    ┆ C    ┆ null          │
        │ 1          ┆ 2    ┆ C    ┆ 6.2           │
        │ 1          ┆ 2    ┆ D    ┆ 2.0           │
        │ 1          ┆ 10   ┆ A    ┆ null          │
        │ 2          ┆ 20   ┆ B    ┆ null          │
        │ 3          ┆ 8    ┆ E    ┆ null          │
        └────────────┴──────┴──────┴───────────────┘
        >>> with TemporaryDirectory() as tmpdir:
        ...     sp_dir = Path(tmpdir)
        ...     df1.write_parquet(sp_dir / "file1.parquet")
        ...     df2.write_parquet(sp_dir / "file2.parquet")
        ...     df3.write_parquet(sp_dir / "df.parquet")
        ...     # We just display the subject ID, time, and code columns as the numeric value column
        ...     # is not guaranteed to be deterministic in the output given some rows will be dropped due to
        ...     # the unique-by constraint.
        ...     merge_subdirs_and_sort(
        ...         sp_dir,
        ...         event_subsets=["file1", "file2", "df"],
        ...         unique_by=["subject_id", "time", "code", "missing_col_will_not_error"],
        ...         additional_sort_by=["code", "numeric_value"]
        ...     ).select("subject_id", "time", "code").collect()
        shape: (6, 3)
        ┌────────────┬──────┬──────┐
        │ subject_id ┆ time ┆ code │
        │ ---        ┆ ---  ┆ ---  │
        │ i64        ┆ i64  ┆ str  │
        ╞════════════╪══════╪══════╡
        │ 1          ┆ 1    ┆ D    │
        │ 1          ┆ 2    ┆ C    │
        │ 1          ┆ 2    ┆ D    │
        │ 1          ┆ 10   ┆ A    │
        │ 2          ┆ 20   ┆ B    │
        │ 3          ┆ 8    ┆ E    │
        └────────────┴──────┴──────┘
        >>> with TemporaryDirectory() as tmpdir:
        ...     sp_dir = Path(tmpdir)
        ...     df1.write_parquet(sp_dir / "file1.parquet")
        ...     df2.write_parquet(sp_dir / "file2.parquet")
        ...     df3.write_parquet(sp_dir / "df.parquet")
        ...     # We just display the subject ID, time, and code columns as the numeric value column
        ...     # is not guaranteed to be deterministic in the output given some rows will be dropped due to
        ...     # the unique-by constraint.
        ...     merge_subdirs_and_sort(
        ...         sp_dir,
        ...         event_subsets=["file1", "file2", "df"],
        ...         unique_by=352.2, # This will error
        ...     )
        Traceback (most recent call last):
            ...
        ValueError: Invalid unique_by value: 352.2
    """
    files_to_read = [(sp_dir / f"{es}.parquet") for es in event_subsets]
    if not files_to_read:
        raise FileNotFoundError(f"No parquet files found in {sp_dir}/*.parquet.")

    file_strs = "\n".join(f"  - {fp.resolve()!s}" for fp in files_to_read)
    logger.info(f"Reading {len(files_to_read)} files:\n{file_strs}")

    dfs = [pl.scan_parquet(fp, glob=False) for fp in files_to_read]
    df = pl.concat(dfs, how="diagonal_relaxed")

    df_columns = set(df.collect_schema().names())

    match unique_by:
        case None:
            pass
        case "*":
            df = df.unique(maintain_order=True)
        case list() if len(unique_by) > 0 and all(isinstance(u, str) for u in unique_by):
            subset = []
            for u in unique_by:
                if u in df_columns:
                    subset.append(u)
                else:
                    logger.warning(f"Column {u} not found in dataframe. Omitting from unique-by subset.")
            df = df.unique(maintain_order=True, subset=subset)
        case _:
            raise ValueError(f"Invalid unique_by value: {unique_by}")

    sort_by = ["subject_id", "time"]
    if additional_sort_by is not None:
        for s in additional_sort_by:
            if s in df_columns:
                sort_by.append(s)
            else:
                logger.warning(f"Column {s} not found in dataframe. Omitting from sort-by list.")

    return df.sort(by=sort_by, maintain_order=True, multithreaded=False)


@Stage.register(is_metadata=False)
def main(cfg: DictConfig):
    """Merges the subject sub-sharded events into a single parquet file per subject shard.

    This function takes all dataframes (in parquet files) in any subdirs of the `cfg.stage_cfg.input_dir` and
    merges them into a single dataframe. All dataframes in the subdirs are assumed to be in the unnested, MEDS
    format, and cover the same group of subjects (specific to the shard being processed). The merged dataframe
    will also be sorted by subject ID and time.

    All arguments are specified through the command line into the `cfg` object through Hydra.

    The `cfg.stage_cfg` object is a special key that is imputed by OmegaConf to contain the stage-specific
    configuration arguments based on the global, pipeline-level configuration file.

    Args:
        unique_by: The list of columns that should be ensured to be unique
            after the dataframes are merged. Defaults to `"*"`, which means all columns are used.
        additional_sort_by: Additional columns to sort by, in addition to
            the default sorting by subject ID and time. Defaults to `None`, which means only subject ID
            and time are used.

    Returns:
        Writes the merged dataframes to the shard-specific output filepath in the `cfg.stage_cfg.output_dir`.
    """
    event_conversion_cfg = OmegaConf.load(cfg.event_conversion_config_fp)
    event_conversion_cfg.pop("subject_id_col", None)

    read_fn = partial(
        merge_subdirs_and_sort,
        event_subsets=list(event_conversion_cfg.keys()),
        unique_by=cfg.stage_cfg.get("unique_by", None),
        additional_sort_by=cfg.stage_cfg.get("additional_sort_by", None),
    )

    map_stage(
        cfg,
        map_fn=identity_fn,
        read_fn=read_fn,
        shard_iterator_fntr=shard_iterator_by_shard_map,
    )
