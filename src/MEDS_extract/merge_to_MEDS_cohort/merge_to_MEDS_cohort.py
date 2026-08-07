"""Merge the per-table MEDS event files for each subject shard into one sorted MEDS parquet per shard.

Note that this stage *drops* the internal ``code_components`` struct column: each source table
carries its own struct fields (one per raw column its codes reference), so concatenating ~30 tables with
``diagonal_relaxed`` unifies them into a single superstruct with the union of all fields — ~70 on
eICU-shaped data — and every dense materialization in the merge (the sort's gather, pyarrow export,
rechunking) then carries all ~70 child buffers for every row. On a 60 MB eICU-shaped synthetic that
superstruct densification peaked at 12.4 GB; dropping the column before concatenation brings the same
merge to 2.2 GB. The drop is safe because ``code_components`` is internal metadata-linkage state:
``extract_code_metadata`` consumes the *pre-merge* per-table event files (``convert_to_MEDS_events``
output), so the merged copy fed only the final data files, where nothing needs it — for now.
"""

import json
import logging
from functools import partial
from pathlib import Path

import polars as pl
from MEDS_transforms.compute_modes.compute_fn import identity_fn
from MEDS_transforms.mapreduce import map_stage
from MEDS_transforms.mapreduce.shard_iteration import shuffle_shards
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig

from .._stage_example import MEDSExtractStageExample
from ..config import PROVENANCE_COL, MessyConfig

logger = logging.getLogger(__name__)


def _unique_merging_provenance(df: pl.LazyFrame, unique_by: list[str], all_cols: list[str]) -> pl.LazyFrame:
    """Deduplicate ``df`` on ``unique_by`` while merging the ``provenance`` column as a set.

    Mirrors ``df.unique(subset=unique_by, maintain_order=True)`` exactly — same rows (the
    first row per key), same columns in the same order — except that the collapsed rows'
    provenance lists are unioned instead of one being kept arbitrarily. ``provenance`` is
    never part of the dedup key: two rows identical except for provenance must still
    collapse, or enabling provenance tracking would change row counts.

    Args:
        df: The input frame; must contain a ``provenance`` column.
        unique_by: The dedup key columns. A ``provenance`` entry is ignored.
        all_cols: Every column of ``df`` in output order.

    Examples:
        >>> _ = pl.Config.set_tbl_width_chars(600)
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 2],
        ...     "code": ["A", "A", "B"],
        ...     "numeric_value": [1.0, 2.0, None],
        ...     "provenance": [
        ...         [{"source_file": "x.csv", "row_idx": 0}],
        ...         [{"source_file": "y.csv", "row_idx": 3}],
        ...         [{"source_file": "x.csv", "row_idx": 1}],
        ...     ],
        ... })
        >>> cols = df.columns

        Keyed on all non-provenance columns (the ``unique_by: "*"`` case), the two
        ``(1, A)`` rows differ on ``numeric_value`` and stay distinct:

        >>> _unique_merging_provenance(df.lazy(), ["subject_id", "code", "numeric_value"], cols).collect()
        shape: (3, 4)
        ┌────────────┬──────┬───────────────┬─────────────────┐
        │ subject_id ┆ code ┆ numeric_value ┆ provenance      │
        │ ---        ┆ ---  ┆ ---           ┆ ---             │
        │ i64        ┆ str  ┆ f64           ┆ list[struct[2]] │
        ╞════════════╪══════╪═══════════════╪═════════════════╡
        │ 1          ┆ A    ┆ 1.0           ┆ [{"x.csv",0}]   │
        │ 1          ┆ A    ┆ 2.0           ┆ [{"y.csv",3}]   │
        │ 2          ┆ B    ┆ null          ┆ [{"x.csv",1}]   │
        └────────────┴──────┴───────────────┴─────────────────┘

        Keyed on a subset, the ``(1, A)`` rows collapse to the first row's values with
        the union of both rows' provenance:

        >>> _unique_merging_provenance(df.lazy(), ["subject_id", "code"], cols).collect()
        shape: (2, 4)
        ┌────────────┬──────┬───────────────┬────────────────────────────┐
        │ subject_id ┆ code ┆ numeric_value ┆ provenance                 │
        │ ---        ┆ ---  ┆ ---           ┆ ---                        │
        │ i64        ┆ str  ┆ f64           ┆ list[struct[2]]            │
        ╞════════════╪══════╪═══════════════╪════════════════════════════╡
        │ 1          ┆ A    ┆ 1.0           ┆ [{"x.csv",0}, {"y.csv",3}] │
        │ 2          ┆ B    ┆ null          ┆ [{"x.csv",1}]              │
        └────────────┴──────┴───────────────┴────────────────────────────┘
    """
    key_cols = [c for c in unique_by if c != PROVENANCE_COL]
    agg_exprs = [pl.col(c).first() for c in all_cols if c not in key_cols and c != PROVENANCE_COL]
    agg_exprs.append(pl.col(PROVENANCE_COL).explode().unique(maintain_order=True))
    return df.group_by(key_cols, maintain_order=True).agg(agg_exprs).select(all_cols)


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
        >>> import tempfile
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
    table_prefixes: list[str],
    unique_by: list[str] | str | None,
    additional_sort_by: list[str] | None = None,
) -> pl.LazyFrame:
    """Reads `<sp_dir>/<prefix>.parquet` for each configured table prefix and merges them into one dataframe.

    Args:
        sp_dir: The directory containing the per-table parquet files to be merged.
        table_prefixes: The list of source-table prefixes whose per-shard parquet files should be
            merged. Each prefix corresponds to ``<sp_dir>/<prefix>.parquet``. The order is preserved
            from the MESSY config so that downstream merging is deterministic.
        unique_by: The list of columns that should be ensured to be unique after the dataframes are merged. If
            `None`, this is ignored. If `*`, all columns are used. If a list of strings, only the columns in
            the list are used. If a column is not found in the dataframe, it is omitted from the unique-by, a
            warning is logged, but an error is *not* raised. Which rows are retained if the unique-by columns
            are not all columns is not guaranteed, but is also *not* random, so this may have statistical
            implications. A ``provenance`` column (present when ``convert_to_MEDS_events`` ran with
            ``do_track_provenance``) is never part of the dedup key; rows collapsing under ``unique_by``
            merge their provenance lists as a set instead (see :func:`_unique_merging_provenance`).
        additional_sort_by: Additional columns to sort by, in addition to the default sorting by subject ID
            and time. If `None`, only subject ID and time are used. If a list of strings, these
            columns are used in addition to the default sorting. If a column is not found in the dataframe, it
            is omitted from the sort-by, a warning is logged, but an error is *not* raised. This functionality
            is useful both for deterministic testing and in cases where a data owner wants to impose
            intra-event measurement ordering in the data, though this is not recommended in general.

    Returns:
        A single dataframe containing all the data from the per-prefix parquet files under `sp_dir`. These
        files will be concatenated diagonally, taking the union of all rows in all dataframes and all unique
        columns in all dataframes to form the merged output. The returned dataframe will be made unique by the
        columns specified in `unique_by` and sorted by first subject ID, then time, then all columns in
        `additional_sort_by`, if any.

        The internal ``code_components`` column is *excluded* from the merged output. Each table's struct
        carries different fields, so the diagonal concat would unify them into a superstruct with the union
        of all fields across all tables, and every dense materialization downstream of the concat would then
        carry every table's child buffers for every row — a 12.4 GB peak on a 60 MB eICU-shaped input, vs.
        2.2 GB with the column dropped. Dropping it is safe: ``extract_code_metadata`` reads the
        pre-merge per-table events, so nothing downstream of the merge consumes the column. (An alternative
        that preserves it exists — explode the struct into name-prefixed flat columns before the concat and
        re-collapse after the sort — but was rejected for now as more code for a column nothing downstream
        needs; JSON-encoding the struct was also measured, at 3.1 GB.) Because the drop happens per input
        scan, projection pushdown means the column is never even read from disk.

    Raises:
        FileNotFoundError: If `table_prefixes` is empty, i.e., no tables are configured to be merged.
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
        ...     merge_subdirs_and_sort(sp_dir, table_prefixes=[], unique_by=None)
        Traceback (most recent call last):
            ...
        FileNotFoundError: No tables configured to merge under ...
        >>> with TemporaryDirectory() as tmpdir:
        ...     sp_dir = Path(tmpdir)
        ...     df1.write_parquet(sp_dir / "file1.parquet")
        ...     df2.write_parquet(sp_dir / "file2.parquet")
        ...     df3.write_parquet(sp_dir / "df.parquet")
        ...     merge_subdirs_and_sort(
        ...         sp_dir,
        ...         table_prefixes=["file1", "file2", "df"],
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

        The internal ``code_components`` and ``metadata_components`` structs are dropped at merge
        (#254) — including when only *some* tables carry them (a table whose codes are all literals
        legitimately has no components; only ``_self``-metadata events carry metadata components):

        >>> df4 = pl.DataFrame({
        ...     "subject_id": [1],
        ...     "time": [5],
        ...     "code": ["F//X"],
        ...     "code_components": [{"f": "X"}],
        ...     "metadata_components": [{"description": "an F"}],
        ... })
        >>> with TemporaryDirectory() as tmpdir:
        ...     sp_dir = Path(tmpdir)
        ...     df1.write_parquet(sp_dir / "file1.parquet")
        ...     df4.write_parquet(sp_dir / "with_components.parquet")
        ...     merge_subdirs_and_sort(
        ...         sp_dir, table_prefixes=["file1", "with_components"], unique_by=None
        ...     ).collect()
        shape: (3, 3)
        ┌────────────┬──────┬──────┐
        │ subject_id ┆ time ┆ code │
        │ ---        ┆ ---  ┆ ---  │
        │ i64        ┆ i64  ┆ str  │
        ╞════════════╪══════╪══════╡
        │ 1          ┆ 5    ┆ F//X │
        │ 1          ┆ 10   ┆ A    │
        │ 2          ┆ 20   ┆ B    │
        └────────────┴──────┴──────┘
        >>> with TemporaryDirectory() as tmpdir:
        ...     sp_dir = Path(tmpdir)
        ...     df1.write_parquet(sp_dir / "file1.parquet")
        ...     df2.write_parquet(sp_dir / "file2.parquet")
        ...     df3.write_parquet(sp_dir / "df.parquet")
        ...     merge_subdirs_and_sort(
        ...         sp_dir,
        ...         table_prefixes=["file1", "file2", "df"],
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
        ...         table_prefixes=["file1", "file2", "df"],
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
        ...         table_prefixes=["file1", "file2", "df"],
        ...         unique_by=352.2, # This will error
        ...     )
        Traceback (most recent call last):
            ...
        ValueError: Invalid unique_by value: 352.2
    """
    files_to_read = [(sp_dir / f"{tp}.parquet") for tp in table_prefixes]
    if not files_to_read:
        raise FileNotFoundError(f"No tables configured to merge under {sp_dir} (empty table_prefixes).")

    file_strs = "\n".join(f"  - {fp.resolve()!s}" for fp in files_to_read)
    logger.info(f"Reading {len(files_to_read)} files:\n{file_strs}")

    # Drop the internal ``code_components`` / ``metadata_components`` structs per scan, *before* the
    # concat, so the field-union superstruct never forms and projection pushdown never reads the
    # columns (#254; see the module and docstring notes above for the memory numbers).
    # ``strict=False`` because a table whose codes are all literals legitimately has no components
    # column, and only ``_self``-metadata events carry metadata components.
    dfs = [
        pl.scan_parquet(fp, glob=False).drop("code_components", "metadata_components", strict=False)
        for fp in files_to_read
    ]
    df = pl.concat(dfs, how="diagonal_relaxed")

    schema_cols = df.collect_schema().names()
    df_columns = set(schema_cols)
    # Provenance is detected by presence, not by a flag: when the convert stage attached it,
    # dedup here must merge it as a set rather than let it (a) split otherwise-identical
    # rows or (b) be dropped arbitrarily. Either way rows match a provenance-free run.
    has_provenance = PROVENANCE_COL in df_columns

    match unique_by:
        case None:
            pass
        case "*":
            if has_provenance:
                df = _unique_merging_provenance(df, schema_cols, schema_cols)
            else:
                df = df.unique(maintain_order=True)
        case list() if len(unique_by) > 0 and all(isinstance(u, str) for u in unique_by):
            subset = []
            for u in unique_by:
                if u == PROVENANCE_COL and has_provenance:
                    logger.warning(f"Column {u} is never a dedup key. Omitting from unique-by subset.")
                elif u in df_columns:
                    subset.append(u)
                else:
                    logger.warning(f"Column {u} not found in dataframe. Omitting from unique-by subset.")
            if has_provenance:
                df = _unique_merging_provenance(df, subset, schema_cols)
            else:
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

    return df.sort(by=sort_by, maintain_order=True, multithreaded=True)


@Stage.register(is_metadata=False, example_class=MEDSExtractStageExample)
def main(cfg: DictConfig):
    """Merges the subject sub-sharded events into a single parquet file per subject shard.

    This function reads, for each shard, the per-table file `<prefix>.parquet` under the shard's directory in
    `cfg.stage_cfg.input_dir` — one file per configured table prefix, in config order — and merges them into a
    single dataframe. All such dataframes are assumed to be in the unnested, MEDS format, and cover the same
    group of subjects (specific to the shard being processed). The merged dataframe will also be sorted by
    subject ID and time. The internal ``code_components`` struct column is dropped during the merge (#254;
    see :func:`merge_subdirs_and_sort`) — metadata extraction reads the pre-merge per-table events, so the
    merged data does not need it and unifying the per-table structs is catastrophically memory-expensive.

    All arguments are specified through the command line into the `cfg` object through Hydra.

    The `cfg.stage_cfg` object is a special key that is imputed by OmegaConf to contain the stage-specific
    configuration arguments based on the global, pipeline-level configuration file.

    Args:
        unique_by: The list of columns that should be ensured to be unique after the dataframes are
            merged. Defaults to `"*"` (all columns): with `code_components` dropped at merge,
            two source observations that differed only in their raw components collapse into identical
            rows, and identical-looking rows in the merged output must not be duplicated — full-row
            uniqueness is a semantic guarantee of the merged output. Set to `None` for raw concat
            semantics that keep such collapsed pairs, or to an explicit column list to dedup on a
            subset.
        additional_sort_by: Additional columns to sort by, in addition to
            the default sorting by subject ID and time. Defaults to `None`, which means only subject ID
            and time are used.

    Returns:
        Writes the merged dataframes to the shard-specific output filepath in the `cfg.stage_cfg.output_dir`.
    """
    table_prefixes = MessyConfig.load(cfg.MESSY_config_fp).table_prefixes

    read_fn = partial(
        merge_subdirs_and_sort,
        table_prefixes=table_prefixes,
        unique_by=cfg.stage_cfg.get("unique_by", None),
        additional_sort_by=cfg.stage_cfg.get("additional_sort_by", None),
    )

    map_stage(
        cfg,
        map_fn=identity_fn,
        read_fn=read_fn,
        shard_iterator_fntr=shard_iterator_by_shard_map,
    )
