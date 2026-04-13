"""Utilities for converting input data structures into MEDS events.

All event config values are compiled through dftly: columns use ``$`` prefix (e.g., ``$col``),
string interpolation uses f-strings (e.g., ``f"CODE//{$col}"``), type casts use ``::``
(e.g., ``$ts::"%Y-%m-%d"``), and bare quoted strings are literals (e.g., ``"ADMISSION"``).
"""

import json
import logging
import random
from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path

import polars as pl
from dftly import Parser
from MEDS_transforms.dataframe import write_df
from MEDS_transforms.mapreduce.rwlock import rwlock_wrap
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig, OmegaConf
from upath import UPath

from ..config import EventConfig, MessyConfig, TableConfig

logger = logging.getLogger(__name__)

pl.enable_string_cache()


def extract_event(
    df: pl.LazyFrame,
    event_cfg: EventConfig,
    do_dedup_text_and_numeric: bool = False,
    source_block: str | None = None,
) -> pl.LazyFrame:
    """Extracts a single event dataframe from the raw data using dftly expressions.

    Every string value in the event config is compiled through dftly. Columns use ``$`` prefix
    (e.g., ``$col``), string interpolation uses f-strings (e.g., ``f"CODE//{$col}"``), and casts
    use ``::`` (e.g., ``$ts::"%Y-%m-%d"``). Bare identifiers without ``$`` are literals.

    Args:
        df: The raw data DataFrame with a ``"subject_id"`` column.
        event_cfg: The :class:`EventConfig` describing the event. ``time`` may be ``None`` for
            static events. ``extras`` values are dftly expressions compiled as additional output
            columns.
        do_dedup_text_and_numeric: If true, nullify ``text_value`` when it equals ``numeric_value``.
        source_block: If provided, added as a ``source_block`` column tracking the MESSY config
            origin of each event (e.g., ``"patients/eye_color"``). When ``None``, defaults to
            ``event_cfg.source_block``.

    Returns:
        A deduplicated DataFrame with ``subject_id``, ``code``, ``time``, and any additional columns.
        If the code expression references source columns, a ``code_components`` struct column is
        included with the individual column values that compose the code.

    Examples:
        >>> _ = pl.Config.set_tbl_width_chars(600)
        >>> raw = pl.DataFrame({
        ...     "subject_id": [1, 2, 3],
        ...     "test_name": ["Lab", "Vital", "Lab"],
        ...     "units": ["mg", "mmHg", "mg"],
        ...     "ts": ["2021-01-01", "2021-01-02", "2021-01-03"],
        ...     "result": [1.5, 2.7, 3.0],
        ... })
        >>> cfg = EventConfig(
        ...     name="lab",
        ...     table_prefix="labs",
        ...     code='f"{$test_name}//{$units}"',
        ...     time='$ts::"%Y-%m-%d"',
        ...     extras={"numeric_value": "$result"},
        ... )
        >>> extract_event(raw, cfg, source_block=None).drop("source_block")
        shape: (3, 5)
        ┌────────────┬─────────────┬──────────────────┬────────────┬───────────────┐
        │ subject_id ┆ code        ┆ code_components  ┆ time       ┆ numeric_value │
        │ ---        ┆ ---         ┆ ---              ┆ ---        ┆ ---           │
        │ i64        ┆ str         ┆ struct[2]        ┆ date       ┆ f64           │
        ╞════════════╪═════════════╪══════════════════╪════════════╪═══════════════╡
        │ 1          ┆ Lab//mg     ┆ {"Lab","mg"}     ┆ 2021-01-01 ┆ 1.5           │
        │ 2          ┆ Vital//mmHg ┆ {"Vital","mmHg"} ┆ 2021-01-02 ┆ 2.7           │
        │ 3          ┆ Lab//mg     ┆ {"Lab","mg"}     ┆ 2021-01-03 ┆ 3.0           │
        └────────────┴─────────────┴──────────────────┴────────────┴───────────────┘

        Static events (no time column):

        >>> static_cfg = EventConfig(
        ...     name="color", table_prefix="patients", code="EYE_COLOR", time=None,
        ... )
        >>> extract_event(raw, static_cfg).drop("source_block")
        shape: (3, 3)
        ┌────────────┬───────────┬──────────────┐
        │ subject_id ┆ code      ┆ time         │
        │ ---        ┆ ---       ┆ ---          │
        │ i64        ┆ str       ┆ datetime[μs] │
        ╞════════════╪═══════════╪══════════════╡
        │ 1          ┆ EYE_COLOR ┆ null         │
        │ 2          ┆ EYE_COLOR ┆ null         │
        │ 3          ┆ EYE_COLOR ┆ null         │
        └────────────┴───────────┴──────────────┘
    """
    event_exprs: dict[str, pl.Expr] = {"subject_id": pl.col("subject_id")}

    code_node = Parser()(event_cfg.code)
    event_exprs["code"] = code_node.polars_expr
    code_cols = code_node.referenced_columns

    if code_cols:
        event_exprs["code_components"] = pl.struct(**{col: pl.col(col) for col in sorted(code_cols)})

    code_null_filter = None
    if code_cols:
        first_col = sorted(code_cols)[0]
        code_null_filter = pl.col(first_col).is_not_null()

    ts_null_filter = None
    if event_cfg.time is None:
        event_exprs["time"] = pl.lit(None, dtype=pl.Datetime)
    else:
        ts_node = Parser()(event_cfg.time)
        event_exprs["time"] = ts_node.polars_expr
        # Filter on source columns being non-null/non-empty rather than on the parsed expression,
        # to avoid a polars predicate-pushdown bug where strptime(strict=True) is evaluated during
        # parquet scanning before nulls are filtered.
        ts_source_cols = ts_node.referenced_columns
        if ts_source_cols:
            schema = df.collect_schema()
            ts_filters = []
            for c in ts_source_cols:
                col_filter = pl.col(c).is_not_null()
                if schema.get(c) == pl.String or schema.get(c) is None:
                    col_filter = col_filter & (pl.col(c) != pl.lit(""))
                ts_filters.append(col_filter)
            ts_null_filter = pl.all_horizontal(*ts_filters)
        else:
            ts_null_filter = event_exprs["time"].is_not_null()

    for k, v in event_cfg.extras.items():
        event_exprs[k] = Parser.expr_to_polars(v)

    if do_dedup_text_and_numeric and "numeric_value" in event_exprs and "text_value" in event_exprs:
        text_expr = event_exprs["text_value"]
        num_expr = event_exprs["numeric_value"]
        event_exprs["text_value"] = (
            pl.when(text_expr.cast(pl.Float32, strict=False) == num_expr.cast(pl.Float32))
            .then(pl.lit(None, pl.String))
            .otherwise(text_expr)
        )

    resolved_source_block = source_block if source_block is not None else event_cfg.source_block
    if resolved_source_block is not None:
        event_exprs["source_block"] = pl.lit(resolved_source_block)

    if code_null_filter is not None:
        df = df.filter(code_null_filter)
    if ts_null_filter is not None:
        df = df.filter(ts_null_filter)

    return df.select(**event_exprs).unique(maintain_order=True)


def convert_to_events(
    df: pl.LazyFrame,
    event_cfgs: Iterable[EventConfig],
    do_dedup_text_and_numeric: bool = False,
) -> pl.LazyFrame:
    """Converts a DataFrame of raw data into a DataFrame of events.

    Args:
        df: The raw data DataFrame with a ``"subject_id"`` column.
        event_cfgs: Iterable of :class:`EventConfig` entries to extract.
        do_dedup_text_and_numeric: If true, nullify ``text_value`` when it equals ``numeric_value``.

    Returns:
        A concatenated DataFrame of all extracted events.

    Raises:
        ValueError: If no event configs provided or if extraction fails.

    Examples:
        >>> _ = pl.Config.set_tbl_width_chars(600)
        >>> raw = pl.DataFrame({
        ...     "subject_id": [1, 2],
        ...     "dept": ["CARDIAC", "PULM"],
        ...     "ts": ["2021-01-01", "2021-01-02"],
        ...     "color": ["blue", "green"],
        ... })
        >>> cfgs = [
        ...     EventConfig(name="admit", table_prefix="data", code="ADMISSION",
        ...                 time='$ts::"%Y-%m-%d"'),
        ...     EventConfig(name="color", table_prefix="data", code="EYE_COLOR",
        ...                 time=None, extras={"eye_color": "$color"}),
        ... ]
        >>> convert_to_events(raw, cfgs)
        shape: (4, 5)
        ┌────────────┬───────────┬─────────────────────┬──────────────┬───────────┐
        │ subject_id ┆ code      ┆ time                ┆ source_block ┆ eye_color │
        │ ---        ┆ ---       ┆ ---                 ┆ ---          ┆ ---       │
        │ i64        ┆ str       ┆ datetime[μs]        ┆ str          ┆ str       │
        ╞════════════╪═══════════╪═════════════════════╪══════════════╪═══════════╡
        │ 1          ┆ ADMISSION ┆ 2021-01-01 00:00:00 ┆ data/admit   ┆ null      │
        │ 2          ┆ ADMISSION ┆ 2021-01-02 00:00:00 ┆ data/admit   ┆ null      │
        │ 1          ┆ EYE_COLOR ┆ null                ┆ data/color   ┆ blue      │
        │ 2          ┆ EYE_COLOR ┆ null                ┆ data/color   ┆ green     │
        └────────────┴───────────┴─────────────────────┴──────────────┴───────────┘
        >>> convert_to_events(raw, [])
        Traceback (most recent call last):
            ...
        ValueError: No event configurations provided.
    """
    event_cfgs = list(event_cfgs)
    if not event_cfgs:
        raise ValueError("No event configurations provided.")

    event_dfs = []
    for event_cfg in event_cfgs:
        try:
            logger.info(f"Building computational graph for extracting {event_cfg.source_block}")
            event_dfs.append(
                extract_event(df, event_cfg, do_dedup_text_and_numeric=do_dedup_text_and_numeric)
            )
        except Exception as e:
            raise ValueError(f"Error extracting event {event_cfg.source_block}: {e}") from e

    return pl.concat(event_dfs, how="diagonal_relaxed")


@Stage.register(is_metadata=False)
def main(cfg: DictConfig):
    """Converts the event-sharded raw data into MEDS events and storing them in subject subsharded flat files.

    All arguments are specified through the command line into the ``cfg`` object through Hydra.
    """

    input_dir = UPath(cfg.stage_cfg.data_input_dir)
    out_dir = UPath(cfg.stage_cfg.output_dir)

    shards = json.loads(Path(cfg.shards_map_fp).read_text())

    event_conversion_cfg_fp = Path(cfg.event_conversion_config_fp)
    if not event_conversion_cfg_fp.exists():
        raise FileNotFoundError(f"Event conversion config file not found: {event_conversion_cfg_fp}")

    logger.info("Starting event conversion.")

    logger.info(f"Reading event conversion config from {event_conversion_cfg_fp}")
    event_conversion_cfg = OmegaConf.load(event_conversion_cfg_fp)
    logger.info(f"Event conversion config:\n{OmegaConf.to_yaml(event_conversion_cfg)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(event_conversion_cfg, out_dir / "event_conversion_config.yaml")

    messy_cfg = MessyConfig.parse(event_conversion_cfg)

    subject_splits = list(shards.items())
    random.shuffle(subject_splits)

    tables = list(messy_cfg.iter_tables())
    random.shuffle(tables)

    raw_opts = cfg.get("cloud_io_storage_options", {})
    cloud_io_storage_options = OmegaConf.to_container(raw_opts) if OmegaConf.is_config(raw_opts) else raw_opts

    read_fn = partial(pl.scan_parquet, glob=False, storage_options=cloud_io_storage_options)

    for sp, _ in subject_splits:
        for table in tables:
            input_fp = input_dir / sp / f"{table.input_prefix}.parquet"

            if not input_fp.is_file():
                input_fp_glob = f"{table.input_prefix}*.parquet"
                matching_files = list((input_dir / sp).glob(input_fp_glob))
                if len(matching_files) == 1:
                    fp = matching_files[0]
                    matching_prefixes = {t.input_prefix for t in tables if fp.stem.startswith(t.input_prefix)}
                    if len(matching_prefixes) != 1:  # pragma: no cover
                        logger.warning(
                            f"Found multiple matching prefixes for {input_fp}: {', '.join(matching_prefixes)}"
                        )
                    else:
                        logger.info(f"Found matching file {matching_files[0]} for {input_fp}")
                        input_fp = matching_files[0]

            out_fp = out_dir / sp / f"{table.input_prefix}.parquet"

            def compute_fntr(tbl: TableConfig, sp: str) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
                def compute_fn(df: pl.LazyFrame) -> pl.LazyFrame:
                    if tbl.subject_id_polars_expr is not None:
                        df = df.with_columns(subject_id=tbl.subject_id_polars_expr)

                    if tbl.cols:
                        col_exprs = Parser.to_polars(dict(tbl.cols))
                        df = df.with_columns(**col_exprs)

                    try:
                        logger.info(f"Extracting events for {tbl.input_prefix}")
                        return convert_to_events(
                            df,
                            event_cfgs=tbl.events,
                            do_dedup_text_and_numeric=cfg.stage_cfg.get("do_dedup_text_and_numeric", False),
                        )
                    except Exception as e:  # pragma: no cover
                        raise ValueError(f"Error converting to MEDS for {sp}/{tbl.input_prefix}: {e}") from e

                return compute_fn

            rwlock_wrap(
                input_fp,
                out_fp,
                read_fn,
                write_df,
                compute_fntr(table, sp),
                do_overwrite=cfg.do_overwrite,
            )

    logger.info("Subsharded into converted events.")
