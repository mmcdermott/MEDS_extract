"""Utilities for converting input data structures into MEDS events.

All event config values are compiled through dftly: columns use ``$`` prefix (e.g., ``$col``),
string interpolation uses f-strings (e.g., ``f"CODE//{$col}"``), type casts use ``::``
(e.g., ``$ts::"%Y-%m-%d"``), and bare quoted strings are literals (e.g., ``"ADMISSION"``).
"""

import copy
import json
import logging
import random
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import polars as pl
from dftly import Parser
from MEDS_transforms.dataframe import write_df
from MEDS_transforms.mapreduce.rwlock import rwlock_wrap
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig, OmegaConf
from upath import UPath

from ..dftly_bridge import EVENT_META_KEYS, compile_subject_id_expr

logger = logging.getLogger(__name__)

pl.enable_string_cache()


def extract_event(
    df: pl.LazyFrame,
    event_cfg: dict[str, str | None],
    do_dedup_text_and_numeric: bool = False,
) -> pl.LazyFrame:
    """Extracts a single event dataframe from the raw data using dftly expressions.

    Every string value in the event config is compiled through dftly. Columns use ``$`` prefix
    (e.g., ``$col``), string interpolation uses f-strings (e.g., ``f"CODE//{$col}"``), and casts
    use ``::`` (e.g., ``$ts::"%Y-%m-%d"``). Bare identifiers without ``$`` are literals.

    Args:
        df: The raw data DataFrame with a ``"subject_id"`` column.
        event_cfg: Event configuration dict. Must contain ``"code"`` and ``"time"`` keys.
            ``"time"`` may be ``None`` for static events. All other keys are treated as
            additional output columns whose values are dftly expressions.
        do_dedup_text_and_numeric: If true, nullify ``text_value`` when it equals ``numeric_value``.

    Returns:
        A deduplicated DataFrame with ``subject_id``, ``code``, ``time``, and any additional columns.

    Examples:
        >>> _ = pl.Config.set_tbl_width_chars(600)
        >>> raw = pl.DataFrame({
        ...     "subject_id": [1, 2, 3],
        ...     "test_name": ["Lab", "Vital", "Lab"],
        ...     "units": ["mg", "mmHg", "mg"],
        ...     "ts": ["2021-01-01", "2021-01-02", "2021-01-03"],
        ...     "result": [1.5, 2.7, 3.0],
        ... })
        >>> cfg = {
        ...     "code": 'f"{$test_name}//{$units}"',
        ...     "time": '$ts::"%Y-%m-%d"',
        ...     "numeric_value": "$result",
        ... }
        >>> extract_event(raw, cfg)
        shape: (3, 4)
        ┌────────────┬─────────────┬────────────┬───────────────┐
        │ subject_id ┆ code        ┆ time       ┆ numeric_value │
        │ ---        ┆ ---         ┆ ---        ┆ ---           │
        │ i64        ┆ str         ┆ date       ┆ f64           │
        ╞════════════╪═════════════╪════════════╪═══════════════╡
        │ 1          ┆ Lab//mg     ┆ 2021-01-01 ┆ 1.5           │
        │ 2          ┆ Vital//mmHg ┆ 2021-01-02 ┆ 2.7           │
        │ 3          ┆ Lab//mg     ┆ 2021-01-03 ┆ 3.0           │
        └────────────┴─────────────┴────────────┴───────────────┘
        >>> static_cfg = {"code": '"EYE_COLOR"', "time": None}
        >>> extract_event(raw, static_cfg)
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
        >>> extract_event(raw, {"time": '$ts::"%Y-%m-%d"'})
        Traceback (most recent call last):
            ...
        KeyError: "Event configuration dictionary must contain 'code' key. Got: [time]."
        >>> extract_event(raw, {"code": '"X"'})
        Traceback (most recent call last):
            ...
        KeyError: "Event configuration dictionary must contain 'time' key. Got: [code]."
    """
    event_cfg = copy.deepcopy(event_cfg)
    event_exprs = {"subject_id": pl.col("subject_id")}

    if "code" not in event_cfg:
        raise KeyError(
            f"Event configuration dictionary must contain 'code' key. Got: [{', '.join(event_cfg.keys())}]."
        )
    if "time" not in event_cfg:
        raise KeyError(
            f"Event configuration dictionary must contain 'time' key. Got: [{', '.join(event_cfg.keys())}]."
        )

    code_value = str(event_cfg.pop("code"))
    code_node = Parser()(code_value)
    event_exprs["code"] = code_node.polars_expr
    code_cols = code_node.referenced_columns

    # Build null filter: if code references columns, filter out rows where the first column is null
    code_null_filter = None
    if code_cols:
        first_col = sorted(code_cols)[0]
        code_null_filter = pl.col(first_col).is_not_null()

    # Compile time field
    ts_value = event_cfg.pop("time")
    ts_null_filter = None
    if ts_value is None:
        event_exprs["time"] = pl.lit(None, dtype=pl.Datetime)
    else:
        event_exprs["time"] = Parser.expr_to_polars(str(ts_value))
        ts_null_filter = event_exprs["time"].is_not_null()

    # Compile remaining fields (value columns, etc.)
    for k, v in event_cfg.items():
        if k in EVENT_META_KEYS:
            continue
        if not isinstance(v, str):
            raise ValueError(f"For event column {k}, value {v} must be a string. Got {type(v)}.")
        event_exprs[k] = Parser.expr_to_polars(v)

    # Text/numeric dedup
    if do_dedup_text_and_numeric and "numeric_value" in event_exprs and "text_value" in event_exprs:
        text_expr = event_exprs["text_value"]
        num_expr = event_exprs["numeric_value"]
        event_exprs["text_value"] = (
            pl.when(text_expr.cast(pl.Float32, strict=False) == num_expr.cast(pl.Float32))
            .then(pl.lit(None, pl.String))
            .otherwise(text_expr)
        )

    # Apply null filters and select
    if code_null_filter is not None:
        df = df.filter(code_null_filter)
    if ts_null_filter is not None:
        df = df.filter(ts_null_filter)

    return df.select(**event_exprs).unique(maintain_order=True)


def convert_to_events(
    df: pl.LazyFrame,
    event_cfgs: dict[str, dict[str, str | None | Sequence[str]]],
    do_dedup_text_and_numeric: bool = False,
) -> pl.LazyFrame:
    """Converts a DataFrame of raw data into a DataFrame of events.

    Args:
        df: The raw data DataFrame with a ``"subject_id"`` column.
        event_cfgs: Dict mapping event names to event config dicts (see ``extract_event``).
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
        >>> cfgs = {
        ...     "admit": {"code": '"ADMISSION"', "time": '$ts::"%Y-%m-%d"'},
        ...     "color": {"code": '"EYE_COLOR"', "time": None, "eye_color": "$color"},
        ... }
        >>> convert_to_events(raw, cfgs)
        shape: (4, 4)
        ┌────────────┬───────────┬─────────────────────┬───────────┐
        │ subject_id ┆ code      ┆ time                ┆ eye_color │
        │ ---        ┆ ---       ┆ ---                 ┆ ---       │
        │ i64        ┆ str       ┆ datetime[μs]        ┆ str       │
        ╞════════════╪═══════════╪═════════════════════╪═══════════╡
        │ 1          ┆ ADMISSION ┆ 2021-01-01 00:00:00 ┆ null      │
        │ 2          ┆ ADMISSION ┆ 2021-01-02 00:00:00 ┆ null      │
        │ 1          ┆ EYE_COLOR ┆ null                ┆ blue      │
        │ 2          ┆ EYE_COLOR ┆ null                ┆ green     │
        └────────────┴───────────┴─────────────────────┴───────────┘
        >>> convert_to_events(raw, {})
        Traceback (most recent call last):
            ...
        ValueError: No event configurations provided.
    """

    if not event_cfgs:
        raise ValueError("No event configurations provided.")

    event_dfs = []
    for event_name, event_cfg in event_cfgs.items():
        if event_name in EVENT_META_KEYS:
            continue

        try:
            logger.info(f"Building computational graph for extracting {event_name}")
            event_dfs.append(
                extract_event(
                    df,
                    event_cfg,
                    do_dedup_text_and_numeric=do_dedup_text_and_numeric,
                )
            )
        except Exception as e:
            raise ValueError(f"Error extracting event {event_name}: {e}") from e

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

    default_subject_id_col = event_conversion_cfg.pop("subject_id_col", "subject_id")

    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(event_conversion_cfg, out_dir / "event_conversion_config.yaml")

    subject_splits = list(shards.items())
    random.shuffle(subject_splits)

    event_configs = list(event_conversion_cfg.items())
    random.shuffle(event_configs)

    cloud_io_storage_options = cfg.get("cloud_io_storage_options", {})

    read_fn = partial(pl.scan_parquet, glob=False, storage_options=cloud_io_storage_options)

    all_input_prefixes = {pfx for pfx, _ in event_configs}

    for sp, _ in subject_splits:
        for input_prefix, event_cfgs in event_configs:
            input_fp = input_dir / sp / f"{input_prefix}.parquet"

            if not input_fp.is_file():
                input_fp_glob = f"{input_prefix}*.parquet"
                matching_files = list((input_dir / sp).glob(f"{input_fp_glob}"))
                if len(matching_files) == 1:
                    fp = matching_files[0]

                    matching_prefixes = {pfx for pfx in all_input_prefixes if fp.stem.startswith(pfx)}
                    if len(matching_prefixes) != 1:  # pragma: no cover
                        logger.warning(
                            f"Found multiple matching prefixes for {input_fp}: {', '.join(matching_prefixes)}"
                        )
                    else:
                        logger.info(f"Found matching file {matching_files[0]} for {input_fp}")
                        input_fp = matching_files[0]

            out_fp = out_dir / sp / f"{input_prefix}.parquet"

            event_cfgs = copy.deepcopy(event_cfgs)
            input_subject_id_column = event_cfgs.pop("subject_id_col", default_subject_id_col)
            subject_id_expr_str = event_cfgs.pop("subject_id_expr", None)
            transforms_cfg = event_cfgs.pop("transforms", None)
            event_cfgs.pop("schema", None)

            def compute_fntr(
                input_subject_id_column: str,
                subject_id_expr_str: str | None,
                transforms_cfg: dict | None,
                input_prefix: str,
                event_cfgs: dict,
                sp: str,
            ) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
                def compute_fn(df: pl.LazyFrame) -> pl.LazyFrame:
                    if subject_id_expr_str is not None:
                        sid_expr, _ = compile_subject_id_expr(subject_id_expr_str)
                        df = df.with_columns(subject_id=sid_expr)
                    elif input_subject_id_column != "subject_id":
                        df = df.rename({input_subject_id_column: "subject_id"})

                    if transforms_cfg is not None:
                        transform_exprs = Parser.to_polars(dict(transforms_cfg))
                        df = df.with_columns(**transform_exprs)

                    try:
                        logger.info(f"Extracting events for {input_prefix}")
                        return convert_to_events(
                            df,
                            event_cfgs=copy.deepcopy(event_cfgs),
                            do_dedup_text_and_numeric=cfg.stage_cfg.get("do_dedup_text_and_numeric", False),
                        )
                    except Exception as e:  # pragma: no cover
                        raise ValueError(f"Error converting to MEDS for {sp}/{input_prefix}: {e}") from e

                return compute_fn

            rwlock_wrap(
                input_fp,
                out_fp,
                read_fn,
                write_df,
                compute_fntr(
                    input_subject_id_column,
                    subject_id_expr_str,
                    transforms_cfg,
                    input_prefix,
                    event_cfgs,
                    sp,
                ),
                do_overwrite=cfg.do_overwrite,
            )

    logger.info("Subsharded into converted events.")
