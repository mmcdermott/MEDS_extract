"""Stage: convert event-sharded raw data into MEDS events, subject-subsharded.

All extraction logic lives on :class:`MEDS_extract.config.TableConfig` and
:class:`MEDS_extract.config.EventConfig`; this module is thin orchestration
over file I/O and :func:`MEDS_transforms.mapreduce.rwlock.rwlock_wrap`.
"""

import json
import logging
import random
from collections.abc import Callable
from functools import partial
from pathlib import Path

import polars as pl
from MEDS_transforms.dataframe import write_df
from MEDS_transforms.mapreduce.rwlock import rwlock_wrap
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig, OmegaConf
from upath import UPath

from ..config import MessyConfig, TableConfig

logger = logging.getLogger(__name__)

pl.enable_string_cache()


@Stage.register(is_metadata=False)
def main(cfg: DictConfig):
    """Convert event-sharded raw data into MEDS events per subject shard.

    All arguments are specified through the command line into the ``cfg`` object
    through Hydra.
    """
    input_dir = UPath(cfg.stage_cfg.data_input_dir)
    out_dir = UPath(cfg.stage_cfg.output_dir)

    shards = json.loads(Path(cfg.shards_map_fp).read_text())

    messy_cfg = MessyConfig.load(cfg.event_conversion_config_fp)

    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.load(cfg.event_conversion_config_fp), out_dir / "event_conversion_config.yaml")

    subject_splits = list(shards.items())
    random.shuffle(subject_splits)

    tables = list(messy_cfg.iter_tables())
    random.shuffle(tables)

    raw_opts = cfg.get("cloud_io_storage_options", {})
    cloud_io_storage_options = OmegaConf.to_container(raw_opts) if OmegaConf.is_config(raw_opts) else raw_opts
    read_fn = partial(pl.scan_parquet, glob=False, storage_options=cloud_io_storage_options)

    do_dedup = cfg.stage_cfg.get("do_dedup_text_and_numeric", False)

    for sp, _ in subject_splits:
        for table in tables:
            if not table.events:
                continue  # Join-target-only tables have nothing to extract here.

            input_fp = _resolve_input_fp(input_dir / sp, table.input_prefix, tables)
            out_fp = out_dir / sp / f"{table.input_prefix}.parquet"

            rwlock_wrap(
                input_fp,
                out_fp,
                read_fn,
                write_df,
                _compute_fn_for(table, sp, do_dedup),
                do_overwrite=cfg.do_overwrite,
            )

    logger.info("Subsharded into converted events.")


def _resolve_input_fp(
    sp_input_dir: UPath,
    input_prefix: str,
    tables: list[TableConfig],
) -> UPath:
    """Find the input parquet for a given table prefix, tolerating suffix mismatches.

    Some stages write to ``{prefix}.parquet`` and some write to ``{prefix}-{shard}.parquet``
    (legacy). If a bare ``{prefix}.parquet`` isn't present, fall back to a glob — but only
    if exactly one prefix matches, to avoid silently picking the wrong file.
    """
    input_fp = sp_input_dir / f"{input_prefix}.parquet"
    if input_fp.is_file():
        return input_fp

    matching_files = list(sp_input_dir.glob(f"{input_prefix}*.parquet"))
    if len(matching_files) != 1:
        return input_fp

    fp = matching_files[0]
    matching_prefixes = {t.input_prefix for t in tables if fp.stem.startswith(t.input_prefix)}
    if len(matching_prefixes) == 1:
        logger.info(f"Found matching file {fp} for {input_fp}")
        return fp

    logger.warning(
        f"Ambiguous matching prefixes for {input_fp}: {', '.join(matching_prefixes)}"
    )  # pragma: no cover
    return input_fp  # pragma: no cover


def _compute_fn_for(table: TableConfig, sp: str, do_dedup: bool) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    def compute_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        try:
            return table.extract_events(df, do_dedup_text_and_numeric=do_dedup)
        except Exception as e:  # pragma: no cover
            raise ValueError(f"Error converting to MEDS for {sp}/{table.input_prefix}: {e}") from e

    return compute_fn
