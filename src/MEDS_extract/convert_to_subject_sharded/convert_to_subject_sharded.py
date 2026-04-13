"""Utilities for converting event-sharded raw data into subject-sharded format."""

import json
import logging
import random
from collections.abc import Callable, Sequence
from pathlib import Path

import polars as pl
from MEDS_transforms.dataframe import write_df
from MEDS_transforms.mapreduce.rwlock import rwlock_wrap
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig, OmegaConf

from ..config import MessyConfig, TableConfig

logger = logging.getLogger(__name__)

pl.enable_string_cache()


@Stage.register(is_metadata=False)
def main(cfg: DictConfig):
    """Converts the event-sharded raw data into a subject sharded format (still by original prefix).

    This step completes the first "phase" of the pipeline, which is to convert the raw data (assumed to be in
    an unsharded format) into a subject-sharded format.

    All arguments are specified through the command line into the `cfg` object through Hydra.

    This stage has no stage-specific configuration arguments. It does, naturally, require the global,
    `event_conversion_config_fp` configuration argument to be set to the path of the event conversion yaml
    file.
    """

    input_dir = Path(cfg.stage_cfg.data_input_dir)
    subject_subsharded_dir = Path(cfg.stage_cfg.output_dir)

    shards = json.loads(Path(cfg.shards_map_fp).read_text())

    event_conversion_cfg_fp = Path(cfg.event_conversion_config_fp)
    if not event_conversion_cfg_fp.exists():
        raise FileNotFoundError(f"Event conversion config file not found: {event_conversion_cfg_fp}")

    logger.info("Starting subject sharding.")

    logger.info(f"Reading event conversion config from {event_conversion_cfg_fp}")
    event_conversion_cfg = OmegaConf.load(event_conversion_cfg_fp)
    logger.info(f"Event conversion config:\n{OmegaConf.to_yaml(event_conversion_cfg)}")

    subject_subsharded_dir.mkdir(parents=True, exist_ok=True)

    messy_cfg = MessyConfig.parse(event_conversion_cfg)

    subject_splits = list(shards.items())
    random.shuffle(subject_splits)

    tables = list(messy_cfg.iter_tables())
    random.shuffle(tables)

    for sp, subjects in subject_splits:
        for table in tables:
            event_shards = list((input_dir / table.input_prefix).glob("*.parquet"))
            random.shuffle(event_shards)

            out_fp = subject_subsharded_dir / sp / f"{table.input_prefix}.parquet"
            input_subject_id_column = table.subject_id_column

            join_df = None
            if table.join is not None:
                join_fps = list((input_dir / table.join.input_prefix).glob("*.parquet"))
                join_df = pl.concat(
                    [pl.scan_parquet(fp, glob=False) for fp in join_fps],
                    how="vertical_relaxed",
                )

            def read_fntr(
                subjects: Sequence[int],
                input_subject_id_column: str,
                _join_df: pl.LazyFrame | None = join_df,
                _table: TableConfig = table,
            ) -> Callable[[Sequence[Path]], pl.LazyFrame]:
                def read_fn(fps: Sequence[Path]) -> pl.LazyFrame:
                    df = pl.concat(
                        [pl.scan_parquet(fp, glob=False) for fp in fps],
                        how="vertical",
                    )
                    if _join_df is not None:
                        df = df.join(
                            _join_df,
                            left_on=_table.join.left_on,
                            right_on=_table.join.right_on,
                            how="left",
                        )
                        subj_dtype = _join_df.collect_schema()[input_subject_id_column]
                    else:
                        subj_dtype = df.collect_schema()[input_subject_id_column]

                    typed_subjects = pl.Series(subjects, dtype=subj_dtype)
                    filter_expr = pl.col(input_subject_id_column).is_in(typed_subjects)
                    df = df.filter(filter_expr)

                    return df

                return read_fn

            def compute_fn(df: pl.LazyFrame) -> pl.LazyFrame:
                return df

            rwlock_wrap(
                event_shards,
                out_fp,
                read_fntr(subjects, input_subject_id_column),
                write_df,
                compute_fn,
                do_overwrite=cfg.do_overwrite,
            )

    logger.info("Created a subject-sharded view.")
