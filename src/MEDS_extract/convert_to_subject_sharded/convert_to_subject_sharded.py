"""Utilities for converting input data structures into MEDS events."""

import copy
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

    default_subject_id_col = event_conversion_cfg.pop("subject_id_col", "subject_id")

    subject_subsharded_dir.mkdir(parents=True, exist_ok=True)

    subject_splits = list(shards.items())
    random.shuffle(subject_splits)

    event_configs = list(event_conversion_cfg.items())
    random.shuffle(event_configs)

    for sp, subjects in subject_splits:
        for input_prefix, event_cfgs in event_configs:
            event_shards = list((input_dir / input_prefix).glob("*.parquet"))
            event_cfgs = copy.deepcopy(event_cfgs)
            random.shuffle(event_shards)

            out_fp = subject_subsharded_dir / sp / f"{input_prefix}.parquet"
            input_subject_id_column = event_cfgs.pop("subject_id_col", default_subject_id_col)

            def read_fntr(
                subjects: Sequence[int], input_subject_id_column: str
            ) -> Callable[[Sequence[Path]], pl.LazyFrame]:
                def read_fn(fps: Sequence[Path]) -> pl.LazyFrame:
                    dfs = [pl.scan_parquet(fps[0], glob=False)]

                    typed_subjects = pl.Series(subjects, dtype=dfs[0].schema[input_subject_id_column])
                    filter_expr = pl.col(input_subject_id_column).is_in(typed_subjects)

                    dfs[0] = dfs[0].filter(filter_expr)

                    for fp in fps[1:]:
                        dfs.append(pl.scan_parquet(fp, glob=False).filter(filter_expr))

                    return pl.concat(dfs, how="vertical")

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
