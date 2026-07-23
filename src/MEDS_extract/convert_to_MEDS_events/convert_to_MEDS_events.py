"""Stage: convert event-sharded raw data into MEDS events, subject-subsharded.

All extraction logic lives on :class:`MEDS_extract.config.TableConfig` and
:class:`MEDS_extract.config.EventConfig`; this module is thin orchestration
over file I/O and :func:`MEDS_transforms.mapreduce.rwlock.rwlock_wrap`.
"""

import json
import logging
import random
from functools import partial
from pathlib import Path

import polars as pl
from MEDS_transforms.dataframe import write_df
from MEDS_transforms.mapreduce.rwlock import rwlock_wrap
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig, OmegaConf
from upath import UPath

from .._stage_example import MEDSExtractStageExample
from ..config import MessyConfig
from ..io import scan_source

logger = logging.getLogger(__name__)

pl.enable_string_cache()


@Stage.register(is_metadata=False, example_class=MEDSExtractStageExample)
def main(cfg: DictConfig):
    """Convert event-sharded raw data into MEDS events per subject shard.

    All arguments are specified through the command line into the ``cfg`` object
    through Hydra.

    Stage options (via ``cfg.stage_cfg``):

    - ``do_dedup_text_and_numeric``: null out ``text_value`` when it numerically
      equals ``numeric_value``.
    - ``do_track_provenance`` (default ``False``): attach a ``provenance`` column
      (``list[struct{source_file, row_idx}]``) naming the raw source rows behind
      each MEDS row. Off by default; when off, output is byte-identical to a run
      without provenance support.
    """
    input_dir = UPath(cfg.stage_cfg.data_input_dir)
    out_dir = UPath(cfg.stage_cfg.output_dir)

    shards = json.loads(Path(cfg.shards_map_fp).read_text())

    messy_cfg = MessyConfig.load(cfg.event_conversion_config_fp)

    out_dir.mkdir(parents=True, exist_ok=True)
    messy_cfg.save(out_dir / "event_conversion_config.yaml")

    subject_splits = list(shards.items())
    random.shuffle(subject_splits)

    do_dedup = cfg.stage_cfg.get("do_dedup_text_and_numeric", False)
    do_track_provenance = cfg.stage_cfg.get("do_track_provenance", False)

    raw_opts = cfg.get("cloud_io_storage_options", {})
    storage_options = OmegaConf.to_container(raw_opts) if OmegaConf.is_config(raw_opts) else raw_opts
    read_fn = partial(scan_source, storage_options=storage_options) if storage_options else scan_source

    for sp, _ in subject_splits:
        for table in messy_cfg.shuffled_tables():
            input_fps = table.source_files(input_dir / sp)
            out_fp = out_dir / sp / f"{table.input_prefix}.parquet"

            rwlock_wrap(
                input_fps,
                out_fp,
                read_fn,
                write_df,
                partial(
                    table.extract_events,
                    do_dedup_text_and_numeric=do_dedup,
                    do_track_provenance=do_track_provenance,
                ),
                do_overwrite=cfg.do_overwrite,
            )

    logger.info("Subsharded into converted events.")
