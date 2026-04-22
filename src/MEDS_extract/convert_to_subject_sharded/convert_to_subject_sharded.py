"""Stage: convert event-sharded raw data into subject-sharded format.

This is the second half of the initial ingestion phase. After ``shard_events``
has sub-sharded each raw source table into fixed-size row chunks, this stage
re-groups rows by subject: for every ``(split, table)`` pair, it reads every
sub-shard of that table, applies the table's ``subject_id`` expression (and
any join it needs), filters down to the rows whose subject is in the split,
and writes the result to ``<split>/<table>.parquet``.

For example, with a ``vitals`` table joined to ``stays`` on ``stay_id``:

.. code-block:: text

    data/vitals/[0-2).parquet  ─┐
    data/vitals/[2-4).parquet  ─┼─► data/train/0/vitals.parquet
    data/vitals/[4-6).parquet  ─┘    (vitals rows for training subjects, with
                                      the joined subject_id materialized)

Each shard is independent, so this stage parallelizes trivially across
``(split, table)`` pairs.
"""

import json
import logging
import random
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import polars as pl
from MEDS_transforms.dataframe import write_df
from MEDS_transforms.mapreduce.rwlock import rwlock_wrap
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig

from ..config import MessyConfig, TableConfig
from ..io import scan_source

logger = logging.getLogger(__name__)

pl.enable_string_cache()


@Stage.register(is_metadata=False)
def main(cfg: DictConfig):
    """Re-shard raw data by subject. See module docstring for details.

    All arguments come through the Hydra ``cfg`` object; this stage has no
    stage-specific options beyond the global ``event_conversion_config_fp``.
    """
    input_dir = Path(cfg.stage_cfg.data_input_dir)
    subject_subsharded_dir = Path(cfg.stage_cfg.output_dir)

    shards = json.loads(Path(cfg.shards_map_fp).read_text())

    messy_cfg = MessyConfig.load(cfg.event_conversion_config_fp)
    subject_subsharded_dir.mkdir(parents=True, exist_ok=True)

    subject_splits = list(shards.items())
    random.shuffle(subject_splits)

    for sp, subjects in subject_splits:
        for table in messy_cfg.shuffled_tables():
            event_shards = list(table.source_files(input_dir))
            random.shuffle(event_shards)

            out_fp = subject_subsharded_dir / sp / f"{table.input_prefix}.parquet"

            rwlock_wrap(
                event_shards,
                out_fp,
                partial(_read_and_join, table=table, input_dir=input_dir),
                write_df,
                partial(_filter_to_subjects, table=table, subjects=subjects),
                do_overwrite=cfg.do_overwrite,
            )

    logger.info("Created a subject-sharded view.")


def _read_and_join(
    fps: Sequence[Path],
    *,
    table: TableConfig,
    input_dir: Path,
) -> pl.LazyFrame:
    """Scan the given subshards and apply the table's join (if any).

    This is a straight read — no filtering — so a row-level rwlock
    wrapper can reuse the same function across multiple stages. Subject
    filtering lives in :func:`_filter_to_subjects` on the compute side.
    """
    df = scan_source(fps)
    if table.join is not None:
        df = table.join.apply(df, input_dir)
    return df


def _filter_to_subjects(df: pl.LazyFrame, *, table: TableConfig, subjects: Sequence[int]) -> pl.LazyFrame:
    """Filter ``df`` to the rows whose subject_id is in ``subjects``.

    Uses the table's ``subject_id_polars_expr`` inline in ``filter`` so that
    the output keeps its original source columns untouched — no new
    materialized ``subject_id`` column gets added.
    """
    return df.filter(table.subject_id_polars_expr.is_in(list(subjects)))
