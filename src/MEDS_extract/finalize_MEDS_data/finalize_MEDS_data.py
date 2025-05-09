"""Sets the MEDS data files to the right schema."""

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from meds import DataSchema
from MEDS_transforms.mapreduce import map_stage
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig


@Stage.register(is_metadata=False)
def main(cfg: DictConfig):
    """Writes out schema compliant MEDS data files for the extracted dataset.

    In particular, this script ensures that all shard files are MEDS compliant with the mandatory columns
      - `subject_id` (Int64)
      - `time` (DateTime)
      - `code` (String)
      - `numeric_value` (Float32)

    This stage *_should almost always be the last data stage in an extraction pipeline._*
    """

    def map_fn(df: pl.LazyFrame) -> pa.Table:
        return DataSchema.align(df.collect().to_arrow())

    map_stage(cfg, map_fn=map_fn, write_fn=pq.write_table)
