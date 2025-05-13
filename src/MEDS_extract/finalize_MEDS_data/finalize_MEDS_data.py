"""Sets the MEDS data files to the right schema."""

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from meds import DataSchema
from MEDS_transforms.stages import Stage


@Stage.register(write_fn=pq.write_table)
def finalize_MEDS_data(df: pl.LazyFrame) -> pa.Table:
    """Writes out schema compliant MEDS data files for the extracted dataset.

    In particular, this script ensures that all shard files are MEDS compliant with the mandatory columns
      - `subject_id` (Int64)
      - `time` (DateTime)
      - `code` (String)
      - `numeric_value` (Float32)

    This stage *_should almost always be the last data stage in an extraction pipeline._*
    """
    return DataSchema.align(df.collect().to_arrow())
