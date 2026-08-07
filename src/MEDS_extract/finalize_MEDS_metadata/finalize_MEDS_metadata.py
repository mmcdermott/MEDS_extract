"""Utilities for finalizing the metadata files for extracted MEDS datasets."""

import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from meds import (
    CodeMetadataSchema,
    DatasetMetadataSchema,
    SubjectSplitSchema,
    code_metadata_filepath,
    dataset_metadata_filepath,
    subject_splits_filepath,
)
from meds import __version__ as MEDS_VERSION
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig

from .._stage_example import MEDSExtractStageExample

logger = logging.getLogger(__name__)


@Stage.register(is_metadata=True, example_class=MEDSExtractStageExample)
def main(cfg: DictConfig):
    """Writes out schema compliant MEDS metadata files for the extracted dataset.

    In particular, this script ensures that
    (1) a `metadata/codes.parquet` file exists, validated against the MEDS code metadata schema:
      - `code` (string) is required
      - `description` (string) and `parent_codes` (list of strings) are typed to the schema when
        present in the input, but are not added when absent (only the empty-input case emits the
        full three-column schema)
    (2) a `metadata/dataset.json` file exists that has the keys
      - `dataset_name` (string)
      - `dataset_version` (string)
      - `etl_name` (string)
      - `etl_version` (string)
      - `meds_version` (string)
    (3) a `metadata/subject_splits.parquet` file exists that has the mandatory columns
      - `subject_id` (Int64)
      - `split` (string)

    This stage *_should almost always be the last metadata stage in an extraction pipeline._*

    The stage is deterministic and cheap, so it is idempotent under resume: any pre-existing
    output files (e.g., left behind when a prior run was killed after writing its outputs but
    before the pipeline runner recorded the stage as complete) are removed and rewritten
    unconditionally. Skipping a completed stage is the runner's job, not this stage's.

    Args:
        etl_metadata.dataset_name: The name of the dataset being extracted.
        etl_metadata.dataset_version: The version of the dataset being extracted.
    """

    if cfg.worker != 0:  # pragma: no cover
        logger.info("Non-zero worker found in reduce-only stage. Exiting")
        return

    input_metadata_dir = Path(cfg.stage_cfg.metadata_input_dir)
    output_metadata_dir = Path(cfg.stage_cfg.reducer_output_dir)

    if output_metadata_dir.parts[-1] != Path(code_metadata_filepath).parts[0]:
        raise ValueError(f"Output metadata directory must end in 'metadata'. Got {output_metadata_dir}")

    output_code_metadata_fp = output_metadata_dir.parent / code_metadata_filepath
    dataset_metadata_fp = output_metadata_dir.parent / dataset_metadata_filepath
    subject_splits_fp = output_metadata_dir.parent / subject_splits_filepath

    for out_fp in [output_code_metadata_fp, dataset_metadata_fp, subject_splits_fp]:
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        out_fp.unlink(missing_ok=True)

    # Code metadata validation
    logger.info("Validating code metadata")
    input_code_metadata_fp = input_metadata_dir / "codes.parquet"
    if input_code_metadata_fp.exists():
        logger.info(f"Reading code metadata from {input_code_metadata_fp.resolve()!s}")
        code_metadata = pl.read_parquet(input_code_metadata_fp, use_pyarrow=True)
        final_metadata_tbl = CodeMetadataSchema.align(code_metadata.to_arrow())
    else:
        logger.info(f"No code metadata found at {input_code_metadata_fp!s}. Making empty metadata file.")
        final_metadata_tbl = pa.Table.from_pylist([], schema=CodeMetadataSchema.schema())

    logger.info(f"Writing finalized metadata df to {output_code_metadata_fp.resolve()!s}")
    pq.write_table(final_metadata_tbl, output_code_metadata_fp)

    # Dataset metadata creation
    logger.info("Creating dataset metadata")

    dataset_metadata = DatasetMetadataSchema(
        **{
            "dataset_name": cfg.etl_metadata.dataset_name,
            "dataset_version": str(cfg.etl_metadata.dataset_version),
            "etl_name": cfg.etl_metadata.package_name,
            "etl_version": str(cfg.etl_metadata.package_version),
            "meds_version": MEDS_VERSION,
            "created_at": datetime.now(tz=UTC).isoformat(),
        }
    )

    logger.info(f"Writing finalized dataset metadata to {dataset_metadata_fp.resolve()!s}")
    dataset_metadata_fp.write_text(json.dumps(dataset_metadata.to_dict()))

    # Split creation
    shards_map_fp = Path(cfg.shards_map_fp)
    logger.info(f"Creating subject splits from {shards_map_fp.resolve()!s}")
    shards_map = json.loads(shards_map_fp.read_text())
    subject_splits = []
    seen_splits = defaultdict(int)
    for shard, subject_ids in shards_map.items():
        split = "/".join(shard.split("/")[:-1])

        seen_splits[split] += len(subject_ids)

        subject_splits.extend(
            [{SubjectSplitSchema.subject_id_name: pid, "split": split} for pid in subject_ids]
        )

    for split, cnt in seen_splits.items():
        if cnt:
            logger.info(f"Split {split} has {cnt} subjects")
        else:  # pragma: no cover
            logger.warning(f"Split {split} not found in shards map")

    subject_splits_tbl = pa.Table.from_pylist(subject_splits, schema=SubjectSplitSchema.schema())
    logger.info(f"Writing finalized subject splits to {subject_splits_fp.resolve()!s}")
    pq.write_table(subject_splits_tbl, subject_splits_fp)
