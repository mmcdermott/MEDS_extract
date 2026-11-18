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
    DataSchema,
    DatasetMetadataSchema,
    SubjectSplitSchema,
    code_metadata_filepath,
    dataset_metadata_filepath,
    subject_splits_filepath,
)
from meds import __version__ as MEDS_VERSION
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig, OmegaConf

from .._stage_example import MEDSExtractStageExample

logger = logging.getLogger(__name__)

# `dataset.json` fields copied verbatim from the stage config when set.
DATASET_DESCRIPTOR_FIELDS = ("license", "location_uri", "description_uri")

# The MEDS extension-column categories a user may assign columns to. The fifth category,
# `other_extension_columns`, is always derived: every extension column not assigned here.
EXTENSION_COLUMN_FIELDS = (
    "raw_source_id_columns",
    "code_modifier_columns",
    "additional_value_modality_columns",
    "site_id_columns",
)

CORE_DATA_COLUMNS = frozenset(DataSchema.schema().names)


def observed_extension_columns(data_dir: Path | None) -> dict[str, pl.DataType] | None:
    """Returns the non-core columns of the cohort's data shards, in first-seen order.

    Only parquet footers are read, so this is cheap regardless of the cohort's size. Shards may carry
    different column sets (a shard with no events from some table lacks that table's properties), so the
    result is the union over all shards. Hidden directories (e.g. stage logs) are skipped.

    Args:
        data_dir: The directory holding the merged data shards, or ``None``.

    Returns:
        A mapping from column name to dtype, or ``None`` if there is no data to inspect.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     (d / "train").mkdir()
        ...     pl.DataFrame({"subject_id": [1], "code": ["A"], "hadm_id": [7]}).write_parquet(
        ...         d / "train" / "0.parquet"
        ...     )
        ...     pl.DataFrame({"subject_id": [2], "code": ["B"], "source_block": ["t/e"]}).write_parquet(
        ...         d / "train" / "1.parquet"
        ...     )
        ...     observed_extension_columns(d)
        {'hadm_id': Int64, 'source_block': String}
        >>> print(observed_extension_columns(None))
        None
    """
    if data_dir is None:
        return None
    data_dir = Path(data_dir)
    fps = sorted(
        fp
        for fp in data_dir.rglob("*.parquet")
        if not any(part.startswith(".") for part in fp.relative_to(data_dir).parts)
    )
    if not fps:
        return None

    columns: dict[str, pl.DataType] = {}
    for fp in fps:
        for name, dtype in pl.read_parquet_schema(fp).items():
            if name not in CORE_DATA_COLUMNS:
                columns.setdefault(name, dtype)
    return columns


def resolve_extension_columns(
    declared: dict[str, list[str] | None], observed: dict[str, pl.DataType] | None
) -> dict[str, list[str]]:
    """Validates the declared extension-column categories and derives ``other_extension_columns``.

    Args:
        declared: The user's category assignments, keyed by ``EXTENSION_COLUMN_FIELDS`` (``None`` = unset).
        observed: The data's extension columns and dtypes (see :func:`observed_extension_columns`), or
            ``None`` if there is no data to validate against.

    Returns:
        The ``dataset.json`` extension-column fields: every declared category, plus
        ``other_extension_columns`` when the data could be inspected.

    Raises:
        ValueError: If a column is a core MEDS column, is assigned to more than one category, is absent
            from the data, or is a non-string code modifier.

    Examples:
        Unassigned extension columns land in ``other_extension_columns``:

        >>> observed = {"hadm_id": pl.Int64, "unit": pl.String, "source_block": pl.String}
        >>> resolve_extension_columns(
        ...     {"raw_source_id_columns": ["hadm_id"], "code_modifier_columns": ["unit"]}, observed
        ... )
        {'raw_source_id_columns': ['hadm_id'], 'code_modifier_columns': ['unit'],
         'other_extension_columns': ['source_block']}
        >>> resolve_extension_columns({}, observed)
        {'other_extension_columns': ['hadm_id', 'unit', 'source_block']}

        Without data to inspect, declarations pass through unvalidated against the data and nothing is
        derived:

        >>> resolve_extension_columns({"site_id_columns": ["site"]}, None)
        {'site_id_columns': ['site']}

        Misconfigurations are rejected:

        >>> resolve_extension_columns({"raw_source_id_columns": ["hadmid"]}, observed)
        Traceback (most recent call last):
            ...
        ValueError: raw_source_id_columns lists column(s) ['hadmid'] that are not extension columns of
        the extracted data. Extension columns: ['hadm_id', 'unit', 'source_block'].
        >>> resolve_extension_columns(
        ...     {"raw_source_id_columns": ["hadm_id"], "site_id_columns": ["hadm_id"]}, observed
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Column 'hadm_id' is assigned to both raw_source_id_columns and site_id_columns; each
        extension column belongs to at most one category.
        >>> resolve_extension_columns({"code_modifier_columns": ["hadm_id"]}, observed)
        Traceback (most recent call last):
            ...
        ValueError: code_modifier_columns must be string columns (per the MEDS dataset metadata
        schema), but 'hadm_id' is Int64.
        >>> resolve_extension_columns({"site_id_columns": ["code"]}, None)
        Traceback (most recent call last):
            ...
        ValueError: site_id_columns lists core MEDS column(s) ['code']; only extension columns can be
        categorized.
    """
    out: dict[str, list[str]] = {}
    category_of: dict[str, str] = {}
    for field in EXTENSION_COLUMN_FIELDS:
        cols = declared.get(field)
        if cols is None:
            continue
        cols = list(cols)

        if core := [c for c in cols if c in CORE_DATA_COLUMNS]:
            raise ValueError(
                f"{field} lists core MEDS column(s) {core}; only extension columns can be categorized."
            )
        if observed is not None and (missing := [c for c in cols if c not in observed]):
            raise ValueError(
                f"{field} lists column(s) {missing} that are not extension columns of the extracted "
                f"data. Extension columns: {list(observed)}."
            )
        for c in cols:
            if c in category_of:
                raise ValueError(
                    f"Column {c!r} is assigned to both {category_of[c]} and {field}; each extension "
                    "column belongs to at most one category."
                )
            category_of[c] = field
        out[field] = cols

    if observed is not None:
        for c in out.get("code_modifier_columns", []):
            if observed[c] != pl.String:
                raise ValueError(
                    "code_modifier_columns must be string columns (per the MEDS dataset metadata "
                    f"schema), but {c!r} is {observed[c]}."
                )
        out["other_extension_columns"] = [c for c in observed if c not in category_of]

    return out


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
      - `created_at` (ISO-8601 string)
      - `other_extension_columns` (list of strings): every non-core column of the extracted data that
        is not assigned to one of the categories below (written whenever the data can be inspected)
      and, when set in the stage config, the optional descriptors `license`, `location_uri`,
      `description_uri` and the extension-column categories `raw_source_id_columns`,
      `code_modifier_columns`, `additional_value_modality_columns`, `site_id_columns`
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
        stage_cfg.data_input_dir: The merged data shards, whose parquet footers determine the extension
            columns. If absent or empty, the extension columns are neither validated nor derived.
        stage_cfg.license: Optional license of the dataset.
        stage_cfg.location_uri: Optional URI where the dataset is stored.
        stage_cfg.description_uri: Optional URI describing the dataset.
        stage_cfg.raw_source_id_columns: Optional extension columns holding raw-source identifiers.
        stage_cfg.code_modifier_columns: Optional (string) extension columns that modify the code.
        stage_cfg.additional_value_modality_columns: Optional extension columns holding non-numeric,
            non-text values.
        stage_cfg.site_id_columns: Optional extension columns identifying the site of care.

    Raises:
        ValueError: If the output directory does not end in ``metadata``, or the extension-column
            categories are invalid (see :func:`resolve_extension_columns`).
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

    stage_cfg = OmegaConf.to_container(cfg.stage_cfg, resolve=True)
    data_input_dir = stage_cfg.get("data_input_dir")
    observed = observed_extension_columns(Path(data_input_dir) if data_input_dir else None)
    if observed is None:
        logger.warning(
            f"No data shards found in {data_input_dir!s}; extension columns are neither validated nor "
            "recorded in other_extension_columns."
        )
    extension_columns = resolve_extension_columns(
        {field: stage_cfg.get(field) for field in EXTENSION_COLUMN_FIELDS}, observed
    )
    descriptors = {f: stage_cfg[f] for f in DATASET_DESCRIPTOR_FIELDS if stage_cfg.get(f) is not None}

    dataset_metadata = DatasetMetadataSchema(
        **{
            "dataset_name": cfg.etl_metadata.dataset_name,
            "dataset_version": str(cfg.etl_metadata.dataset_version),
            "etl_name": cfg.etl_metadata.package_name,
            "etl_version": str(cfg.etl_metadata.package_version),
            "meds_version": MEDS_VERSION,
            "created_at": datetime.now(tz=UTC).isoformat(),
            **descriptors,
            **extension_columns,
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
