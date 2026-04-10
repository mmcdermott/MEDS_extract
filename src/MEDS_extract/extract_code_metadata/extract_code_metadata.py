"""Utilities for extracting code metadata about the codes produced for the MEDS events."""

import copy
import logging
import random
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import polars as pl
from dftly import Parser
from meds import CodeMetadataSchema
from MEDS_transforms.dataframe import write_df
from MEDS_transforms.mapreduce.rwlock import rwlock_wrap
from MEDS_transforms.parser import cfg_to_expr
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig, OmegaConf
from upath import UPath

from .utils import get_supported_fp

logger = logging.getLogger(__name__)

# TODO(mmd): This should really somehow be pulled from MEDS.
MEDS_METADATA_MANDATORY_TYPES = {
    CodeMetadataSchema.code_name: pl.String,
    CodeMetadataSchema.description_name: pl.String,
    CodeMetadataSchema.parent_codes_name: pl.List(pl.String),
}


def extract_metadata(
    metadata_df: pl.LazyFrame,
    event_cfg: dict[str, str | None],
    allowed_codes: list | None = None,
) -> pl.LazyFrame:
    """Extracts a single metadata dataframe block for an event configuration from the raw metadata.

    Args:
        df: The raw metadata DataFrame. Mandatory columns are determined by the `event_cfg` configuration
            dictionary.
        event_cfg: A dictionary containing the configuration for the event. This must contain the critical
            `"code"` key alongside a mandatory `_metadata` block, which must contain some columns that should
            be extracted from the metadata to link to the code.
            The `"code"`` value is a dftly expression: string literals must be quoted
            (e.g., ``'"MY_CODE"'``), column references use ``$`` prefix (e.g., ``$col``),
            and interpolation uses f-strings (e.g., ``f"PREFIX//{$col}"``).


    Returns:
        A DataFrame containing the metadata extracted and linked to appropriately constructed code strings for
        the event configuration. The output DataFrame will contain at least two columns: `"code"` and whatever
        metadata column is specified for extraction in the metadata block. The output dataframe will not
        necessarily be unique by code if the input metadata is not unique by code.

    Raises:
        KeyError: If the event configuration dictionary is missing the `"code"` or `"_metadata"` keys or if
            the `"_metadata_"` key is empty or if columns referenced by the event configuration dictionary are
            not found in the raw metadata.

    Examples:
        >>> extract_metadata(pl.DataFrame(), {})
        Traceback (most recent call last):
            ...
        KeyError: "Event configuration dictionary must contain 'code' key. Got: []."
        >>> extract_metadata(pl.DataFrame(), {"code": "test"})
        Traceback (most recent call last):
            ...
        KeyError: "Event configuration dictionary must contain a non-empty '_metadata' key. Got: [code]."
        >>> raw_metadata = pl.DataFrame({
        ...     "code": ["A", "B", "C", "D", "E"],
        ...     "code_modifier": ["1", "2", "3", "4", "5"],
        ...     "name": ["Code A-1", "B-2", "C with 3", "D, but 4", None],
        ...     "priority": [1, 2, 3, 4, 5],
        ... })
        >>> event_cfg = {
        ...     "code": 'f"FOO//{$code}//{$code_modifier}"',
        ...     "_metadata": {"desc": "name"},
        ... }
        >>> extract_metadata(raw_metadata, event_cfg)
        shape: (4, 3)
        ┌───────────┬─────────────────────────────────┬──────────┐
        │ code      ┆ code_template                   ┆ desc     │
        │ ---       ┆ ---                             ┆ ---      │
        │ str       ┆ str                             ┆ str      │
        ╞═══════════╪═════════════════════════════════╪══════════╡
        │ FOO//A//1 ┆ f"FOO//{$code}//{$code_modifie… ┆ Code A-1 │
        │ FOO//B//2 ┆ f"FOO//{$code}//{$code_modifie… ┆ B-2      │
        │ FOO//C//3 ┆ f"FOO//{$code}//{$code_modifie… ┆ C with 3 │
        │ FOO//D//4 ┆ f"FOO//{$code}//{$code_modifie… ┆ D, but 4 │
        └───────────┴─────────────────────────────────┴──────────┘
        >>> extract_metadata(raw_metadata, event_cfg, allowed_codes=["FOO//A//1", "FOO//C//3"])
        shape: (2, 3)
        ┌───────────┬─────────────────────────────────┬──────────┐
        │ code      ┆ code_template                   ┆ desc     │
        │ ---       ┆ ---                             ┆ ---      │
        │ str       ┆ str                             ┆ str      │
        ╞═══════════╪═════════════════════════════════╪══════════╡
        │ FOO//A//1 ┆ f"FOO//{$code}//{$code_modifie… ┆ Code A-1 │
        │ FOO//C//3 ┆ f"FOO//{$code}//{$code_modifie… ┆ C with 3 │
        └───────────┴─────────────────────────────────┴──────────┘
        >>> extract_metadata(raw_metadata.drop("code_modifier"), event_cfg)  # doctest: +SKIP
        >>> extract_metadata(raw_metadata, ['foo'])
        Traceback (most recent call last):
            ...
        TypeError: Event configuration must be a dictionary. Got: <class 'list'> ['foo'].

    You can also manipulate the columns in more complex ways when assigning metadata from the input source,
    and mandatory MEDS metadata columns will be cast to the correct types:
        >>> raw_metadata = pl.DataFrame({
        ...     "code": ["A", "A", "C", "D"],
        ...     "code_modifier": ["1", "1", "2", "3"],
        ...     "code_modifier_2": ["1", "2", "3", "4"],
        ...     "title": ["A-1-1", "A-1-2", "C-2-3", None],
        ...     "special_title": ["used", None, None, None],
        ... })
        >>> event_cfg = {
        ...     "code": 'f"FOO//{$code}//{$code_modifier}"',
        ...     "_metadata": {
        ...         "description": ["special_title", "title"],
        ...         "parent_codes": [
        ...             {"OUT_VAL/{code_modifier}/2": {"code_modifier_2": "2"}},
        ...             {"OUT_VAL_for_3/{code_modifier}": {"code_modifier_2": "3"}},
        ...             {
        ...                 "matcher": {"code_modifier_2": "4"},
        ...                 "output": {"literal": "expanded form"},
        ...             },
        ...         ],
        ...     },
        ... }
        >>> extract_metadata(raw_metadata, event_cfg)  # doctest: +SKIP
    """
    event_cfg = copy.deepcopy(event_cfg)

    if not isinstance(event_cfg, dict | DictConfig):
        raise TypeError(f"Event configuration must be a dictionary. Got: {type(event_cfg)} {event_cfg}.")

    if "code" not in event_cfg:
        raise KeyError(
            f"Event configuration dictionary must contain 'code' key. Got: [{', '.join(event_cfg.keys())}]."
        )
    if "_metadata" not in event_cfg or not event_cfg["_metadata"]:
        raise KeyError(
            "Event configuration dictionary must contain a non-empty '_metadata' key. "
            f"Got: [{', '.join(event_cfg.keys())}]."
        )

    metadata_cfg = dict(event_cfg["_metadata"])
    match_on = metadata_cfg.pop("_match_on", None)

    df_select_exprs = {}
    final_cols = []
    needed_cols = set()
    for out_col, in_cfg in metadata_cfg.items():
        in_expr, needed = cfg_to_expr(in_cfg)
        df_select_exprs[out_col] = in_expr
        final_cols.append(out_col)
        needed_cols.update(needed)

    code_template_str = str(event_cfg.pop("code"))
    code_node = Parser()(code_template_str)
    code_expr = code_node.polars_expr
    code_referenced_cols = code_node.referenced_columns

    columns = metadata_df.collect_schema().names()

    if match_on is not None:
        # Partial matching: join metadata on specific code component columns rather than the full code.
        # The metadata table only needs the _match_on columns, not all code columns.
        if isinstance(match_on, str):
            match_on = [match_on]

        invalid_match_cols = set(match_on) - code_referenced_cols
        if invalid_match_cols:
            raise KeyError(
                f"_match_on columns {invalid_match_cols} are not referenced by the code expression "
                f"'{code_template_str}'. Valid columns: {code_referenced_cols}"
            )

        missing_match_cols = set(match_on) - set(columns) - set(final_cols)
        if missing_match_cols:
            raise KeyError(f"_match_on columns {missing_match_cols} not found in metadata columns: {columns}")

        missing_metadata_cols = needed_cols - set(columns) - set(final_cols)
        if missing_metadata_cols:
            raise KeyError(f"Columns {missing_metadata_cols} not found in metadata columns: {columns}")

        for col in match_on:
            if col not in df_select_exprs:
                df_select_exprs[col] = pl.col(col)

        metadata_df = metadata_df.select(**df_select_exprs).with_columns(
            code_template=pl.lit(code_template_str),
        )

        if allowed_codes is not None:
            # For partial matching, we can't filter by exact code — we broadcast metadata to all
            # matching codes. allowed_codes filtering happens after the join in the reducer.
            pass

    else:
        # Full matching: reconstruct the complete code from the metadata table
        missing_cols = (needed_cols | code_referenced_cols) - set(columns) - set(final_cols)
        if missing_cols:
            raise KeyError(f"Columns {missing_cols} not found in metadata columns: {columns}")

        for col in code_referenced_cols:
            if col not in df_select_exprs:
                df_select_exprs[col] = pl.col(col)

        metadata_df = metadata_df.select(**df_select_exprs).with_columns(
            code=code_expr,
            code_template=pl.lit(code_template_str),
        )

        if allowed_codes:
            metadata_df = metadata_df.filter(pl.col("code").is_in(allowed_codes))

    metadata_df = metadata_df.filter(~pl.all_horizontal(*[pl.col(c).is_null() for c in final_cols]))

    for mandatory_col, mandatory_type in MEDS_METADATA_MANDATORY_TYPES.items():
        if mandatory_col not in final_cols:
            continue

        if metadata_df.collect_schema()[mandatory_col] != mandatory_type:
            logger.warning(f"Metadata column '{mandatory_col}' must be of type {mandatory_type}. Casting.")
            metadata_df = metadata_df.with_columns(pl.col(mandatory_col).cast(mandatory_type, strict=False))

    if match_on is not None:
        return metadata_df.unique(maintain_order=True).select(*match_on, "code_template", *final_cols)
    else:
        return metadata_df.unique(maintain_order=True).select("code", "code_template", *final_cols)


def extract_all_metadata(
    metadata_df: pl.LazyFrame, event_cfgs: list[dict], allowed_codes: list | None = None
) -> pl.LazyFrame:
    """Extracts all metadata for a list of event configurations.

    Args:
        metadata_df: The raw metadata DataFrame. Mandatory columns are determined by the `event_cfg`
            configurations.
        event_cfgs: A list of event configuration dictionaries. Each dictionary must contain the code
            and metadata elements.
        allowed_codes: A list of codes to allow in the output metadata. If None, all codes are allowed.

    Returns:
        A unified DF containing all metadata for all event configurations.

    Examples:
        >>> raw_metadata = pl.DataFrame({
        ...     "code": ["A", "B", "C", "D"],
        ...     "code_modifier": ["1", "2", "3", "4"],
        ...     "name": ["Code A-1", "B-2", "C with 3", "D, but 4"],
        ...     "priority": [1, 2, 3, 4],
        ... })
        >>> event_cfg_1 = {
        ...     "code": 'f"FOO//{$code}//{$code_modifier}"',
        ...     "_metadata": {"desc": "name"},
        ... }
        >>> event_cfg_2 = {
        ...     "code": 'f"BAR//{$code}//{$code_modifier}"',
        ...     "_metadata": {"desc2": "name"},
        ... }
        >>> event_cfgs = [event_cfg_1, event_cfg_2]
        >>> extract_all_metadata(
        ...     raw_metadata, event_cfgs, allowed_codes=["FOO//A//1", "BAR//B//2"]
        ... )
        shape: (2, 4)
        ┌───────────┬─────────────────────────────────┬──────────┬───────┐
        │ code      ┆ code_template                   ┆ desc     ┆ desc2 │
        │ ---       ┆ ---                             ┆ ---      ┆ ---   │
        │ str       ┆ str                             ┆ str      ┆ str   │
        ╞═══════════╪═════════════════════════════════╪══════════╪═══════╡
        │ FOO//A//1 ┆ f"FOO//{$code}//{$code_modifie… ┆ Code A-1 ┆ null  │
        │ BAR//B//2 ┆ f"BAR//{$code}//{$code_modifie… ┆ null     ┆ B-2   │
        └───────────┴─────────────────────────────────┴──────────┴───────┘
    """

    all_metadata = []
    for event_cfg in event_cfgs:
        all_metadata.append(extract_metadata(metadata_df, event_cfg, allowed_codes=allowed_codes))

    return pl.concat(all_metadata, how="diagonal_relaxed").unique(maintain_order=True)


def get_events_and_metadata_by_metadata_fp(
    event_configs: dict | DictConfig,
) -> dict[str, dict[str, dict]]:
    """Reformats the event conversion config to map metadata file input prefixes to linked event configs.

    Args:
        event_configs: The event conversion configuration dictionary.

    Returns:
        A dictionary keyed by metadata input file prefix mapping to a dictionary of event configurations that
        link to that metadata prefix.

    Examples:
        >>> event_configs = {
        ...     "subject_id_col": "MRN",
        ...     "icu/procedureevents": {
        ...         "subject_id_col": "subject_id",
        ...         "start": {
        ...             "code": 'f"PROCEDURE//START//{$itemid}"',
        ...             "_metadata": {
        ...                 "proc_datetimeevents": {"desc": ["omop_concept_name", "label"]},
        ...                 "proc_itemid": {"desc": ["omop_concept_name", "label"]},
        ...             },
        ...         },
        ...         "end": {
        ...             "code": 'f"PROCEDURE//END//{$itemid}"',
        ...             "_metadata": {
        ...                 "proc_datetimeevents": {"desc": ["omop_concept_name", "label"]},
        ...                 "proc_itemid": {"desc": ["omop_concept_name", "label"]},
        ...             },
        ...         },
        ...     },
        ...     "icu/inputevents": {
        ...         "event": {
        ...             "code": 'f"INFUSION//{$itemid}"',
        ...             "_metadata": {
        ...                 "inputevents_to_rxnorm": {"desc": 'f"{$label}"', "itemid": 'f"{$foo}"'}
        ...             },
        ...         },
        ...     },
        ... }
        >>> get_events_and_metadata_by_metadata_fp(event_configs)
        {'proc_datetimeevents': [{'code': 'f"PROCEDURE//START//{$itemid}"',
                                  '_metadata': {'desc': ['omop_concept_name', 'label']}},
                                 {'code': 'f"PROCEDURE//END//{$itemid}"',
                                  '_metadata': {'desc': ['omop_concept_name', 'label']}}],
         'proc_itemid':         [{'code': 'f"PROCEDURE//START//{$itemid}"',
                                  '_metadata': {'desc': ['omop_concept_name', 'label']}},
                                 {'code': 'f"PROCEDURE//END//{$itemid}"',
                                  '_metadata': {'desc': ['omop_concept_name', 'label']}}],
         'inputevents_to_rxnorm': [{'code': 'f"INFUSION//{$itemid}"',
                                    '_metadata': {'desc': 'f"{$label}"', 'itemid': 'f"{$foo}"'}}]}
        >>> no_metadata_event_configs = {
        ...     "icu/procedureevents": {
        ...         "start": {"code": 'f"PROCEDURE//START//{$itemid}"'},
        ...         "end": {"code": 'f"PROCEDURE//END//{$itemid}"'},
        ...     },
        ...     "icu/inputevents": {
        ...         "event": {"code": 'f"INFUSION//{$itemid}"'},
        ...     },
        ... }
        >>> get_events_and_metadata_by_metadata_fp(no_metadata_event_configs)
        {}
    """

    out = {}

    for file_pfx, event_cfgs_for_pfx in event_configs.items():
        if file_pfx == "subject_id_col":
            continue

        for event_key, event_cfg in event_cfgs_for_pfx.items():
            if event_key == "subject_id_col":
                continue

            for metadata_pfx, metadata_cfg in event_cfg.get("_metadata", {}).items():
                if metadata_pfx not in out:
                    out[metadata_pfx] = []
                metadata_entry = {"code": event_cfg["code"], "_metadata": metadata_cfg}
                out[metadata_pfx].append(metadata_entry)

    return out


@Stage.register(is_metadata=True)
def main(cfg: DictConfig):
    """Extracts any dataset-specific metadata and adds it to any existing code metadata file.

    This script can extract arbitrary, code-linked metadata columns from input mappings and add them to the
    `metadata/codes.parquet` file. The metadata columns are extracted from the raw metadata files using a
    parsing DSL that is specified in the `event_conversion_config_fp` file. See `parser.py` for more details
    on this DSL.

    Note that there are two sentinel columns in the output metadata that have certain mandates for MEDS
    compliance: The `description` column and the `parent_codes` column. The `description` column must be a
    string, and if there are multiple matches in the extracted metadata for a code, in this script they will
    be concatenated into a single string with the `description_separator` string. The `parent_codes` column
    must be a list of strings, each formatted as an OMOP vocabulary name, followed by a "/", followed by the
    OMOP concept code. This column is used to link codes to their parent codes in the OMOP vocabulary.

    All arguments are specified through the command line into the `cfg` object through Hydra.

    The `cfg.stage_cfg` object is a special key that is imputed by OmegaConf to contain the stage-specific
    configuration arguments based on the global, pipeline-level configuration file.

    Args:
        stage_cfg.description_separator: If there are multiple metadata matches for
            a row, this string will be used as a separator to join the matches for the sentinel
            `"description"` column into a single string in the output metadata, per compliance with the MEDS
            schema.
    """

    stage_input_dir = Path(cfg.stage_cfg.data_input_dir)
    partial_metadata_dir = Path(cfg.stage_cfg.output_dir)
    raw_input_dir = UPath(cfg.input_dir)

    event_conversion_cfg_fp = Path(cfg.event_conversion_config_fp)
    if not event_conversion_cfg_fp.exists():
        raise FileNotFoundError(f"Event conversion config file not found: {event_conversion_cfg_fp}")

    logger.info(f"Reading event conversion config from {event_conversion_cfg_fp}")
    event_conversion_cfg = OmegaConf.load(event_conversion_cfg_fp)
    logger.info(f"Event conversion config:\n{OmegaConf.to_yaml(event_conversion_cfg)}")

    partial_metadata_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(event_conversion_cfg, partial_metadata_dir / "event_conversion_config.yaml")

    events_and_metadata_by_metadata_fp = get_events_and_metadata_by_metadata_fp(event_conversion_cfg)
    if not events_and_metadata_by_metadata_fp:
        logger.info("No _metadata blocks in the event_conversion_config.yaml found. Exiting...")
        return

    event_metadata_configs = list(events_and_metadata_by_metadata_fp.items())
    random.shuffle(event_metadata_configs)

    # Load all codes and code_components from extracted data, handling heterogeneous schemas
    # (some event files have code_components and others don't).
    event_parquet_files = list(Path(stage_input_dir).rglob("*.parquet"))
    all_event_dfs = [pl.scan_parquet(fp, glob=False) for fp in event_parquet_files]
    all_data = pl.concat(all_event_dfs, how="diagonal_relaxed")
    all_codes = all_data.select(pl.col("code").unique()).collect().get_column("code").to_list()

    # Build code_components mapping for partial metadata joins
    data_schema = all_data.collect_schema()
    if "code_components" in data_schema:
        code_component_map = (
            all_data.select("code", "code_components").unique().collect().unnest("code_components")
        )
    else:
        code_component_map = None

    all_out_fps = []
    # Collect _match_on columns per output file for use during reduction
    match_on_by_fp: dict[Path, list[str]] = {}
    for input_prefix, event_metadata_cfgs in event_metadata_configs:
        event_metadata_cfgs = copy.deepcopy(event_metadata_cfgs)

        metadata_fps, read_fn = get_supported_fp(raw_input_dir, input_prefix)

        if isinstance(metadata_fps, Path):
            metadata_fps = [metadata_fps]

        if metadata_fps[0].suffix != ".parquet":
            read_fn = partial(read_fn, infer_schema=False)

        if len(metadata_fps) > 1:
            read_fn_raw = read_fn

            def read_fn(fps):
                return pl.concat([read_fn_raw(fp) for fp in fps], how="vertical")  # noqa: B023

            metadata_fp = metadata_fps
        else:
            metadata_fp = metadata_fps[0]

        # Write one output file per individual event config so each is unambiguously
        # full-match or partial-match. A single metadata prefix can be referenced by
        # multiple event configs with different match modes.
        for cfg_idx, event_cfg in enumerate(event_metadata_cfgs):
            out_fp = partial_metadata_dir / f"{input_prefix}_{cfg_idx}.parquet"
            logger.info(f"Extracting metadata from {metadata_fp} and saving to {out_fp}")

            compute_fn = partial(
                extract_all_metadata,
                event_cfgs=[event_cfg],
                allowed_codes=all_codes,
            )

            rwlock_wrap(
                metadata_fp,
                out_fp,
                read_fn,
                write_df,
                compute_fn,
                do_overwrite=cfg.do_overwrite,
            )
            all_out_fps.append(out_fp)

            # Record _match_on columns for this shard so the reducer can use them explicitly
            match_on = event_cfg.get("_metadata", {}).get("_match_on")
            if match_on is not None:
                if isinstance(match_on, str):
                    match_on = [match_on]
                match_on_by_fp[out_fp] = list(match_on)

    logger.info("Extracted metadata for all events. Merging.")

    if cfg.worker != 0:  # pragma: no cover
        logger.info("Code metadata extraction completed. Exiting")
        return

    logger.info("Starting reduction process")

    while not all(fp.exists() for fp in all_out_fps):  # pragma: no cover
        missing_files_str = "\n".join(f"  - {fp.resolve()!s}" for fp in all_out_fps if not fp.exists())
        logger.info(f"Waiting to begin reduction for all files to be written...\n{missing_files_str}")
        time.sleep(cfg.polling_time)

    start = datetime.now(tz=UTC)
    logger.info("All map shards complete! Starting code metadata reduction computation.")

    # Separate partial-match outputs (no "code" column) from full-match outputs
    full_match_dfs = []
    partial_match_dfs = []
    for fp in all_out_fps:
        df = pl.scan_parquet(fp, glob=False)
        if "code" in df.collect_schema():
            full_match_dfs.append(df)
        elif fp in match_on_by_fp:
            partial_match_dfs.append((df, match_on_by_fp[fp]))

    # Expand partial-match metadata to full codes via code_components
    if partial_match_dfs and code_component_map is not None:
        for pdf, match_cols in partial_match_dfs:
            pdf_cols = pdf.collect_schema().names()
            metadata_cols_partial = [c for c in pdf_cols if c not in match_cols]
            expanded = (
                code_component_map.lazy()
                .join(pdf, on=match_cols, how="inner")
                .select("code", *metadata_cols_partial)
            )
            full_match_dfs.append(expanded)
    elif partial_match_dfs:
        logger.warning("Partial-match metadata found but no code_components in data. Skipping.")

    if not full_match_dfs:
        logger.info("No metadata to reduce. Writing empty metadata file.")
        reduced = pl.DataFrame({"code": []}).cast({"code": pl.String}).lazy()
    else:
        reduced = pl.concat(full_match_dfs, how="diagonal_relaxed").unique(maintain_order=True)

    join_cols = ["code", *cfg.get("code_modifier_cols", [])]
    reduced_cols = reduced.collect_schema().names()
    metadata_cols = [c for c in reduced_cols if c not in join_cols]

    n_unique_obs = reduced.select(pl.n_unique(*join_cols)).collect().item()
    n_rows = reduced.select(pl.len()).collect().item()
    logger.info(f"Collected metadata for {n_unique_obs} unique codes among {n_rows} total observations.")

    if n_unique_obs != n_rows:
        skip_cols = {*MEDS_METADATA_MANDATORY_TYPES, "code_template"}
        aggs = {c: pl.col(c) for c in metadata_cols if c not in skip_cols}
        if "description" in metadata_cols:
            separator = cfg.stage_cfg.description_separator
            aggs["description"] = pl.col("description").str.join(separator)
        if "parent_codes" in metadata_cols:
            aggs["parent_codes"] = pl.col("parent_codes").explode()
        if "code_template" in metadata_cols:
            aggs["code_template"] = pl.col("code_template").first()

        reduced = reduced.group_by(join_cols).agg(**aggs)

    reduced = reduced.collect()

    metadata_input_dir = Path(cfg.stage_cfg.metadata_input_dir)
    old_metadata_fp = metadata_input_dir / "codes.parquet"

    if old_metadata_fp.exists():
        logger.info(f"Joining to existing code metadata at {old_metadata_fp.resolve()!s}")
        existing = pl.read_parquet(old_metadata_fp, use_pyarrow=True)
        reduced = existing.join(reduced, on=join_cols, how="full", coalesce=True)

    reducer_fp = Path(cfg.stage_cfg.reducer_output_dir) / "codes.parquet"
    reducer_fp.parent.mkdir(parents=True, exist_ok=True)
    reduced.write_parquet(reducer_fp, use_pyarrow=True)
    logger.info(f"Finished reduction in {datetime.now(tz=UTC) - start}")
