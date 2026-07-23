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
from MEDS_transforms.mapreduce.rwlock import is_complete_parquet_file, rwlock_wrap
from MEDS_transforms.parser import cfg_to_expr
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig
from upath import UPath

from .._stage_example import MEDSExtractStageExample
from ..config import SOURCE_BLOCK_COL, MessyConfig
from ..io import resolve_source_files, scan_source

logger = logging.getLogger(__name__)

# TODO(mmd): This should really somehow be pulled from MEDS.
MEDS_METADATA_MANDATORY_TYPES = {
    CodeMetadataSchema.code_name: pl.String,
    CodeMetadataSchema.description_name: pl.String,
    CodeMetadataSchema.parent_codes_name: pl.List(pl.String),
}

# Reserved internal alias for the fully-assembled output code inside the code-component map.
# Keeping the full code under this name (rather than "code") means unnesting the
# ``code_components`` struct can never collide with it — a code expression may reference
# a source column literally named ``code`` (the idiomatic ICD/OMOP vocabulary shape).
FULL_CODE_COL = "__meds_full_code"


def normalize_join_key(expr: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    """Render the join-key expression ``expr`` of type ``dtype`` as a canonical String expression.

    Code components keep their raw source dtypes (an ``Int64`` ``itemid``, say), while
    csv-sourced metadata keys are uniformly ``String`` (they are read with
    ``infer_schema=False``). Joining the two directly is a ``SchemaError``, so both sides
    of the partial-match join are normalized through this single canonical string
    rendering:

    - String expressions pass through unchanged.
    - Non-float expressions cast directly: ``220045`` renders as ``"220045"``.
    - Float expressions render integer-valued entries via ``Int64`` so that ``220045.0``
      matches the metadata string ``"220045"`` rather than rendering as ``"220045.0"``;
      non-integer values keep their float rendering (``1.5`` renders as ``"1.5"``).
      Integer-valued floats outside the ``Int64`` range render as null (and thus match
      nothing).

    The output keeps the input expression's root name (nulls stay null and never match
    under the default ``join_nulls=False``).

    Examples:
        >>> df = pl.DataFrame({
        ...     "i": [220045, 13, None],
        ...     "f": [220045.0, 1.5, None],
        ...     "s": ["220045", "01.90", None],
        ... })
        >>> df.select(
        ...     normalize_join_key(pl.col("i"), df.schema["i"]),
        ...     normalize_join_key(pl.col("f"), df.schema["f"]),
        ...     normalize_join_key(pl.col("s"), df.schema["s"]),
        ... )
        shape: (3, 3)
        ┌────────┬────────┬────────┐
        │ i      ┆ f      ┆ s      │
        │ ---    ┆ ---    ┆ ---    │
        │ str    ┆ str    ┆ str    │
        ╞════════╪════════╪════════╡
        │ 220045 ┆ 220045 ┆ 220045 │
        │ 13     ┆ 1.5    ┆ 01.90  │
        │ null   ┆ null   ┆ null   │
        └────────┴────────┴────────┘
    """
    if dtype == pl.String:
        return expr
    if dtype.is_float():
        return (
            pl.when(expr == expr.round(0))
            .then(expr.cast(pl.Int64, strict=False).cast(pl.String))
            .otherwise(expr.cast(pl.String))
            .name.keep()
        )
    return expr.cast(pl.String)


def validate_event_data_schema(data_schema: pl.Schema) -> bool:
    """Validate extracted-event input schema for metadata extraction; return whether components exist.

    ``code_components`` is only attached by ``EventConfig.extract`` when a code expression
    references at least one source column — a dataset whose codes are all literals
    legitimately has no components (and therefore nothing to partial-match on), so its
    absence is allowed and reported as ``False``.

    When components ARE present, ``source_block`` must be too: ``EventConfig.extract``
    stamps it on every row unconditionally, so its absence means the events were produced
    by a pre-0.7 extraction pipeline — and without it, partial-match metadata cannot be
    scoped to the event that declared it (one event's metadata would silently attach to
    other events' codes sharing a component value).

    Examples:
        >>> validate_event_data_schema(pl.Schema({"code": pl.String}))
        False
        >>> validate_event_data_schema(pl.Schema({
        ...     "code": pl.String,
        ...     "code_components": pl.Struct({"itemid": pl.Int64}),
        ...     "source_block": pl.String,
        ... }))
        True
        >>> validate_event_data_schema(pl.Schema({
        ...     "code": pl.String,
        ...     "code_components": pl.Struct({"itemid": pl.Int64}),
        ... }))
        Traceback (most recent call last):
            ...
        ValueError: Extracted event data carries 'code_components' but no 'source_block' column. ...
    """
    if "code_components" not in data_schema:
        return False
    if SOURCE_BLOCK_COL not in data_schema:
        raise ValueError(
            f"Extracted event data carries 'code_components' but no {SOURCE_BLOCK_COL!r} "
            "column. These events were produced by a pre-0.7 convert_to_MEDS_events; "
            "re-run the extraction pipeline before extracting code metadata."
        )
    return True


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

        Referencing a metadata column that doesn't exist in the table raises, naming the
        missing column and the columns that were available:

        >>> extract_metadata(raw_metadata.drop("code_modifier"), event_cfg)
        Traceback (most recent call last):
            ...
        KeyError: "Columns {'code_modifier'} not found in metadata columns: ['code', 'name', 'priority']"

    A ``_match_on`` column may itself be a ``_metadata`` output expression — the key-rename
    mechanism, sourcing the join key from a differently-named metadata column. The renamed key
    is emitted exactly once, as the join key (previously this collided with the metadata
    output and raised ``DuplicateError``):
        >>> raw_metadata = pl.DataFrame({
        ...     "itemid_alias": ["220045"],
        ...     "label": ["Heart Rate"],
        ... })
        >>> event_cfg = {
        ...     "code": 'f"CHART//{$itemid}"',
        ...     "_metadata": {"_match_on": "itemid", "itemid": "itemid_alias", "description": "label"},
        ... }
        >>> extract_metadata(raw_metadata, event_cfg)
        shape: (1, 3)
        ┌────────┬─────────────────────┬─────────────┐
        │ itemid ┆ code_template       ┆ description │
        │ ---    ┆ ---                 ┆ ---         │
        │ str    ┆ str                 ┆ str         │
        ╞════════╪═════════════════════╪═════════════╡
        │ 220045 ┆ f"CHART//{$itemid}" ┆ Heart Rate  │
        └────────┴─────────────────────┴─────────────┘

    Multi-column ``_match_on`` keys are all carried through, ahead of the metadata outputs:

        >>> raw_metadata = pl.DataFrame({"a": ["X", "Y"], "b": ["1", "2"], "desc": ["X-1", "Y-2"]})
        >>> event_cfg = {
        ...     "code": 'f"{$a}//{$b}//{$c}"',
        ...     "_metadata": {"_match_on": ["a", "b"], "description": "desc"},
        ... }
        >>> extract_metadata(raw_metadata, event_cfg)
        shape: (2, 4)
        ┌─────┬─────┬─────────────────────┬─────────────┐
        │ a   ┆ b   ┆ code_template       ┆ description │
        │ --- ┆ --- ┆ ---                 ┆ ---         │
        │ str ┆ str ┆ str                 ┆ str         │
        ╞═════╪═════╪═════════════════════╪═════════════╡
        │ X   ┆ 1   ┆ f"{$a}//{$b}//{$c}" ┆ X-1         │
        │ Y   ┆ 2   ┆ f"{$a}//{$b}//{$c}" ┆ Y-2         │
        └─────┴─────┴─────────────────────┴─────────────┘

    ``_match_on`` validation errors name the offending column. A ``_match_on`` column must be
    referenced by the code expression:

        >>> extract_metadata(
        ...     pl.DataFrame({"medication_name": ["X"], "desc": ["Y"]}),
        ...     {"code": "$medication_name",
        ...      "_metadata": {"_match_on": "typo_column", "description": "desc"}},
        ... )
        Traceback (most recent call last):
            ...
        KeyError: "_match_on columns {'typo_column'} are not referenced by the code expression
        '$medication_name'. Valid columns: {'medication_name'}"

    ...must exist in the metadata table:

        >>> extract_metadata(
        ...     pl.DataFrame({"dose": ["500mg"], "desc": ["some desc"]}),
        ...     {"code": 'f"{$medication_name}//{$dose}"',
        ...      "_metadata": {"_match_on": "medication_name", "description": "desc"}},
        ... )
        Traceback (most recent call last):
            ...
        KeyError: "_match_on columns {'medication_name'} not found in metadata columns: ['dose', 'desc']"

    ...and metadata output columns must exist even in partial-match mode:

        >>> extract_metadata(
        ...     pl.DataFrame({"medication_name": ["X"]}),
        ...     {"code": 'f"{$medication_name}//{$dose}"',
        ...      "_metadata": {"_match_on": "medication_name", "description": "nonexistent_col"}},
        ... )
        Traceback (most recent call last):
            ...
        KeyError: "Columns {'nonexistent_col'} not found in metadata columns: ['medication_name']"

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

    # code may be either a raw dftly string (direct-call/doctest path) or a
    # pre-parsed NodeBase (when dispatched from `MessyConfig.events_by_metadata_prefix`).
    from dftly.nodes.base import NodeBase

    code_value = event_cfg.pop("code")
    if isinstance(code_value, NodeBase):
        code_node = code_value
        code_template_str = repr(code_node)
    else:
        code_template_str = str(code_value)
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
        # A _match_on column may itself be a _metadata output expression — the only key-rename
        # mechanism available (e.g. ``_match_on: itemid`` with ``itemid: $itemid_alias`` sourcing
        # the join key from a differently-named metadata column). Such columns are already
        # selected as join keys, so they must not be selected a second time as outputs (#137's
        # DuplicateError crash).
        non_key_cols = [c for c in final_cols if c not in match_on]
        return metadata_df.unique(maintain_order=True).select(*match_on, "code_template", *non_key_cols)
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


def atomic_write_parquet(df: pl.LazyFrame | pl.DataFrame, out_fp: Path) -> None:
    """Write ``df`` to ``out_fp`` atomically — write to a sibling ``.tmp`` then rename.

    The upstream :func:`MEDS_transforms.dataframe.write_df` calls
    :meth:`pl.DataFrame.write_parquet` directly, which opens the destination
    with ``O_TRUNC`` and writes in-place — the path is observable as a
    zero-byte (or partial) file for the duration of the write. Under
    ``N_WORKERS>1`` another worker's reducer can ``stat`` the path between
    ``O_TRUNC`` and footer flush and (a) wrongly conclude the file is "ready"
    via :meth:`Path.exists`, then (b) crash on
    ``pl.scan_parquet(fp, glob=False)`` with the file-out-of-specification
    error from #51.

    Writing to ``out_fp.with_suffix(out_fp.suffix + '.tmp')`` then
    :meth:`Path.replace`-ing into place makes the destination atomically
    appear as a fully-written file (POSIX ``rename(2)`` is atomic within a
    filesystem). No reader can ever observe a partial state.

    The :func:`wait_for_complete_parquets` helper still uses
    :func:`is_complete_parquet_file` rather than :meth:`Path.exists` as
    defense-in-depth — protects against any non-stage-managed parquet that
    might land in the partial-metadata directory through a different code
    path.
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    tmp_fp = out_fp.with_suffix(out_fp.suffix + ".tmp")
    df.write_parquet(tmp_fp, use_pyarrow=True)
    tmp_fp.replace(out_fp)


def wait_for_complete_parquets(fps: list[Path], polling_time: float) -> None:
    """Block until every path in ``fps`` is a fully-flushed parquet file.

    Uses :func:`MEDS_transforms.mapreduce.rwlock.is_complete_parquet_file`
    rather than :meth:`Path.exists` because :meth:`pl.DataFrame.write_parquet`
    opens its destination with ``O_TRUNC`` and no temp-then-rename, so the
    target path is observable as a zero-byte file mid-write under
    ``N_WORKERS>1``. The reducer's downstream
    ``pl.scan_parquet(fp, glob=False)`` then raises
    ``ComputeError: parquet: File out of specification...`` (#51).

    Extracted from :func:`main` so the regression test in
    ``tests/test_extract_code_metadata_race.py`` can call it directly — that
    way the test exercises the actual production polling logic, not a copy
    that could drift if the polling check is changed in one place but not
    the other.

    Sleeps :data:`polling_time` seconds between checks. Loops forever; the
    caller is responsible for any timeout (in practice a Hydra job timeout
    or a user ``Ctrl-C``).
    """
    while not all(is_complete_parquet_file(fp) for fp in fps):
        missing_files_str = "\n".join(
            f"  - {fp.resolve()!s}" for fp in fps if not is_complete_parquet_file(fp)
        )
        logger.info(f"Waiting to begin reduction for all files to be written...\n{missing_files_str}")
        time.sleep(polling_time)


@Stage.register(is_metadata=True, example_class=MEDSExtractStageExample)
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

    The reduced output has a canonical, data-independent shape (#137): every extracted metadata column
    other than `description` is aggregated per code to a `List(String)` — a single-source unique code
    yields a one-element list — with values sorted within each group; `code_template` is aggregated to
    the sorted unique list of every contributing source's template; `description` keeps its MEDS-mandated
    separator-joined String form. A code with no value for a column gets a null (never an empty list or
    empty string). Reduction happens in canonical `(metadata_prefix, config_index)` order and the output
    is sorted by code, so `codes.parquet` is byte-identical across runs regardless of worker scheduling.

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

    messy_cfg = MessyConfig.load(cfg.event_conversion_config_fp)

    partial_metadata_dir.mkdir(parents=True, exist_ok=True)

    events_and_metadata_by_metadata_fp = messy_cfg.events_by_metadata_prefix()
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

    # Build the code_components mapping for partial metadata joins: full code (under the
    # reserved collision-proof alias), unnested component columns, and the declaring
    # source_block so each expansion joins only against its own event's codes.
    if validate_event_data_schema(all_data.collect_schema()):
        code_component_map = (
            all_data.select(pl.col("code").alias(FULL_CODE_COL), "code_components", SOURCE_BLOCK_COL)
            .unique()
            .collect()
            .unnest("code_components")
        )
    else:
        code_component_map = None

    all_out_fps = []
    # Deterministic reduction order (#137): partial files are produced in each worker's
    # shuffled config order, but the reduction must not depend on which worker shuffled how.
    # Record the canonical (metadata_prefix, cfg_idx) key for every output file so the
    # reducer can sort frames back into config order before concatenating.
    out_fp_keys: dict[Path, tuple[str, int]] = {}
    # Explicit bookkeeping for partial-match outputs: out_fp -> (match_cols, source_block).
    # The reducer is driven by this record rather than by sniffing output schemas — a partial
    # output whose _match_on includes a column named "code" would otherwise be misclassified
    # as full-match and land raw source codes in codes.parquet.
    partial_info: dict[Path, tuple[list[str], str]] = {}
    for input_prefix, event_metadata_cfgs in event_metadata_configs:
        event_metadata_cfgs = copy.deepcopy(event_metadata_cfgs)

        metadata_fps = resolve_source_files(raw_input_dir, input_prefix)

        # ``infer_schema=False`` gives csv sources a uniform all-String schema. Format
        # dispatch happens per file inside ``scan_source`` (csv-only kwargs are dropped for
        # parquet chunks), so a prefix mixing csv and parquet chunks reads cleanly (#137).
        def read_fn(fps):
            return scan_source(fps, infer_schema=False)

        # Write one output file per individual event config so each is unambiguously
        # full-match or partial-match. A single metadata prefix can be referenced by
        # multiple event configs with different match modes.
        for cfg_idx, event_cfg in enumerate(event_metadata_cfgs):
            out_fp = partial_metadata_dir / f"{input_prefix}_{cfg_idx}.parquet"
            logger.info(f"Extracting metadata from {metadata_fps} and saving to {out_fp}")

            # Always present: ``events_by_metadata_prefix`` stamps every entry (see
            # SOURCE_BLOCK_COL in config.py); a KeyError here means that contract broke.
            source_block = event_cfg.pop(SOURCE_BLOCK_COL)

            compute_fn = partial(
                extract_all_metadata,
                event_cfgs=[event_cfg],
                allowed_codes=all_codes,
            )

            rwlock_wrap(
                metadata_fps,
                out_fp,
                read_fn,
                atomic_write_parquet,
                compute_fn,
                do_overwrite=cfg.do_overwrite,
            )
            all_out_fps.append(out_fp)
            out_fp_keys[out_fp] = (input_prefix, cfg_idx)

            # Record _match_on columns and the declaring event's source_block for this shard
            # so the reducer can classify and scope it explicitly.
            match_on = event_cfg.get("_metadata", {}).get("_match_on")
            if match_on is not None:
                if isinstance(match_on, str):
                    match_on = [match_on]
                partial_info[out_fp] = (list(match_on), source_block)

    logger.info("Extracted metadata for all events. Merging.")

    if cfg.worker != 0:  # pragma: no cover
        logger.info("Code metadata extraction completed. Exiting")
        return

    logger.info("Starting reduction process")

    wait_for_complete_parquets(all_out_fps, polling_time=cfg.polling_time)

    start = datetime.now(tz=UTC)
    logger.info("All map shards complete! Starting code metadata reduction computation.")

    # Separate partial-match outputs from full-match outputs. Classification is driven by the
    # explicit partial_info record from map time, never by output schemas — a partial output
    # whose _match_on includes "code" carries a "code" column of raw component values and
    # would be silently misclassified as full-match by schema sniffing.
    #
    # Frames are processed in canonical (metadata_prefix, cfg_idx) order — NOT in this
    # worker's shuffled map order — so the reduction (concat order, and with it description
    # join order and list-aggregation order) is identical across runs and workers.
    full_match_dfs = []
    partial_match_dfs = []
    for fp in sorted(all_out_fps, key=out_fp_keys.__getitem__):
        df = pl.scan_parquet(fp, glob=False)
        if fp in partial_info:
            match_cols, source_block = partial_info[fp]
            partial_match_dfs.append((fp, df, match_cols, source_block))
        else:
            full_match_dfs.append(df)

    # Expand partial-match metadata to full codes via code_components
    if partial_match_dfs and code_component_map is not None:
        component_schema = code_component_map.schema
        for fp, pdf, match_cols, source_block in partial_match_dfs:
            pdf_schema = pdf.collect_schema()
            metadata_cols_partial = [c for c in pdf_schema.names() if c not in match_cols]

            missing = [c for c in match_cols if c not in component_schema]
            if missing:
                logger.warning(
                    f"Partial-match metadata from {fp} (source block {source_block!r}) requires "
                    f"component columns {missing} that are absent from the extracted data. Skipping."
                )
                continue

            components = code_component_map.lazy()
            # Scope the expansion to the event config that declared this _metadata block —
            # other events may reference same-named component columns with colliding values,
            # and must not receive this metadata.
            components = components.filter(pl.col(SOURCE_BLOCK_COL) == source_block)

            # Restrict the left side to exactly the full code and the join keys: any other
            # component column sharing a name with a metadata output column would otherwise
            # shadow it in the post-join select. Join keys are normalized to a canonical
            # String rendering on both sides — components keep raw source dtypes while
            # csv-sourced metadata keys are uniformly String.
            components = components.select(
                FULL_CODE_COL,
                *[normalize_join_key(pl.col(c), component_schema[c]) for c in match_cols],
            ).unique()
            pdf = pdf.with_columns(normalize_join_key(pl.col(c), pdf_schema[c]) for c in match_cols)

            expanded = (
                components.join(pdf, on=match_cols, how="inner")
                .select(pl.col(FULL_CODE_COL).alias("code"), *metadata_cols_partial)
                .collect()
            )
            if expanded.is_empty():
                logger.warning(
                    f"Partial-match metadata from {fp} (source block {source_block!r}, "
                    f"match columns {match_cols}) matched zero codes."
                )
            full_match_dfs.append(expanded.lazy())
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

    # Aggregation is UNCONDITIONAL so the output schema never depends on whether some code
    # happened to be duplicated across metadata rows (#137). Canonical output shape:
    #   - ``description``: String, non-null values joined with ``description_separator``
    #     (MEDS-mandated String semantics; joined in canonical config order).
    #   - ``parent_codes``: List(String), flattened across sources, nulls dropped.
    #   - ``code_template``: sorted unique List(String) — every contributing source's
    #     template is preserved (``.first()`` used to silently drop the others' provenance).
    #   - every other metadata column: List(String), nulls dropped, values sorted within each
    #     group so multi-source aggregation is order-stable.
    # A code with no value for a column gets null (never an empty list / empty string).
    reduced_schema = reduced.collect_schema()
    aggs = {}
    for c in metadata_cols:
        if c == CodeMetadataSchema.description_name:
            aggs[c] = pl.col(c).drop_nulls()
        elif c == CodeMetadataSchema.parent_codes_name:
            aggs[c] = pl.col(c).explode().drop_nulls()
        elif c == "code_template":
            aggs[c] = pl.col(c).drop_nulls().unique().sort()
        elif isinstance(reduced_schema[c], pl.List):
            aggs[c] = pl.col(c).explode().cast(pl.String).drop_nulls().sort()
        else:
            aggs[c] = pl.col(c).cast(pl.String).drop_nulls().sort()

    reduced = reduced.group_by(join_cols, maintain_order=True).agg(**aggs)

    empty_to_null = []
    for c in metadata_cols:
        if c == CodeMetadataSchema.description_name:
            separator = cfg.stage_cfg.description_separator
            expr = (
                pl.when(pl.col(c).list.len() > 0)
                .then(pl.col(c).list.join(separator))
                .otherwise(pl.lit(None, dtype=pl.String))
            )
        else:
            expr = pl.when(pl.col(c).list.len() > 0).then(pl.col(c)).otherwise(None)
        empty_to_null.append(expr.alias(c))
    if empty_to_null:
        reduced = reduced.with_columns(empty_to_null)

    reduced = reduced.collect()

    metadata_input_dir = Path(cfg.stage_cfg.metadata_input_dir)
    old_metadata_fp = metadata_input_dir / "codes.parquet"

    if old_metadata_fp.exists():
        logger.info(f"Joining to existing code metadata at {old_metadata_fp.resolve()!s}")
        existing = pl.read_parquet(old_metadata_fp, use_pyarrow=True)
        # Same-named metadata columns are coalesced explicitly — freshly extracted values
        # take precedence, pre-existing values survive wherever nothing was re-extracted.
        # (The bare full join used to silently fork overlaps into ``*_right`` columns; #137.)
        overlap = [c for c in existing.columns if c in reduced.columns and c not in join_cols]
        for c in overlap:
            if existing.schema[c] != reduced.schema[c]:
                raise ValueError(
                    f"Cannot merge extracted metadata with pre-existing codes.parquet at "
                    f"{old_metadata_fp.resolve()!s}: column '{c}' has dtype "
                    f"{existing.schema[c]} in the pre-existing file but {reduced.schema[c]} "
                    "in the extracted metadata."
                )
        reduced = existing.join(reduced, on=join_cols, how="full", coalesce=True, suffix="_right")
        if overlap:
            reduced = reduced.with_columns(
                pl.coalesce(pl.col(f"{c}_right"), pl.col(c)).alias(c) for c in overlap
            ).drop([f"{c}_right" for c in overlap])

    # Sort so the on-disk file is byte-identical across runs regardless of worker shuffle
    # or join execution order.
    reduced = reduced.sort(join_cols)

    reducer_fp = Path(cfg.stage_cfg.reducer_output_dir) / "codes.parquet"
    reducer_fp.parent.mkdir(parents=True, exist_ok=True)
    reduced.write_parquet(reducer_fp, use_pyarrow=True)
    logger.info(f"Finished reduction in {datetime.now(tz=UTC) - start}")
