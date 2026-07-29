"""Utilities for extracting code metadata about the codes produced for the MEDS events."""

import copy
import logging
import random
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import polars as pl
from dftly import Parser
from dftly.nodes.base import NodeBase
from meds import CodeMetadataSchema
from MEDS_transforms.mapreduce.rwlock import is_complete_parquet_file, rwlock_wrap
from MEDS_transforms.parser import cfg_to_expr
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig
from upath import UPath

from .._stage_example import MEDSExtractStageExample
from ..config import SOURCE_BLOCK_COL, MessyConfig
from ..io import _format_family, resolve_source_files, scan_source

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
    of the component join are normalized through this single canonical string rendering:

    - String expressions pass through unchanged.
    - Non-float expressions cast directly: ``220045`` renders as ``"220045"``.
    - Float expressions render integer-valued entries via ``Int64`` so that ``220045.0``
      matches the metadata string ``"220045"`` rather than rendering as ``"220045.0"``;
      non-integer values keep their float rendering (``1.5`` renders as ``"1.5"``).
      Integer-valued floats outside the ``Int64`` range render as null.

    The output keeps the input expression's root name. Nulls stay null — the component
    join passes ``nulls_equal=True``, so a null key matches exactly a null component.

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
    legitimately has no components (and therefore nothing to join metadata onto), so its
    absence is allowed and reported as ``False``.

    When components ARE present, ``source_block`` must be too: ``EventConfig.extract``
    stamps it on every row unconditionally, so its absence means the events were produced
    by a pre-0.7 extraction pipeline — and without it, metadata joins cannot be scoped to
    the event that declared them (one event's metadata would silently attach to other
    events' codes sharing a component value).

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


def build_code_component_map(all_data: pl.LazyFrame) -> pl.DataFrame:
    """Materialize the code-components map every reducer-side metadata join runs against.

    One row per distinct (full code, components, declaring source block): the full code
    under the reserved collision-proof :data:`FULL_CODE_COL` alias, the unnested
    component columns, and the declaring ``source_block`` so each join attaches only to
    its own event's codes.

    This is a full-dataset scan + unique + collect — the single most expensive step of
    the stage outside the map compute itself — and its output is consumed only by the
    reduction, so :func:`main` calls it exclusively in worker 0 after the map phase.

    Examples:
        >>> all_data = pl.LazyFrame({
        ...     "code": ["CHART//1", "CHART//1", "LAB//1"],
        ...     "code_components": [{"itemid": "1"}, {"itemid": "1"}, {"itemid": "1"}],
        ...     "source_block": ["chartevents/chart", "chartevents/chart", "labevents/lab"],
        ... })
        >>> build_code_component_map(all_data).sort("__meds_full_code")
        shape: (2, 3)
        ┌──────────────────┬────────┬───────────────────┐
        │ __meds_full_code ┆ itemid ┆ source_block      │
        │ ---              ┆ ---    ┆ ---               │
        │ str              ┆ str    ┆ str               │
        ╞══════════════════╪════════╪═══════════════════╡
        │ CHART//1         ┆ 1      ┆ chartevents/chart │
        │ LAB//1           ┆ 1      ┆ labevents/lab     │
        └──────────────────┴────────┴───────────────────┘
    """
    return (
        all_data.select(pl.col("code").alias(FULL_CODE_COL), "code_components", SOURCE_BLOCK_COL)
        .unique()
        .collect()
        .unnest("code_components")
    )


def _parse_code(code_value: str | NodeBase) -> tuple[NodeBase, str]:
    """Return the parsed dftly node and template string for a ``code`` config value.

    ``code`` may be either a raw dftly string (direct-call/doctest path) or a pre-parsed
    node (when dispatched from ``MessyConfig.events_by_metadata_prefix`` for an event
    whose raw expression string was not retained).
    """
    if isinstance(code_value, NodeBase):
        return code_value, repr(code_value)
    code_template_str = str(code_value)
    return Parser()(code_template_str), code_template_str


def resolve_match_columns(event_cfg: dict) -> list[str]:
    """Derive the code-component columns a ``_metadata`` entry joins on.

    Every metadata join is a component join: the metadata table's join keys are raw
    metadata columns whose names match source columns referenced by the code expression.
    By default an entry matches on *all* code-referenced columns (a full match). An
    optional ``_match_on`` narrows the key set to a subset of those columns, broadcasting
    the metadata to every code sharing the named component values.

    Args:
        event_cfg: A mapping carrying the entry's ``code`` (a raw dftly string or a
            pre-parsed node) and its ``_metadata`` block. Not mutated.

    Returns:
        The join-key column names: ``_match_on`` in declared order when given, otherwise
        all code-referenced columns in sorted order.

    Raises:
        ValueError: If the code expression is a literal (it references no source columns,
            so there are no components to match metadata on), or if a match column is also
            declared as a ``_metadata`` output expression.
        KeyError: If a ``_match_on`` column is not referenced by the code expression.

    Examples:
        >>> resolve_match_columns({
        ...     "code": 'f"LAB//{$itemid}//{$valueuom}"',
        ...     "_metadata": {"description": "label"},
        ... })
        ['itemid', 'valueuom']
        >>> resolve_match_columns({
        ...     "code": 'f"LAB//{$itemid}//{$valueuom}"',
        ...     "_metadata": {"_match_on": "itemid", "description": "label"},
        ... })
        ['itemid']

        A literal code has no components to match metadata on:

        >>> resolve_match_columns({"code": "MEDS_BIRTH", "_metadata": {"description": "label"}})
        Traceback (most recent call last):
            ...
        ValueError: The code expression 'MEDS_BIRTH' is a literal: it references no source columns, ...

        ``_match_on`` columns must be referenced by the code expression:

        >>> resolve_match_columns({
        ...     "code": "$medication_name",
        ...     "_metadata": {"_match_on": "typo_column", "description": "desc"},
        ... })
        Traceback (most recent call last):
            ...
        KeyError: "_match_on columns ['typo_column'] are not referenced by the code expression
        '$medication_name'. Valid columns: ['medication_name']"

        A match column may not also be a ``_metadata`` output expression — join keys are
        raw metadata columns, never derived ones:

        >>> resolve_match_columns({
        ...     "code": 'f"CHART//{$itemid}"',
        ...     "_metadata": {"_match_on": "itemid", "itemid": "itemid_alias", "description": "label"},
        ... })
        Traceback (most recent call last):
            ...
        ValueError: Match column(s) ['itemid'] may not also be declared as _metadata output ...
    """
    code_node, code_template_str = _parse_code(event_cfg["code"])
    referenced_cols = set(code_node.referenced_columns)

    if not referenced_cols:
        raise ValueError(
            f"The code expression {code_template_str!r} is a literal: it references no source "
            "columns, so a literal code has no components to match metadata on. Add static "
            "metadata for literal codes to a pre-existing codes.parquet instead of a _metadata "
            "block."
        )

    metadata_cfg = event_cfg["_metadata"]
    match_on = metadata_cfg.get("_match_on")
    if match_on is None:
        return sorted(referenced_cols)

    if isinstance(match_on, str):
        match_on = [match_on]
    match_on = list(match_on)

    invalid_match_cols = sorted(set(match_on) - referenced_cols)
    if invalid_match_cols:
        raise KeyError(
            f"_match_on columns {invalid_match_cols} are not referenced by the code expression "
            f"'{code_template_str}'. Valid columns: {sorted(referenced_cols)}"
        )

    aliased = sorted(set(match_on) & {k for k in metadata_cfg if k != "_match_on"})
    if aliased:
        raise ValueError(
            f"Match column(s) {aliased} may not also be declared as _metadata output "
            "expressions. Join keys must be raw metadata columns whose names match the "
            "code components they join against."
        )

    return match_on


def extract_metadata(
    metadata_df: pl.LazyFrame | pl.DataFrame, event_cfg: dict[str, Any]
) -> pl.LazyFrame | pl.DataFrame:
    """Extracts a single metadata dataframe block for an event configuration from the raw metadata.

    Every metadata join is a component join, so this function never assembles a ``code``
    column. It emits the entry's match-key columns verbatim (all code-referenced columns
    by default, or the ``_match_on`` narrowing — see ``resolve_match_columns``), a
    ``code_template`` provenance column, and the configured metadata output columns. The
    reducer joins the keys against the observed ``code_components`` map — scoped to the
    declaring event, with null keys matching null components — to attach the metadata to
    full codes.

    Args:
        metadata_df: The raw metadata frame (lazy or eager; the output matches the input's
            laziness). Mandatory columns are determined by the `event_cfg` configuration dictionary.
        event_cfg: A dictionary containing the configuration for the event. Not mutated (values may
            be nested `_metadata` mappings, not just strings). This must contain the critical
            `"code"` key alongside a mandatory `_metadata` block, which must contain some columns that should
            be extracted from the metadata to link to the code.
            The `"code"`` value is a dftly expression: string literals must be quoted
            (e.g., ``'"MY_CODE"'``), column references use ``$`` prefix (e.g., ``$col``),
            and interpolation uses f-strings (e.g., ``f"PREFIX//{$col}"``).

    Returns:
        A DataFrame containing the match-key columns, the ``code_template`` column, and whatever metadata
        columns are specified for extraction in the metadata block. The output dataframe will not
        necessarily be unique by key if the input metadata is not unique by key.

    Raises:
        KeyError: If the event configuration dictionary is missing the `"code"` or `"_metadata"` keys, or if
            columns referenced by the event configuration dictionary are not found in the raw metadata.
        ValueError: If the code expression is a literal (nothing to match metadata on), or if a reserved or
            match-column name is redefined as a metadata output.
        TypeError: If the event configuration is not a dictionary.

    Examples:
        >>> extract_metadata(pl.DataFrame(), {})
        Traceback (most recent call last):
            ...
        KeyError: "Event configuration dictionary must contain 'code' key. Got: []."
        >>> extract_metadata(pl.DataFrame(), {"code": "$test"})
        Traceback (most recent call last):
            ...
        KeyError: "Event configuration dictionary must contain a non-empty '_metadata' key. Got: [code]."

        By default an entry matches on every code-referenced column, in sorted order. The
        output carries the raw key columns (never an assembled ``code``), the template,
        and the metadata outputs. Rows whose outputs are all null are dropped; rows with
        null *keys* are retained — the reducer's component join matches them against null
        components:

        >>> raw_metadata = pl.DataFrame({
        ...     "itemid": ["A", "B", "C", None],
        ...     "modifier": ["1", "2", "3", "4"],
        ...     "name": ["Code A-1", "B-2", None, "Null-key row"],
        ... })
        >>> event_cfg = {
        ...     "code": 'f"FOO//{$itemid}//{$modifier}"',
        ...     "_metadata": {"desc": "name"},
        ... }
        >>> extract_metadata(raw_metadata, event_cfg)
        shape: (3, 4)
        ┌────────┬──────────┬────────────────────────────────┬──────────────┐
        │ itemid ┆ modifier ┆ code_template                  ┆ desc         │
        │ ---    ┆ ---      ┆ ---                            ┆ ---          │
        │ str    ┆ str      ┆ str                            ┆ str          │
        ╞════════╪══════════╪════════════════════════════════╪══════════════╡
        │ A      ┆ 1        ┆ f"FOO//{$itemid}//{$modifier}" ┆ Code A-1     │
        │ B      ┆ 2        ┆ f"FOO//{$itemid}//{$modifier}" ┆ B-2          │
        │ null   ┆ 4        ┆ f"FOO//{$itemid}//{$modifier}" ┆ Null-key row │
        └────────┴──────────┴────────────────────────────────┴──────────────┘

        Match columns must exist in the metadata table:

        >>> extract_metadata(raw_metadata.drop("modifier"), event_cfg)
        Traceback (most recent call last):
            ...
        KeyError: "Match column(s) ['modifier'] not found in metadata columns: ['itemid', 'name']"

        ...and referencing a metadata column that doesn't exist in the table raises, naming the
        missing column and the columns that were available:

        >>> extract_metadata(
        ...     raw_metadata,
        ...     {"code": 'f"FOO//{$itemid}//{$modifier}"', "_metadata": {"desc": "nonexistent_col"}},
        ... )
        Traceback (most recent call last):
            ...
        KeyError: "Columns {'nonexistent_col'} not found in metadata columns: ['itemid', 'modifier', 'name']"

        A ``_metadata`` block on a literal code is rejected — there are no components to
        match metadata on:

        >>> extract_metadata(raw_metadata, {"code": "MEDS_BIRTH", "_metadata": {"desc": "name"}})
        Traceback (most recent call last):
            ...
        ValueError: The code expression 'MEDS_BIRTH' is a literal: it references no source columns, ...

        ``_match_on`` narrows the key set to a subset of the code-referenced columns; the
        metadata table then only needs the named columns. Multi-column ``_match_on`` keys
        are all carried through, ahead of the metadata outputs:

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

        A match column may not also be a ``_metadata`` output expression — join keys are
        raw metadata columns (key sourcing/renaming belongs to the planned dftly metadata
        block):

        >>> extract_metadata(
        ...     pl.DataFrame({"itemid_alias": ["220045"], "label": ["Heart Rate"]}),
        ...     {
        ...         "code": 'f"CHART//{$itemid}"',
        ...         "_metadata": {"_match_on": "itemid", "itemid": "itemid_alias", "description": "label"},
        ...     },
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Match column(s) ['itemid'] may not also be declared as _metadata output ...

        ``_match_on`` validation errors name the offending column. A ``_match_on`` column must be
        referenced by the code expression:

        >>> extract_metadata(
        ...     pl.DataFrame({"medication_name": ["X"], "desc": ["Y"]}),
        ...     {"code": "$medication_name",
        ...      "_metadata": {"_match_on": "typo_column", "description": "desc"}},
        ... )
        Traceback (most recent call last):
            ...
        KeyError: "_match_on columns ['typo_column'] are not referenced by the code expression
        '$medication_name'. Valid columns: ['medication_name']"

        ...must exist in the metadata table:

        >>> extract_metadata(
        ...     pl.DataFrame({"dose": ["500mg"], "desc": ["some desc"]}),
        ...     {"code": 'f"{$medication_name}//{$dose}"',
        ...      "_metadata": {"_match_on": "medication_name", "description": "desc"}},
        ... )
        Traceback (most recent call last):
            ...
        KeyError: "Match column(s) ['medication_name'] not found in metadata columns: ['dose', 'desc']"

        ...and metadata output columns must exist even under a ``_match_on`` narrowing:

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
    and mandatory MEDS metadata columns will be cast to the correct types. Note that the ``code`` column
    below carries the raw source values of the component literally named ``code`` (the idiomatic ICD/OMOP
    vocabulary shape) — it is a join key, not the assembled MEDS code, which never appears in mapper output:
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
        >>> extract_metadata(raw_metadata, event_cfg)
        shape: (4, 5)
        ┌──────┬───────────────┬─────────────────────────────────┬─────────────┬─────────────────────┐
        │ code ┆ code_modifier ┆ code_template                   ┆ description ┆ parent_codes        │
        │ ---  ┆ ---           ┆ ---                             ┆ ---         ┆ ---                 │
        │ str  ┆ str           ┆ str                             ┆ str         ┆ list[str]           │
        ╞══════╪═══════════════╪═════════════════════════════════╪═════════════╪═════════════════════╡
        │ A    ┆ 1             ┆ f"FOO//{$code}//{$code_modifie… ┆ used        ┆ null                │
        │ A    ┆ 1             ┆ f"FOO//{$code}//{$code_modifie… ┆ A-1-2       ┆ ["OUT_VAL/1/2"]     │
        │ C    ┆ 2             ┆ f"FOO//{$code}//{$code_modifie… ┆ C-2-3       ┆ ["OUT_VAL_for_3/2"] │
        │ D    ┆ 3             ┆ f"FOO//{$code}//{$code_modifie… ┆ null        ┆ ["expanded form"]   │
        └──────┴───────────────┴─────────────────────────────────┴─────────────┴─────────────────────┘
    """
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

    match_cols = resolve_match_columns(event_cfg)
    _, code_template_str = _parse_code(event_cfg["code"])

    metadata_cfg = dict(event_cfg["_metadata"])
    metadata_cfg.pop("_match_on", None)

    # ``code`` and ``code_template`` are pipeline-generated output columns with mandated
    # meanings; a ``_metadata`` block may not redefine them.
    reserved = sorted({"code", "code_template"} & set(metadata_cfg))
    if reserved:
        raise ValueError(
            f"_metadata output column name(s) {reserved} are reserved: 'code' and "
            "'code_template' are generated by the pipeline and cannot be overwritten."
        )

    df_select_exprs = {}
    final_cols = []
    needed_cols = set()
    for out_col, in_cfg in metadata_cfg.items():
        in_expr, needed = cfg_to_expr(in_cfg)
        df_select_exprs[out_col] = in_expr
        final_cols.append(out_col)
        needed_cols.update(needed)

    columns = metadata_df.collect_schema().names()

    missing_match_cols = sorted(set(match_cols) - set(columns))
    if missing_match_cols:
        raise KeyError(f"Match column(s) {missing_match_cols} not found in metadata columns: {columns}")

    missing_metadata_cols = needed_cols - set(columns) - set(final_cols)
    if missing_metadata_cols:
        raise KeyError(f"Columns {missing_metadata_cols} not found in metadata columns: {columns}")

    for col in match_cols:
        if col not in df_select_exprs:
            df_select_exprs[col] = pl.col(col)

    metadata_df = metadata_df.select(**df_select_exprs).with_columns(
        code_template=pl.lit(code_template_str),
    )

    metadata_df = metadata_df.filter(~pl.all_horizontal(*[pl.col(c).is_null() for c in final_cols]))

    for mandatory_col, mandatory_type in MEDS_METADATA_MANDATORY_TYPES.items():
        if mandatory_col not in final_cols:
            continue

        if metadata_df.collect_schema()[mandatory_col] != mandatory_type:
            logger.warning(f"Metadata column '{mandatory_col}' must be of type {mandatory_type}. Casting.")
            metadata_df = metadata_df.with_columns(pl.col(mandatory_col).cast(mandatory_type, strict=False))

    return metadata_df.unique(maintain_order=True).select(*match_cols, "code_template", *final_cols)


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
    ``tests/test_extract_code_metadata.py`` can call it directly — that
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

    Metadata is attached to codes through one join path: each ``_metadata`` entry's match
    columns (all code-referenced columns by default, or the ``_match_on`` narrowing) are
    joined against the observed ``code_components`` map, scoped to the declaring event
    block, with null keys matching null components. Because that map is built from the
    observed data, the join inherently restricts the output to observed codes.

    Note that there are two sentinel columns in the output metadata that have certain mandates for MEDS
    compliance: The `description` column and the `parent_codes` column. The `description` column must be a
    string, and if there are multiple matches in the extracted metadata for a code, in this script they will
    be concatenated into a single string with the `description_separator` string. The `parent_codes` column
    must be a list of strings, each formatted as an OMOP vocabulary name, followed by a "/", followed by the
    OMOP concept code. This column is used to link codes to their parent codes in the OMOP vocabulary.

    The reduced output has a canonical, data-independent shape: every extracted metadata column
    other than `description` and `code_template` is aggregated per code to a `List(String)` of
    distinct values, sorted within each group (a single-source unique code yields a one-element
    list). `description` joins its distinct values with `description_separator` in canonical
    config order. `code_template` is a plain `String` — one code has exactly one template, and
    distinct templates colliding on a single code is a configuration error. Missing values are
    null, never an empty list or empty string.

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

    # Load the extracted event data, handling heterogeneous schemas (files whose codes
    # reference source columns carry code_components; all-literal files don't).
    event_parquet_files = list(Path(stage_input_dir).rglob("*.parquet"))
    all_event_dfs = [pl.scan_parquet(fp, glob=False) for fp in event_parquet_files]
    all_data = pl.concat(all_event_dfs, how="diagonal_relaxed")

    # Schema validation runs in EVERY worker (it's a cheap metadata-only check) so a
    # pre-0.7 events layout fails loudly everywhere — but the component map itself is
    # only materialized by the reducer (worker 0) below: it is a full-dataset
    # scan/unique/collect that the N-1 map-only workers never use.
    has_code_components = validate_event_data_schema(all_data.collect_schema())

    all_out_fps = []
    # Deterministic reduction order: partial files are produced in each worker's
    # shuffled config order, but the reduction must not depend on which worker shuffled how.
    # Record the canonical (metadata_prefix, cfg_idx) key for every output file so the
    # reducer can sort frames back into config order before concatenating.
    out_fp_keys: dict[Path, tuple[str, int]] = {}
    # Explicit join bookkeeping: out_fp -> (match_cols, source_block). The reducer is
    # driven by this record rather than by sniffing output schemas — a match column named
    # "code" (the idiomatic ICD/OMOP vocabulary shape) carries raw component values that
    # schema sniffing could mistake for assembled output codes.
    join_info: dict[Path, tuple[list[str], str]] = {}
    for input_prefix, event_metadata_cfgs in event_metadata_configs:
        event_metadata_cfgs = copy.deepcopy(event_metadata_cfgs)

        metadata_fps = resolve_source_files(raw_input_dir, input_prefix)

        # Reader kwargs are chosen per resolved prefix: csv-family sources read with
        # ``infer_schema=False`` for a uniform all-String schema, while parquet sources
        # keep their intrinsic types (``infer_schema`` is csv-only and would crash
        # ``scan_parquet``). Deciding from the first file is sound because
        # ``scan_source`` enforces format homogeneity across a multi-file source.
        read_kwargs = {} if _format_family(metadata_fps[0]) == "parquet" else {"infer_schema": False}

        def read_fn(fps, read_kwargs=read_kwargs):
            return scan_source(fps, **read_kwargs)

        # Write one output file per individual event config: entries sharing a metadata
        # prefix can join on different match columns and are scoped to different declaring
        # events.
        for cfg_idx, event_cfg in enumerate(event_metadata_cfgs):
            out_fp = partial_metadata_dir / f"{input_prefix}_{cfg_idx}.parquet"
            logger.info(f"Extracting metadata from {metadata_fps} and saving to {out_fp}")

            # Always present: ``events_by_metadata_prefix`` stamps every entry (see
            # SOURCE_BLOCK_COL in config.py); a KeyError here means that contract broke.
            source_block = event_cfg.pop(SOURCE_BLOCK_COL)

            # Derive and validate the join keys up front so configuration errors (a
            # literal code with a _metadata block, a bad _match_on) surface in every
            # worker even when the output shard already exists and the compute is skipped.
            match_cols = resolve_match_columns(event_cfg)

            rwlock_wrap(
                metadata_fps,
                out_fp,
                read_fn,
                atomic_write_parquet,
                partial(extract_metadata, event_cfg=event_cfg),
                do_overwrite=cfg.do_overwrite,
            )
            all_out_fps.append(out_fp)
            out_fp_keys[out_fp] = (input_prefix, cfg_idx)
            join_info[out_fp] = (match_cols, source_block)

    logger.info("Extracted metadata for all events. Merging.")

    if cfg.worker != 0:
        logger.info("Code metadata extraction completed. Exiting")
        return

    logger.info("Starting reduction process")

    # Build the code_components map every metadata join runs against: full code (under
    # the reserved collision-proof alias), unnested component columns, and the declaring
    # source_block so each join attaches only to its own event's codes. Reducer-only:
    # this is the one full-dataset collect in the stage.
    code_component_map = build_code_component_map(all_data) if has_code_components else None

    wait_for_complete_parquets(all_out_fps, polling_time=cfg.polling_time)

    start = datetime.now(tz=UTC)
    logger.info("All map shards complete! Starting code metadata reduction computation.")

    # Expand every metadata file to full codes through one component join, scoped to the
    # declaring event and keyed on the recorded match columns.
    #
    # Frames are processed in canonical (metadata_prefix, cfg_idx) order so the reduction
    # (concat order, and with it description join order and list-aggregation order) is
    # identical across runs. This sort happens ONLY here, in worker 0's reduction over
    # already-written partial files — the mappers' per-worker random.shuffle above is
    # untouched, so map-phase lock contention and runtime spreading are unaffected.
    expanded_dfs = []
    if code_component_map is None:
        logger.warning(
            "Extracted metadata found but the event data carries no code_components; "
            "there is nothing to join metadata onto. Writing an empty metadata table."
        )
    else:
        component_schema = code_component_map.schema
        for fp in sorted(all_out_fps, key=out_fp_keys.__getitem__):
            pdf = pl.scan_parquet(fp, glob=False)
            match_cols, source_block = join_info[fp]
            pdf_schema = pdf.collect_schema()
            metadata_cols = [c for c in pdf_schema.names() if c not in match_cols]

            missing = [c for c in match_cols if c not in component_schema]
            if missing:
                logger.warning(
                    f"Metadata from {fp} (source block {source_block!r}) requires component "
                    f"columns {missing} that are absent from the extracted data. Skipping."
                )
                continue

            # Scope the join to the event config that declared this _metadata block —
            # other events may reference same-named component columns with colliding
            # values, and must not receive this metadata.
            components = code_component_map.lazy().filter(pl.col(SOURCE_BLOCK_COL) == source_block)

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

            # ``nulls_equal=True`` is a deliberate semantic choice: a null join key on the
            # metadata side matches exactly the data rows whose (``??``-coalesced)
            # component is null — e.g. a vocabulary row keyed on a null unit attaches to
            # the code that a ``{$valueuom ?? 'UNK'}`` component rendered for unit-less
            # data rows.
            expanded = (
                components.join(pdf, on=match_cols, how="inner", nulls_equal=True)
                .select(pl.col(FULL_CODE_COL).alias("code"), *metadata_cols)
                .collect()
            )
            if expanded.is_empty():
                logger.warning(
                    f"Metadata from {fp} (source block {source_block!r}, "
                    f"match columns {match_cols}) matched zero codes."
                )
            expanded_dfs.append(expanded.lazy())

    if not expanded_dfs:
        logger.info("No metadata to reduce. Writing empty metadata file.")
        reduced = pl.DataFrame({"code": []}).cast({"code": pl.String}).lazy()
    else:
        reduced = pl.concat(expanded_dfs, how="diagonal_relaxed").unique(maintain_order=True)

    # The reduction is keyed on the assembled MEDS code alone: component-level
    # narrowing already happened in the expansion join above, so by this point
    # every metadata row is fully resolved to a full code.
    join_cols = ["code"]
    reduced_cols = reduced.collect_schema().names()
    metadata_cols = [c for c in reduced_cols if c not in join_cols]

    n_unique_obs = reduced.select(pl.n_unique(*join_cols)).collect().item()
    n_rows = reduced.select(pl.len()).collect().item()
    logger.info(f"Collected metadata for {n_unique_obs} unique codes among {n_rows} total observations.")

    # Aggregation is UNCONDITIONAL so the output schema never depends on whether some code
    # happened to be duplicated across metadata rows. Canonical output shape (values are
    # deduplicated everywhere — repeating identical metadata per code is pure waste):
    #   - ``description``: String, distinct non-null values joined with
    #     ``description_separator`` in canonical config order.
    #   - ``parent_codes``: List(String), flattened across sources, nulls dropped,
    #     deduplicated in first-seen order.
    #   - ``code_template``: String. One code has exactly one template; distinct templates
    #     colliding on one code is a config error and raises below.
    #   - every other metadata column: List(String), nulls dropped, distinct values sorted
    #     within each group so multi-source aggregation is order-stable.
    # A code with no value for a column gets null (never an empty list / empty string).
    reduced_schema = reduced.collect_schema()
    aggs = {}
    for c in metadata_cols:
        if c == CodeMetadataSchema.description_name:
            aggs[c] = pl.col(c).drop_nulls().unique(maintain_order=True)
        elif c == CodeMetadataSchema.parent_codes_name:
            aggs[c] = pl.col(c).explode().drop_nulls().unique(maintain_order=True)
        elif c == "code_template":
            aggs[c] = pl.col(c).drop_nulls().unique(maintain_order=True)
        elif isinstance(reduced_schema[c], pl.List):
            aggs[c] = pl.col(c).explode().cast(pl.String).drop_nulls().unique().sort()
        else:
            aggs[c] = pl.col(c).cast(pl.String).drop_nulls().unique().sort()

    reduced = reduced.group_by(join_cols, maintain_order=True).agg(**aggs)

    if "code_template" in metadata_cols:
        conflicted = (
            reduced.filter(pl.col("code_template").list.len() > 1)
            .select(join_cols[0], "code_template")
            .head(5)
            .collect()
        )
        if conflicted.height:
            raise ValueError(
                "One code maps to multiple distinct code templates — each code must be "
                "produced by exactly one template. Conflicts (first 5): "
                f"{conflicted.rows()}"
            )
        reduced = reduced.with_columns(pl.col("code_template").list.first())

    empty_to_null = []
    for c in metadata_cols:
        if c == "code_template":
            # Already a scalar String: ``.list.first()`` above yields null for a code with
            # no template, so there is no empty-list state left to normalize.
            continue
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
        # (The bare full join used to silently fork overlaps into ``*_right`` columns.)
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
