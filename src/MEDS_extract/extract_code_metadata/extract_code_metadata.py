"""Utilities for extracting code metadata about the codes produced for the MEDS events."""

import copy
import logging
import random
import time
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import polars as pl
from dftly import Parser
from meds import CodeMetadataSchema
from MEDS_transforms.mapreduce.rwlock import is_complete_parquet_file, run_marker_dir, rwlock_wrap
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig
from upath import UPath

from .._stage_example import MEDSExtractStageExample
from ..config import (
    METADATA_COMPONENTS_COL,
    SELF_METADATA_PREFIX,
    SOURCE_BLOCK_COL,
    CompiledMetadataBlock,
    MessyConfig,
    compile_metadata_block,
    compile_self_metadata_block,
)
from ..io import _format_family, resolve_source_files, scan_source

logger = logging.getLogger(__name__)

# On-disk size above which an external metadata table draws a WARNING: metadata tables
# are fully materialized by the mapper and joined against every observed code, so they
# are sized for vocabularies, not data tables. A larger source is almost always an event
# table pointed at itself (the ``_self`` prefix is the fix) or a mis-typed prefix.
METADATA_TABLE_WARN_BYTES = 256 * 2**20

# The MEDS-mandated dtypes of the sentinel code-metadata columns, derived from the
# authoritative ``meds.CodeMetadataSchema`` (a pyarrow schema) rather than hand-written:
# an empty-table round-trip through ``pl.from_arrow`` performs the pyarrow -> polars
# dtype mapping, so a MEDS schema change propagates here automatically.
MEDS_METADATA_MANDATORY_TYPES: dict[str, pl.DataType] = dict(
    pl.from_arrow(CodeMetadataSchema.schema().empty_table()).schema
)

# Mapper-level mandated dtypes for the MEDS-sentinel columns a ``_metadata`` block can
# produce — the final MEDS dtypes above, with one deliberate divergence:
# ``parent_codes``' *final* shape is ``List(String)``, but a dftly expression yields at
# most one parent per metadata ROW, so the mapper emits a nullable scalar String and
# the reducer aggregates the per-row scalars into the canonical per-code
# ``List(String)``.
_MAPPER_MANDATORY_TYPES: dict[str, pl.DataType] = {
    **MEDS_METADATA_MANDATORY_TYPES,
    CodeMetadataSchema.parent_codes_name: pl.String,
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
    """Validate one extracted-event file's schema; return whether that file carries components.

    ``code_components`` is only attached by ``EventConfig.extract`` when a code expression
    references at least one source column — a file whose codes are all literals
    legitimately has no components (and therefore nothing to join metadata onto), so its
    absence is allowed and reported as ``False``. This check is applied per event file
    (see :func:`union_component_fields`): one file's components must never mask another
    file's malformed pairing.

    When components ARE present, ``source_block`` must be too: ``EventConfig.extract``
    stamps both on every row together, so an input carrying one without the other is
    malformed — and without ``source_block``, metadata joins cannot be scoped to the
    event that declared them (one event's metadata would silently attach to other
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
            "column. convert_to_MEDS_events stamps both together, so this input is not a "
            "valid extracted-events directory; re-run the extraction pipeline before "
            "extracting code metadata."
        )
    return True


def union_component_fields(data_schemas: Iterable[pl.Schema]) -> set[str]:
    """Union the ``code_components`` struct fields across per-file extracted-event schemas.

    Every metadata join in the reducer runs against component columns, so the set of
    component columns the stage may join on is decided here — as the UNION over every
    event file's ``code_components`` struct fields. Files are validated individually
    through :func:`validate_event_data_schema`, so the decision cannot depend on which
    file a directory enumeration yields first, and a malformed file (components without
    ``source_block``) raises even when other files are well-formed. An empty set means no
    file carries components.

    Examples:
        Struct fields union across heterogeneous files; all-literal files (no
        ``code_components`` column) contribute nothing:

        >>> sorted(union_component_fields([
        ...     pl.Schema({
        ...         "code": pl.String,
        ...         "code_components": pl.Struct({"itemid": pl.Int64}),
        ...         "source_block": pl.String,
        ...     }),
        ...     pl.Schema({
        ...         "code": pl.String,
        ...         "code_components": pl.Struct({"icd_code": pl.String, "icd_version": pl.String}),
        ...         "source_block": pl.String,
        ...     }),
        ...     pl.Schema({"code": pl.String, "source_block": pl.String}),
        ... ]))
        ['icd_code', 'icd_version', 'itemid']

        No file carrying components yields the empty set:

        >>> union_component_fields([pl.Schema({"code": pl.String})])
        set()

        A malformed file raises even when a sibling file is well-formed:

        >>> union_component_fields([
        ...     pl.Schema({
        ...         "code": pl.String,
        ...         "code_components": pl.Struct({"itemid": pl.Int64}),
        ...         "source_block": pl.String,
        ...     }),
        ...     pl.Schema({"code": pl.String, "code_components": pl.Struct({"itemid": pl.Int64})}),
        ... ])
        Traceback (most recent call last):
            ...
        ValueError: Extracted event data carries 'code_components' but no 'source_block' column. ...
    """
    fields: set[str] = set()
    for schema in data_schemas:
        if validate_event_data_schema(schema):
            fields.update(f.name for f in schema["code_components"].fields)
    return fields


def _compile_metadata_entry(event_cfg: Mapping, *, self_block: bool = False) -> CompiledMetadataBlock:
    """Compile one ``{code, _metadata}`` entry: parse the code, compile+validate the block.

    ``self_block=True`` compiles a ``_self`` entry (implicit keys, outputs evaluated at
    event-extraction time) via ``compile_self_metadata_block``; the shape checks are shared.

    The single seam between the entry dicts ``MessyConfig.events_by_metadata_prefix``
    emits and :func:`MEDS_extract.config.compile_metadata_block`. ``main`` compiles
    each entry exactly once here — deriving the join-key bookkeeping from
    ``key_cols`` and handing the compiled block to :func:`extract_metadata` — so
    nothing downstream ever re-parses the config. ``code`` is always the raw dftly
    expression string: ``events_by_metadata_prefix`` guarantees it (a metadata-carrying
    event must retain its raw code string — enforced at ``EventConfig`` construction —
    because that string is stamped verbatim as ``code_template``).

    Examples:
        >>> compiled = _compile_metadata_entry({
        ...     "code": 'f"CHART//{$itemid}"',
        ...     "_metadata": {"itemid": "$itemid", "description": "$label"},
        ... })
        >>> compiled.key_cols, compiled.output_cols, compiled.code_template
        (('itemid',), ('description',), 'f"CHART//{$itemid}"')

        Entry-shape mistakes are caught before any compilation:

        >>> _compile_metadata_entry(["foo"])
        Traceback (most recent call last):
            ...
        TypeError: Event configuration must be a dictionary. Got: <class 'list'> ['foo'].
        >>> _compile_metadata_entry({})
        Traceback (most recent call last):
            ...
        KeyError: "Event configuration dictionary must contain 'code' key. Got: []."
        >>> _compile_metadata_entry({"code": "$test"})
        Traceback (most recent call last):
            ...
        KeyError: "Event configuration dictionary must contain a non-empty '_metadata' key. Got: [code]."
        >>> from dftly import Parser
        >>> _compile_metadata_entry({"code": Parser()("$itemid"), "_metadata": {"itemid": "$itemid"}})
        Traceback (most recent call last):
            ...
        TypeError: Entry 'code' must be the raw dftly expression string, got Column. ...
    """
    if not isinstance(event_cfg, Mapping):
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

    code_template_str = event_cfg["code"]
    if not isinstance(code_template_str, str):
        raise TypeError(
            f"Entry 'code' must be the raw dftly expression string, got "
            f"{type(code_template_str).__name__}. Metadata-carrying events always retain the raw "
            f"code string (enforced at EventConfig construction) — it is stamped verbatim as "
            f"'code_template'."
        )
    code_node = Parser()(code_template_str)

    compile_fn = compile_self_metadata_block if self_block else compile_metadata_block
    return compile_fn(
        event_cfg["_metadata"],
        frozenset(code_node.referenced_columns),
        code_template_str=code_template_str,
    )


def extract_metadata(
    metadata_df: pl.LazyFrame | pl.DataFrame, compiled: CompiledMetadataBlock
) -> pl.LazyFrame | pl.DataFrame:
    """Evaluate one compiled ``_metadata`` block over the raw metadata table.

    A ``_metadata`` entry is a small dftly program over the raw metadata table; it is
    compiled and validated up front by
    :func:`MEDS_extract.config.compile_metadata_block` (every config-shape error — a
    literal code, zero join keys, reserved names — fires
    there, at config load and again in ``main``'s per-entry loop, never here). This
    function is the pure mapper compute: it evaluates the compiled expressions, emits
    the join-key columns, a ``code_template`` provenance column, and the metadata
    outputs — it never assembles a ``code`` column. The reducer joins the keys against
    the observed ``code_components`` map — scoped to the declaring event, with null
    keys matching null components — to attach the metadata to full codes.

    Args:
        metadata_df: The raw metadata frame (lazy or eager; the output matches the
            input's laziness). It must carry every column the compiled expressions
            reference — the one check performed here, since only this function sees
            the actual table.
        compiled: The compiled ``_metadata`` block, from
            :func:`MEDS_extract.config.compile_metadata_block` (or ``main``'s
            per-entry :func:`_compile_metadata_entry` seam).

    Returns:
        A frame containing the join-key columns (sorted), the ``code_template``
        column, and the metadata output columns in declared order. The output will
        not necessarily be unique by key if the input metadata is not unique by key.

    Raises:
        ValueError: If the compiled expressions reference columns absent from the
            metadata table.

    Examples:
        Producing every component column is a full match. The output carries the key
        columns in sorted order (never an assembled ``code``), the template, and the
        metadata outputs in declared order. Rows whose outputs are all null are
        dropped; rows with null *keys* are retained — the reducer's component join
        matches them against null components:

        >>> raw_metadata = pl.DataFrame({
        ...     "itemid": ["A", "B", "C", None],
        ...     "modifier": ["1", "2", "3", "4"],
        ...     "name": ["Code A-1", "B-2", None, "Null-key row"],
        ... })
        >>> compiled = compile_metadata_block(
        ...     {"itemid": "$itemid", "modifier": "$modifier", "desc": "$name"},
        ...     {"itemid", "modifier"},
        ...     code_template_str='f"FOO//{$itemid}//{$modifier}"',
        ... )
        >>> extract_metadata(raw_metadata, compiled)
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

        A bare, unquoted string is a dftly string LITERAL, not a column reference —
        ``desc: name`` stamps the constant text ``"name"`` on every row (the pre-0.7
        shorthand where it read the ``name`` column is gone; write ``$name``):

        >>> literal_block = compile_metadata_block(
        ...     {"itemid": "$itemid", "modifier": "$modifier", "desc": "name"},
        ...     {"itemid", "modifier"},
        ...     code_template_str='f"FOO//{$itemid}//{$modifier}"',
        ... )
        >>> extract_metadata(raw_metadata, literal_block)["desc"].unique().to_list()
        ['name']

        Because keys are expressions too, sourcing a join key from a differently-named
        metadata column is just a rename expression — here the ``itemid`` key comes
        from the metadata table's ``omop_source_code`` column, with a full-dftly
        literal alongside:

        >>> compiled = compile_metadata_block(
        ...     {"itemid": "$omop_source_code", "description": "$label", "vocab": '"MIMIC-IV"'},
        ...     {"itemid"},
        ...     code_template_str='f"CHART//{$itemid}"',
        ... )
        >>> extract_metadata(
        ...     pl.DataFrame({"omop_source_code": ["220045"], "label": ["Heart Rate"]}), compiled
        ... )
        shape: (1, 4)
        ┌────────┬─────────────────────┬─────────────┬──────────┐
        │ itemid ┆ code_template       ┆ description ┆ vocab    │
        │ ---    ┆ ---                 ┆ ---         ┆ ---      │
        │ str    ┆ str                 ┆ str         ┆ str      │
        ╞════════╪═════════════════════╪═════════════╪══════════╡
        │ 220045 ┆ f"CHART//{$itemid}" ┆ Heart Rate  ┆ MIMIC-IV │
        └────────┴─────────────────────┴─────────────┴──────────┘

        Producing a *subset* of the component columns is a partial match: the metadata
        table only needs the produced keys, and the reducer broadcasts the metadata to
        every code sharing them:

        >>> compiled = compile_metadata_block(
        ...     {"a": "$a", "b": "$b", "description": "$desc"},
        ...     {"a", "b", "c"},
        ...     code_template_str='f"{$a}//{$b}//{$c}"',
        ... )
        >>> extract_metadata(
        ...     pl.DataFrame({"a": ["X", "Y"], "b": ["1", "2"], "desc": ["X-1", "Y-2"]}), compiled
        ... )
        shape: (2, 4)
        ┌─────┬─────┬─────────────────────┬─────────────┐
        │ a   ┆ b   ┆ code_template       ┆ description │
        │ --- ┆ --- ┆ ---                 ┆ ---         │
        │ str ┆ str ┆ str                 ┆ str         │
        ╞═════╪═════╪═════════════════════╪═════════════╡
        │ X   ┆ 1   ┆ f"{$a}//{$b}//{$c}" ┆ X-1         │
        │ Y   ┆ 2   ┆ f"{$a}//{$b}//{$c}" ┆ Y-2         │
        └─────┴─────┴─────────────────────┴─────────────┘

        Key normalization is user-visible dftly — casts and ``??`` coalescing shape
        the key values the join will run against:

        >>> compiled = compile_metadata_block(
        ...     {"itemid": "$itemid::str", "description": "$label ?? $alt_label"},
        ...     {"itemid"},
        ...     code_template_str='f"CHART//{$itemid}"',
        ... )
        >>> extract_metadata(
        ...     pl.DataFrame({"itemid": [220045], "alt_label": ["HR alt"], "label": [None]}), compiled
        ... )
        shape: (1, 3)
        ┌────────┬─────────────────────┬─────────────┐
        │ itemid ┆ code_template       ┆ description │
        │ ---    ┆ ---                 ┆ ---         │
        │ str    ┆ str                 ┆ str         │
        ╞════════╪═════════════════════╪═════════════╡
        │ 220045 ┆ f"CHART//{$itemid}" ┆ HR alt      │
        └────────┴─────────────────────┴─────────────┘

        Referencing a metadata column that doesn't exist in the table raises, naming
        the missing column(s) and the columns that were available:

        >>> compiled = compile_metadata_block(
        ...     {"itemid": "$itemid", "desc": "$nonexistent_col"},
        ...     {"itemid"},
        ...     code_template_str='f"CHART//{$itemid}"',
        ... )
        >>> extract_metadata(pl.DataFrame({"itemid": ["A"], "name": ["x"]}), compiled)
        Traceback (most recent call last):
            ...
        ValueError: _metadata block for code 'f"CHART//{$itemid}"' references column(s)
        ['nonexistent_col'] not present in the metadata table. Available columns:
        ['itemid', 'name'].

    Mandatory MEDS metadata columns are cast to the mapper's mandated types, and ``parent_codes``
    is an ordinary dftly output: the old multi-case matcher list is a chained conditional whose
    omitted final ``else`` yields a real null for unmatched rows. Each metadata row produces at
    most ONE parent (a nullable String); the reducer unions parents across rows/sources into the
    canonical per-code ``List(String)``. Note that the ``code`` column below carries the raw
    source values of the component literally named ``code`` (the idiomatic ICD/OMOP vocabulary
    shape) — it is a join key (allowed precisely because it names a component), not the assembled
    MEDS code, which never appears in mapper output:
        >>> raw_metadata = pl.DataFrame({
        ...     "code": ["A", "A", "C", "D"],
        ...     "code_modifier": ["1", "1", "2", "3"],
        ...     "code_modifier_2": ["1", "2", "3", "4"],
        ...     "title": ["A-1-1", "A-1-2", "C-2-3", None],
        ...     "special_title": ["used", None, None, None],
        ... })
        >>> compiled = compile_metadata_block(
        ...     {
        ...         "code": "$code",
        ...         "code_modifier": "$code_modifier",
        ...         "description": "coalesce($special_title, $title)",
        ...         "parent_codes": (
        ...             'f"OUT_VAL/{$code_modifier}/2" if $code_modifier_2 == "2"'
        ...             ' else f"OUT_VAL_for_3/{$code_modifier}" if $code_modifier_2 == "3"'
        ...             ' else "expanded form" if $code_modifier_2 == "4"'
        ...         ),
        ...     },
        ...     {"code", "code_modifier"},
        ...     code_template_str='f"FOO//{$code}//{$code_modifier}"',
        ... )
        >>> extract_metadata(raw_metadata, compiled)
        shape: (4, 5)
        ┌──────┬───────────────┬─────────────────────────────────┬─────────────┬─────────────────┐
        │ code ┆ code_modifier ┆ code_template                   ┆ description ┆ parent_codes    │
        │ ---  ┆ ---           ┆ ---                             ┆ ---         ┆ ---             │
        │ str  ┆ str           ┆ str                             ┆ str         ┆ str             │
        ╞══════╪═══════════════╪═════════════════════════════════╪═════════════╪═════════════════╡
        │ A    ┆ 1             ┆ f"FOO//{$code}//{$code_modifie… ┆ used        ┆ null            │
        │ A    ┆ 1             ┆ f"FOO//{$code}//{$code_modifie… ┆ A-1-2       ┆ OUT_VAL/1/2     │
        │ C    ┆ 2             ┆ f"FOO//{$code}//{$code_modifie… ┆ C-2-3       ┆ OUT_VAL_for_3/2 │
        │ D    ┆ 3             ┆ f"FOO//{$code}//{$code_modifie… ┆ null        ┆ expanded form   │
        └──────┴───────────────┴─────────────────────────────────┴─────────────┴─────────────────┘

    A metadata source may already carry a scalar-mandated column in its final aggregated
    ``List`` shape — ``parent_codes`` as ``List(String)``, the shape a MEDS ``codes.parquet``
    stores when it is itself the metadata source. Such a list is exploded into the mapper's
    scalar-per-row model: one row per element, with the join-key and sibling output columns
    replicated onto each. Empty and null lists explode to a single null element, flowing
    exactly like a null scalar (the reducer re-aggregates and dedups per code):

        >>> raw_metadata = pl.DataFrame({
        ...     "code": ["A", "B", "C"],
        ...     "label": ["a", "b", "c"],
        ...     "parent_codes": [["P/1", "P/2"], [], None],
        ... })
        >>> compiled = compile_metadata_block(
        ...     {"code": "$code", "description": "$label", "parent_codes": "$parent_codes"},
        ...     {"code"},
        ...     code_template_str='f"LAB//{$code}"',
        ... )
        >>> extract_metadata(raw_metadata, compiled)
        shape: (4, 4)
        ┌──────┬─────────────────┬─────────────┬──────────────┐
        │ code ┆ code_template   ┆ description ┆ parent_codes │
        │ ---  ┆ ---             ┆ ---         ┆ ---          │
        │ str  ┆ str             ┆ str         ┆ str          │
        ╞══════╪═════════════════╪═════════════╪══════════════╡
        │ A    ┆ f"LAB//{$code}" ┆ a           ┆ P/1          │
        │ A    ┆ f"LAB//{$code}" ┆ a           ┆ P/2          │
        │ B    ┆ f"LAB//{$code}" ┆ b           ┆ null         │
        │ C    ┆ f"LAB//{$code}" ┆ c           ┆ null         │
        └──────┴─────────────────┴─────────────┴──────────────┘
    """
    df_select_exprs: dict[str, pl.Expr] = {k: node.polars_expr for k, node in compiled.exprs.items()}

    match_cols = list(compiled.key_cols)
    final_cols = list(compiled.output_cols)

    columns = metadata_df.collect_schema().names()
    missing_cols = sorted(compiled.referenced_columns - set(columns))
    if missing_cols:
        raise ValueError(
            f"_metadata block for code {compiled.code_template!r} references column(s) {missing_cols} "
            f"not present in the metadata table. Available columns: {sorted(columns)}."
        )

    metadata_df = metadata_df.select(**df_select_exprs).with_columns(
        code_template=pl.lit(compiled.code_template),
    )

    metadata_df = metadata_df.filter(~pl.all_horizontal(*[pl.col(c).is_null() for c in final_cols]))

    metadata_df = _apply_mapper_mandatory_types(metadata_df, final_cols)

    return metadata_df.unique(maintain_order=True).select(*match_cols, "code_template", *final_cols)


def _apply_mapper_mandatory_types(
    metadata_df: pl.LazyFrame | pl.DataFrame, final_cols: list[str]
) -> pl.LazyFrame | pl.DataFrame:
    """Coerce MEDS-sentinel output columns to the mapper's mandated dtypes.

    Shared by :func:`extract_metadata` and :func:`extract_self_metadata` so both mapper
    paths emit identical shapes for the reducer — a ``description`` must be String and a
    ``parent_codes`` a scalar String regardless of which path produced it (the reducer's
    ``list.join`` / list aggregation assume these dtypes).
    """
    for mandatory_col, mandatory_type in _MAPPER_MANDATORY_TYPES.items():
        if mandatory_col not in final_cols:
            continue

        actual_type = metadata_df.collect_schema()[mandatory_col]
        if actual_type == mandatory_type:
            continue

        if isinstance(actual_type, pl.List) and not isinstance(mandatory_type, pl.List):
            # A metadata source may already carry a scalar-mandated column in its final
            # aggregated List shape — e.g. ``parent_codes`` as ``List(String)`` when a MEDS
            # ``codes.parquet`` is itself the metadata source. The mapper's model is
            # scalar-per-row, so explode the list into one row per element; the join-key and
            # sibling output columns replicate onto every emitted row, and the reducer
            # re-aggregates the scalars per code (deduplicating in the process). Empty and
            # null lists explode to a single null element, flowing exactly like a null
            # scalar.
            logger.warning(
                f"Metadata column '{mandatory_col}' is {actual_type} but the mapper mandates the "
                f"scalar {mandatory_type}. Exploding to one row per element."
            )
            metadata_df = metadata_df.explode(mandatory_col)
            actual_type = metadata_df.collect_schema()[mandatory_col]
            if actual_type == mandatory_type:
                continue

        logger.warning(f"Metadata column '{mandatory_col}' must be of type {mandatory_type}. Casting.")
        metadata_df = metadata_df.with_columns(pl.col(mandatory_col).cast(mandatory_type, strict=False))

    return metadata_df


def observed_codes_shard(events_df: pl.LazyFrame) -> pl.DataFrame:
    """One shard's distinct observed codes — the map half of the vocabulary union.

    The reducer unions these vocabulary-sized partials into the full observed-code
    universe, so no single pass ever scans the whole dataset: per-task memory is
    bounded by one shard's distinct codes.

    Examples:
        >>> observed_codes_shard(pl.LazyFrame({"code": ["B", "A", "B", None]}))
        shape: (2, 1)
        ┌──────┐
        │ code │
        │ ---  │
        │ str  │
        ╞══════╡
        │ A    │
        │ B    │
        └──────┘
    """
    return events_df.select("code").drop_nulls().unique().sort("code").collect()


def extract_self_metadata(
    events_df: pl.LazyFrame, compiled: CompiledMetadataBlock, source_block: str
) -> pl.DataFrame:
    """Produce one ``_self`` block's expanded metadata rows from ONE shard's event rows.

    The ``_self`` expressions were already evaluated during event extraction into the
    ``METADATA_COMPONENTS_COL`` struct, and the assembled code sits on the same rows —
    so the mapper needs no join at all: it emits the shard's distinct
    ``(code, code_template, outputs)`` rows directly, scoped to the declaring event.
    Rows whose outputs are all null are dropped, mirroring :func:`extract_metadata`;
    sentinel outputs get the same mandatory-dtype coercions as the external path; the
    result is sorted so partial-file bytes are deterministic regardless of event-file
    row order. Per-task memory is bounded by one shard.

    Struct fields are read via ``struct.field`` rather than ``unnest`` so a component
    name used by one event and a metadata output name used by another can coexist
    without a duplicate-column collision.

    Examples:
        >>> events = pl.LazyFrame({
        ...     "code": ["CHART//1", "CHART//1", "CHART//2", "OTHER"],
        ...     "code_components": [{"itemid": 1}, {"itemid": 1}, {"itemid": 2}, {"itemid": None}],
        ...     "metadata_components": [
        ...         {"desc": "HR"}, {"desc": "HR"}, {"desc": "RR"}, {"desc": "ignored"},
        ...     ],
        ...     "source_block": ["chart/c", "chart/c", "chart/c", "other/o"],
        ... })
        >>> compiled = compile_self_metadata_block(
        ...     {"desc": "$long_label"}, {"itemid"}, code_template_str='f"CHART//{$itemid}"'
        ... )
        >>> extract_self_metadata(events, compiled, "chart/c")
        shape: (2, 3)
        ┌──────────┬─────────────────────┬──────┐
        │ code     ┆ code_template       ┆ desc │
        │ ---      ┆ ---                 ┆ ---  │
        │ str      ┆ str                 ┆ str  │
        ╞══════════╪═════════════════════╪══════╡
        │ CHART//1 ┆ f"CHART//{$itemid}" ┆ HR   │
        │ CHART//2 ┆ f"CHART//{$itemid}" ┆ RR   │
        └──────────┴─────────────────────┴──────┘
    """
    out_exprs = [pl.col(METADATA_COMPONENTS_COL).struct.field(c).alias(c) for c in compiled.output_cols]
    scoped = events_df.filter(pl.col(SOURCE_BLOCK_COL) == source_block)

    # A shard file carrying rows of this block but no METADATA_COMPONENTS_COL (or null
    # structs — extraction always emits a non-null struct on a ``_self`` event's rows)
    # predates the ``_self`` config. Without this check, stale shards' rows would flow
    # through as all-null outputs and be silently dropped, so metadata would just be
    # missing for their codes. A file with NO rows of this block (e.g. a false positive
    # from table-name matching) contributes an empty, correctly-shaped partial instead.
    schema_names = events_df.collect_schema().names()
    if METADATA_COMPONENTS_COL not in schema_names:
        n_total = scoped.select(pl.len()).collect().item()
        if n_total == 0:
            return pl.DataFrame(
                schema={
                    "code": pl.String,
                    "code_template": pl.String,
                    **dict.fromkeys(compiled.output_cols, pl.String),
                }
            )
        raise ValueError(
            f"{n_total} extracted event rows for source block {source_block!r} carry no "
            f"'{METADATA_COMPONENTS_COL}' values. Those event files predate this "
            f"'{SELF_METADATA_PREFIX}' configuration; re-run the extraction pipeline before "
            "extracting code metadata."
        )
    n_total, n_null = (
        scoped.select(pl.len(), pl.col(METADATA_COMPONENTS_COL).is_null().sum()).collect().row(0)
    )
    if n_null:
        raise ValueError(
            f"{n_null} of {n_total} extracted event rows for source block {source_block!r} carry "
            f"no '{METADATA_COMPONENTS_COL}' values. Those event files predate this "
            f"'{SELF_METADATA_PREFIX}' configuration; re-run the extraction pipeline before "
            "extracting code metadata."
        )

    out = (
        scoped.select("code", pl.lit(compiled.code_template).alias("code_template"), *out_exprs)
        .filter(~pl.all_horizontal(*(pl.col(c).is_null() for c in compiled.output_cols)))
        .unique()
        .sort(["code", *compiled.output_cols])
        .collect()
    )
    return _apply_mapper_mandatory_types(out, list(compiled.output_cols)).select(
        "code", "code_template", *compiled.output_cols
    )


def expand_metadata_shard(
    events_df: pl.LazyFrame,
    metadata_df: pl.LazyFrame | pl.DataFrame,
    compiled: CompiledMetadataBlock,
    source_block: str,
) -> pl.DataFrame:
    """Expand one external ``_metadata`` block against ONE shard's observed components.

    The per-shard form of the component join: evaluate the block's dftly program over
    the (vocabulary-sized) raw metadata table via :func:`extract_metadata`, build this
    shard's distinct component map scoped to the declaring event, and inner-join the two
    to attach metadata to full codes. Per-task memory is bounded by one shard plus the
    metadata table; the reducer only ever unions these vocabulary-sized expansions.

    Join semantics are unchanged from the previous reducer-side expansion:

    - Keys are normalized to a canonical String rendering on both sides
      (:func:`normalize_join_key`) — components keep raw source dtypes while csv-sourced
      metadata keys are uniformly String.
    - ``nulls_equal=True``: a null metadata key matches exactly the rows whose
      (``??``-coalesced) component is null.
    - The component side is deduplicated and sorted into a canonical total order, and
      ``maintain_order="left_right"`` pins within-key row order to the metadata table's
      row order — both load-bearing for byte-identical output, because the downstream
      aggregations fold row order into description join order and list order.

    Examples:
        >>> events = pl.LazyFrame({
        ...     "code": ["CHART//1", "CHART//2", "LAB//9"],
        ...     "code_components": [{"itemid": 1}, {"itemid": 2}, {"itemid": 9}],
        ...     "source_block": ["chart/c", "chart/c", "labs/l"],
        ... })
        >>> compiled = compile_metadata_block(
        ...     {"itemid": "$itemid", "description": "$label"},
        ...     {"itemid"},
        ...     code_template_str='f"CHART//{$itemid}"',
        ... )
        >>> metadata = pl.DataFrame({"itemid": ["1", "2", "3"], "label": ["HR", "RR", "unused"]})
        >>> expand_metadata_shard(events, metadata, compiled, "chart/c")
        shape: (2, 3)
        ┌──────────┬─────────────────────┬─────────────┐
        │ code     ┆ code_template       ┆ description │
        │ ---      ┆ ---                 ┆ ---         │
        │ str      ┆ str                 ┆ str         │
        ╞══════════╪═════════════════════╪═════════════╡
        │ CHART//1 ┆ f"CHART//{$itemid}" ┆ HR          │
        │ CHART//2 ┆ f"CHART//{$itemid}" ┆ RR          │
        └──────────┴─────────────────────┴─────────────┘
    """
    pdf = extract_metadata(metadata_df, compiled).lazy()
    match_cols = list(compiled.key_cols)
    pdf_schema = pdf.collect_schema()
    metadata_cols = [c for c in pdf_schema.names() if c not in match_cols]
    pdf = pdf.with_columns(normalize_join_key(pl.col(c), pdf_schema[c]) for c in match_cols)

    events_schema = events_df.collect_schema()
    component_dtypes = (
        {f.name: f.dtype for f in events_schema["code_components"].fields}
        if "code_components" in events_schema.names()
        else {}
    )
    missing = [c for c in match_cols if c not in component_dtypes]
    if missing:
        # A shard file carrying rows of this block but missing its component columns
        # predates the configuration; a file with NO rows of this block (e.g. a false
        # positive from table-name matching) contributes an empty, correctly-shaped
        # partial instead.
        n_scoped = (
            events_df.filter(pl.col(SOURCE_BLOCK_COL) == source_block).select(pl.len()).collect().item()
        )
        if n_scoped == 0:
            return pl.DataFrame(schema={"code": pl.String, **{c: pdf_schema[c] for c in metadata_cols}})
        raise ValueError(
            f"Extracted event rows for source block {source_block!r} carry no component "
            f"column(s) {missing} (components present: {sorted(component_dtypes)}). The event "
            "files predate this configuration; re-run the extraction pipeline before "
            "extracting code metadata."
        )
    components = (
        events_df.filter(pl.col(SOURCE_BLOCK_COL) == source_block)
        .select(
            pl.col("code").alias(FULL_CODE_COL),
            *[
                normalize_join_key(pl.col("code_components").struct.field(c), component_dtypes[c]).alias(c)
                for c in match_cols
            ],
        )
        .unique()
        .sort(FULL_CODE_COL, *match_cols)
    )

    return (
        components.join(pdf, on=match_cols, how="inner", nulls_equal=True, maintain_order="left_right")
        .select(pl.col(FULL_CODE_COL).alias("code"), *metadata_cols)
        .collect()
    )


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
    `metadata/codes.parquet` file. The metadata columns are extracted from the raw metadata files using
    ``_metadata`` blocks in the `MESSY_config_fp` file; see ``MEDS_extract.config`` (``MessyConfig`` and
    ``compile_metadata_block``) for the block semantics and the dftly documentation for the expression
    DSL.

    Metadata is attached to codes through one join path: each ``_metadata`` entry's key
    columns (the produced columns whose names match the code expression's component
    columns — see ``MEDS_extract.config.compile_metadata_block``) are joined against the
    observed ``code_components`` values, scoped to the declaring event block, with null
    keys matching null components.

    The stage is a shard-scoped map-reduce whose work items are the extracted event
    files, so per-task memory is bounded by one shard plus one (vocabulary-sized)
    metadata table — never by the dataset. Each map task writes vocabulary-sized
    partials (a shard's distinct observed codes, and per ``_metadata`` block the
    shard's metadata-to-code expansion — ``_self`` blocks read values off the shard's
    own rows with no join at all), and worker 0's reduction only ever unions and
    aggregates those partials. External metadata tables are re-read once per (shard,
    block) task — deliberate: they are vocabulary-sized, so re-reading a small file
    beats any whole-dataset pass.

    The output enumerates EVERY code observed in the data — recent MEDS spec versions
    require the full observed code vocabulary in ``metadata/codes.parquet`` for the
    dataset to be valid. Codes with no metadata match carry all-null metadata columns.
    When no ``_metadata`` blocks are configured at all, the stage still writes the full
    observed code vocabulary (a codes-only table).

    When ``_metadata`` blocks ARE configured, the stage's data input must be the
    component-bearing ``convert_to_MEDS_events`` output: every block's join keys must
    name component columns some event file carries, and an input with no
    ``code_components`` anywhere (e.g. merged data — ``merge_to_MEDS_cohort`` drops that
    column) is an error, never a silent codes-only degrade.

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

    messy_cfg = MessyConfig.load(cfg.MESSY_config_fp)

    partial_metadata_dir.mkdir(parents=True, exist_ok=True)

    events_and_metadata_by_metadata_fp = messy_cfg.events_by_metadata_prefix()
    if not events_and_metadata_by_metadata_fp:
        # Not an early return: MEDS validity requires codes.parquet to enumerate every
        # observed code even when there is no metadata to attach, so the (empty) map loop
        # no-ops and worker 0 still runs the reducer to write the codes-only table.
        logger.info(
            "No _metadata blocks in the MESSY config found. "
            "The output codes.parquet will hold the observed code vocabulary with no metadata columns."
        )

    # Enumerate the extracted event files, handling heterogeneous schemas (files whose
    # codes reference source columns carry code_components; all-literal files don't).
    # The file list is sorted so partial naming and reduction order never depend on
    # directory enumeration order, and hidden files are never source data, so they are
    # excluded from the scan.
    event_parquet_files = sorted(
        fp for fp in Path(stage_input_dir).rglob("*.parquet") if not fp.name.startswith(".")
    )
    all_event_dfs = [pl.scan_parquet(fp, glob=False) for fp in event_parquet_files]

    # Schema validation runs in EVERY worker (it's a cheap metadata-only check) so a
    # malformed events layout fails loudly everywhere — but the component map itself is
    # only materialized by the reducer (worker 0) below: it is a full-dataset
    # scan/unique/collect that the N-1 map-only workers never use. The joinable component
    # columns are the union across per-file schemas, so the decision cannot depend on
    # which file an enumeration yields first.
    event_schemas = [df.collect_schema() for df in all_event_dfs]
    component_fields = union_component_fields(event_schemas)

    # The union of ``_self`` metadata output fields across event files — the analogue of
    # ``component_fields`` for METADATA_COMPONENTS_COL, used to fail fast (per block, in
    # every worker) when the extracted events predate the ``_self`` configuration.
    metadata_fields: set[str] = set()
    for s in event_schemas:
        if METADATA_COMPONENTS_COL in s.names():
            metadata_fields.update(f.name for f in s[METADATA_COMPONENTS_COL].fields)

    # A `_metadata` block always attaches to a component-bearing code (a literal code
    # with a `_metadata` block is rejected at config compile), so configured blocks over
    # an input with no components ANYWHERE can never be a valid convert_to_MEDS_events
    # output. Degrading to a codes-only table here would silently discard every extracted
    # metadata column, so this is an error, not a warning.
    if events_and_metadata_by_metadata_fp and not component_fields:
        raise ValueError(
            "The MESSY config declares _metadata blocks, but no extracted-event file under "
            f"{stage_input_dir} carries a 'code_components' column, so no metadata could ever "
            "be joined onto the codes. A _metadata block always attaches to a component-bearing "
            "code, so a convert_to_MEDS_events output cannot lack components entirely — this "
            "input is most likely merged data (merge_to_MEDS_cohort drops 'code_components'). "
            "Run extract_code_metadata on the convert_to_MEDS_events output: order it before "
            "merge_to_MEDS_cohort in the pipeline's stages list."
        )

    # ── Map phase: work items are event-shard files ─────────────────────────────
    # Two kinds of vocabulary-sized partials, each computed from ONE shard per task so
    # per-task memory is bounded by shard size (the pipeline's shard-size-controls-
    # memory guarantee):
    #   - observed-code partials: one per event file, that shard's distinct codes; the
    #     reducer unions them into the full observed vocabulary.
    #   - expanded-metadata partials: one per (event file, _metadata block); ``_self``
    #     blocks read metadata values off the shard's own rows (no join at all), and
    #     external blocks join the vocabulary-sized metadata table against the shard's
    #     distinct component map.
    #
    # Every block is compiled and validated up front, in EVERY worker, so configuration
    # errors (a literal code with a _metadata block, a block producing no join-key
    # columns, config-vs-data drift) surface even when every output shard already exists
    # and the compute is skipped. The block ordinal is the canonical reduction order.
    compiled_blocks: list[dict] = []
    for input_prefix, event_metadata_cfgs in events_and_metadata_by_metadata_fp.items():
        is_self = input_prefix == SELF_METADATA_PREFIX
        for event_cfg in copy.deepcopy(event_metadata_cfgs):
            # Always present: ``events_by_metadata_prefix`` stamps every entry (see
            # SOURCE_BLOCK_COL in config.py); a KeyError here means that contract broke.
            source_block = event_cfg.pop(SOURCE_BLOCK_COL)
            compiled = _compile_metadata_entry(event_cfg, self_block=is_self)
            match_cols = list(compiled.key_cols)

            # Every join key must name a component column some event file carries, or the
            # block could never attach and its extracted metadata would be discarded
            # wholesale — config-vs-data drift fails fast rather than surfacing as a
            # reducer-side degrade after the map compute.
            missing = [c for c in match_cols if c not in component_fields]
            if missing:
                raise ValueError(
                    f"_metadata block for code {compiled.code_template!r} (source block "
                    f"{source_block!r}) joins on component column(s) {missing} that no "
                    f"extracted-event file carries. Available component columns: "
                    f"{sorted(component_fields)}. The extracted events do not match this "
                    "configuration; re-run the extraction pipeline before extracting code "
                    "metadata."
                )
            # The ``_self`` analogue: every output must exist as a metadata field in some
            # event file, or the extracted events predate this block's configuration.
            # (Per-shard staleness is caught at compute time in
            # ``extract_self_metadata`` via its null-struct check.)
            if is_self:
                missing_out = [c for c in compiled.output_cols if c not in metadata_fields]
                if missing_out:
                    raise ValueError(
                        f"'{SELF_METADATA_PREFIX}' metadata block for code "
                        f"{compiled.code_template!r} (source block {source_block!r}) produces "
                        f"column(s) {missing_out} that no extracted-event file carries in "
                        f"'{METADATA_COMPONENTS_COL}'. Available metadata fields: "
                        f"{sorted(metadata_fields)}. The extracted events predate this "
                        "configuration; re-run the extraction pipeline before extracting "
                        "code metadata."
                    )

            compiled_blocks.append(
                {
                    "ordinal": len(compiled_blocks),
                    "prefix": input_prefix,
                    "is_self": is_self,
                    "source_block": source_block,
                    "compiled": compiled,
                    "match_cols": match_cols,
                }
            )

    # External metadata sources are resolved (and size-checked) once per prefix. The
    # tables are read and joined once per (shard, block) task — deliberately: they are
    # vocabulary-sized, so re-reading a small file per shard is far cheaper than any
    # whole-dataset pass, and it keeps the stage a single map-reduce.
    metadata_sources: dict[str, tuple[list, dict]] = {}
    for rec in compiled_blocks:
        if rec["is_self"] or rec["prefix"] in metadata_sources:
            continue
        metadata_fps = resolve_source_files(raw_input_dir, rec["prefix"])
        try:
            total_bytes = sum(fp.stat().st_size for fp in metadata_fps)
        except Exception:
            total_bytes = 0
        if total_bytes > METADATA_TABLE_WARN_BYTES:
            logger.warning(
                f"Metadata table '{rec['prefix']}' is {total_bytes / 2**20:.0f} MiB on disk. "
                f"Metadata tables are fully materialized and joined once per event shard; a "
                f"table this large will be slow and may exhaust memory. If these columns come "
                f"from the event's own source table, use the '{SELF_METADATA_PREFIX}' metadata "
                f"prefix instead."
            )
        # Reader kwargs are chosen per resolved prefix: csv-family sources read with
        # ``infer_schema=False`` for a uniform all-String schema, while parquet sources
        # keep their intrinsic types (``infer_schema`` is csv-only and would crash
        # ``scan_parquet``). Deciding from the first file is sound because
        # ``scan_source`` enforces format homogeneity across a multi-file source.
        read_kwargs = {} if _format_family(metadata_fps[0]) == "parquet" else {"infer_schema": False}
        metadata_sources[rec["prefix"]] = (metadata_fps, read_kwargs)

    def shard_key(fp: Path) -> str:
        """Deterministic, path-safe partial-file stem for one event shard file."""
        return str(fp.relative_to(stage_input_dir).with_suffix("")).replace("/", "_")

    def block_matches_file(rec: dict, fp: Path) -> bool:
        """Whether a block's declaring table could have rows in this shard file.

        A block's rows live only in its declaring table's shard files, whose relative
        path (sans the split/shard directory prefix and format suffix) ends with the
        table prefix. The compute functions additionally filter by ``source_block``,
        so a false positive here only costs an empty partial.
        """
        rel = str(fp.relative_to(stage_input_dir).with_suffix(""))
        table_prefix = rec["source_block"].rsplit("/", 1)[0]
        return rel == table_prefix or rel.endswith("/" + table_prefix)

    observed_dir = partial_metadata_dir / "observed_codes"
    expanded_dir = partial_metadata_dir / "expanded"

    observed_fps: list[Path] = []
    tasks: list[tuple[Path, Path, dict | None]] = []
    for event_fp in event_parquet_files:
        out_fp = observed_dir / f"{shard_key(event_fp)}.parquet"
        observed_fps.append(out_fp)
        tasks.append((event_fp, out_fp, None))
    block_partial_fps: dict[int, list[Path]] = {rec["ordinal"]: [] for rec in compiled_blocks}
    for rec in compiled_blocks:
        # Path matching is a pruning optimization for the standard per-table layout
        # (<split>/<shard>/<table>.parquet). A layout that doesn't encode the table in
        # the path (e.g. one file per shard holding several tables' events) matches
        # nothing, so the block falls back to fanning out over every file — the
        # compute-side ``source_block`` filter keeps that correct, at the cost of
        # per-file scans that mostly produce empty partials.
        matched = [fp for fp in event_parquet_files if block_matches_file(rec, fp)]
        for event_fp in matched or event_parquet_files:
            out_fp = expanded_dir / f"b{rec['ordinal']}" / f"{shard_key(event_fp)}.parquet"
            block_partial_fps[rec["ordinal"]].append(out_fp)
            tasks.append((event_fp, out_fp, rec))

    def read_events(fp):
        return pl.scan_parquet(fp, glob=False)

    # Shuffled so parallel workers spread across tasks instead of contending on the
    # first one; the bookkeeping above is deterministic and identical in every worker.
    shuffled_tasks = list(tasks)
    random.shuffle(shuffled_tasks)
    for event_fp, out_fp, rec in shuffled_tasks:
        if rec is None:
            in_fps, read_fn, compute_fn = event_fp, read_events, observed_codes_shard
        elif rec["is_self"]:
            in_fps, read_fn = event_fp, read_events
            compute_fn = partial(
                extract_self_metadata, compiled=rec["compiled"], source_block=rec["source_block"]
            )
        else:
            metadata_fps, read_kwargs = metadata_sources[rec["prefix"]]
            in_fps = [event_fp, *metadata_fps]

            def read_fn(fps, read_kwargs=read_kwargs):
                return pl.scan_parquet(fps[0], glob=False), scan_source(fps[1:], **read_kwargs)

            def compute_fn(pair, rec=rec):
                return expand_metadata_shard(
                    pair[0], pair[1], compiled=rec["compiled"], source_block=rec["source_block"]
                )

        rwlock_wrap(
            in_fps,
            out_fp,
            read_fn,
            atomic_write_parquet,
            compute_fn,
            do_overwrite=cfg.do_overwrite,
            # Run-scoped do_overwrite (MT 0.7.0): without the marker dir, parallel
            # workers treat each other's fresh outputs as stale and redo the work.
            marker_dir=run_marker_dir(cfg),
        )

    logger.info("Extracted metadata for all events. Merging.")

    if cfg.worker != 0:
        logger.info("Code metadata extraction completed. Exiting")
        return

    logger.info("Starting reduction process")

    all_partial_fps = observed_fps + [fp for fps in block_partial_fps.values() for fp in fps]
    wait_for_complete_parquets(all_partial_fps, polling_time=cfg.polling_time)

    start = datetime.now(tz=UTC)
    logger.info("All map shards complete! Starting code metadata reduction computation.")

    # Every reduction input below is a vocabulary-sized partial — the expansion to full
    # codes already happened per shard in the map phase, so the reducer never touches
    # event data. Frames are processed in canonical (block ordinal, shard path) order so
    # the reduction (concat order, and with it description join order and
    # list-aggregation order) is identical across runs — the mappers' per-worker
    # random.shuffle above never leaks into reduction order.
    expanded_dfs = []
    for rec in compiled_blocks:
        fps = sorted(block_partial_fps[rec["ordinal"]])
        if not fps:
            # No shard file even belongs to the declaring table — nothing to contribute,
            # not even a schema. (Distinct from the matched-zero-codes case below, where
            # empty partials still carry the block's output schema into the reduction.)
            logger.warning(
                f"Metadata for source block {rec['source_block']!r} (metadata prefix "
                f"{rec['prefix']!r}): no extracted event file matches the declaring table."
            )
            continue
        block_df = pl.concat(
            [pl.scan_parquet(fp, glob=False) for fp in fps], how="diagonal_relaxed"
        ).collect()
        if block_df.is_empty():
            logger.warning(
                f"Metadata for source block {rec['source_block']!r} (metadata prefix "
                f"{rec['prefix']!r}, match columns {rec['match_cols']}) matched zero codes."
            )

        # One code, several distinct metadata rows = per-occurrence data leaking into
        # code-level metadata. The aggregation below resolves the conflict (description
        # joined, other columns to lists), so this is legal — but it usually means the
        # varying value belongs in the code itself or in an event value column, hence a
        # WARNING naming an offending code. Only checked for ``_self`` blocks: external
        # vocabulary tables legitimately carry multiple rows per key (e.g. multiple
        # parent codes).
        if rec["is_self"]:
            distinct = block_df.unique()
            n_codes = distinct.select(pl.col("code").n_unique()).item()
            if distinct.height > n_codes:
                example = (
                    distinct.group_by("code", maintain_order=True)
                    .len()
                    .filter(pl.col("len") > 1)
                    .head(1)["code"]
                    .item()
                )
                logger.warning(
                    f"_self metadata for source block {rec['source_block']!r}: "
                    f"{distinct.height - n_codes} of {distinct.height} distinct metadata rows "
                    f"are extra rows for an already-seen code (e.g. {example!r}). Metadata "
                    f"that varies across occurrences of one code is usually data leaking "
                    f"into metadata — consider making the varying column part of the code or "
                    f"an event value column instead. The conflicting values will be "
                    f"aggregated per code (description joined with the separator; other "
                    f"columns collected into lists)."
                )
        expanded_dfs.append(block_df.lazy())

    if not expanded_dfs:
        logger.info("No metadata to reduce. The output will hold only the observed code vocabulary.")
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
    #   - ``parent_codes``: List(String), unioned across rows/sources, nulls dropped,
    #     deduplicated in first-seen order. Mapper rows always carry parent_codes as a
    #     nullable scalar String (one dftly expression -> at most one parent per
    #     metadata row; ``_MAPPER_MANDATORY_TYPES`` enforces the cast), so the
    #     aggregation assumes the scalar shape.
    #   - ``code_template``: String. One code has exactly one template; distinct templates
    #     colliding on one code is a config error and raises below.
    #   - every other metadata column: List(String), nulls dropped, distinct values sorted
    #     within each group so multi-source aggregation is order-stable.
    # A code with no value for a column gets null (never an empty list / empty string).
    reduced_schema = reduced.collect_schema()
    aggs = {}
    for c in metadata_cols:
        if (
            c == CodeMetadataSchema.description_name
            or c == CodeMetadataSchema.parent_codes_name
            or c == "code_template"
        ):
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

    # MEDS validity requires codes.parquet to enumerate EVERY code observed in the data,
    # so seed the reduction with the observed code universe: metadata-matched codes keep
    # their rows, and every other observed code gets one all-null metadata row. The left
    # join is exact, not lossy: every metadata code above came through an inner join
    # against observed components, so metadata codes are a subset of observed codes.
    # The universe is the union of the per-shard observed-code partials — every input
    # here is vocabulary-sized, never a full-dataset scan.
    observed_codes = (
        pl.concat([pl.scan_parquet(fp, glob=False) for fp in observed_fps], how="vertical")
        .unique()
        .sort("code")
        .collect()
    )
    reduced = observed_codes.join(reduced, on="code", how="left")

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
