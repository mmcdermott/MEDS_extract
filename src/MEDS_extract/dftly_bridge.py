"""Bridge module between MEDS_extract's MESSY config system and the dftly expression language.

All MESSY config values (code, time, value fields) are parsed through dftly with the file's schema.
dftly disambiguates columns vs literals via node types: names matching the schema become ``Column``
nodes (→ ``pl.col(name)``), names not in the schema become ``Literal`` nodes (→ ``pl.lit(value)``),
and expressions with operators/functions become ``Expression`` nodes (→ compiled Polars expressions).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import polars as pl
from dftly import Column, Expression, Literal, parse  # noqa: F401 (Literal used in doctests)
from dftly.polars import map_to_polars, to_polars

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

# Polars dtype -> dftly type string mapping
_POLARS_TO_DFTLY_TYPE: dict[type[pl.DataType], str] = {
    pl.Int8: "int",
    pl.Int16: "int",
    pl.Int32: "int",
    pl.Int64: "int",
    pl.UInt8: "int",
    pl.UInt16: "int",
    pl.UInt32: "int",
    pl.UInt64: "int",
    pl.Float32: "float",
    pl.Float64: "float",
    pl.Boolean: "bool",
    pl.Utf8: "str",
    pl.Date: "date",
    pl.Datetime: "datetime",
    pl.Duration: "duration",
}

# Structural keys in the event config that are not event field definitions.
EVENT_META_KEYS = {"_metadata", "join", "transforms", "schema", "subject_id_expr", "subject_id_col"}


def polars_schema_to_dftly_schema(schema: Mapping[str, pl.DataType]) -> dict[str, str | None]:
    """Converts a Polars schema to a dftly input_schema dict.

    Args:
        schema: A Polars schema mapping column names to data types.

    Returns:
        A dict mapping column names to dftly type strings. Unknown types map to None.

    Examples:
        >>> polars_schema_to_dftly_schema({"a": pl.Int64, "b": pl.Utf8, "c": pl.Datetime})
        {'a': 'int', 'b': 'str', 'c': 'datetime'}
        >>> polars_schema_to_dftly_schema({"x": pl.Categorical})
        {'x': None}
    """
    result = {}
    for name, dtype in schema.items():
        dtype_class = type(dtype) if isinstance(dtype, pl.DataType) else dtype
        result[name] = _POLARS_TO_DFTLY_TYPE.get(dtype_class)
    return result


def get_referenced_columns(node) -> set[str]:
    """Walks a dftly AST node tree to extract all referenced column names.

    Args:
        node: A dftly node (Column, Expression, Literal) or a dict/list of nodes.

    Returns:
        A set of column name strings referenced in the expression tree.

    Examples:
        >>> get_referenced_columns(Column(name="a", type="int"))
        {'a'}
        >>> get_referenced_columns(Literal(value=42))
        set()
        >>> sorted(get_referenced_columns(Expression(
        ...     type="ADD",
        ...     arguments=[Column(name="x", type="int"), Column(name="y", type="int")]
        ... )))
        ['x', 'y']
    """
    columns: set[str] = set()
    _walk_node(node, columns)
    return columns


def _walk_node(node, columns: set[str]) -> None:
    """Recursively walks a dftly node tree, collecting column names."""
    if isinstance(node, Column):
        columns.add(node.name)
    elif isinstance(node, Expression):
        if isinstance(node.arguments, list):
            for arg in node.arguments:
                _walk_node(arg, columns)
        elif isinstance(node.arguments, dict):
            for arg in node.arguments.values():
                _walk_node(arg, columns)
    elif isinstance(node, dict):
        for val in node.values():
            _walk_node(val, columns)
    elif isinstance(node, list):
        for item in node:
            _walk_node(item, columns)


def compile_field_expr(
    field_name: str,
    value: str,
    input_schema: dict[str, str | None],
) -> pl.Expr:
    """Compiles a single MESSY event field value into a Polars expression via dftly.

    The ``input_schema`` drives disambiguation:
    - Names matching the schema → ``pl.col(name)`` (Column node)
    - Names not in the schema → ``pl.lit(value)`` (Literal node)
    - Expressions with operators → compiled Polars expression (Expression node)

    Args:
        field_name: The output column name (e.g., "code", "time", "numeric_value").
        value: The dftly expression string.
        input_schema: The dftly input schema.

    Returns:
        A compiled Polars expression.

    Examples:
        >>> expr = compile_field_expr("time", 'ts as "%Y-%m-%d"', {"ts": "str"})
        >>> import polars as pl
        >>> df = pl.DataFrame({"ts": ["2021-01-01", "2021-06-15"]})
        >>> df.select(time=expr)
        shape: (2, 1)
        ┌────────────┐
        │ time       │
        │ ---        │
        │ date       │
        ╞════════════╡
        │ 2021-01-01 │
        │ 2021-06-15 │
        └────────────┘
        >>> compile_field_expr("code", "ADMISSION", {"ts": "str"})  # doctest: +SKIP
        >>> compile_field_expr("code", "ts", {"ts": "str"})  # doctest: +SKIP
    """
    parsed = parse({field_name: value}, input_schema=input_schema)
    return to_polars(parsed[field_name])


def compile_field_expr_with_columns(
    field_name: str,
    value: str,
    input_schema: dict[str, str | None],
) -> tuple[pl.Expr, set[str]]:
    """Like ``compile_field_expr`` but also returns the set of referenced column names.

    Args:
        field_name: The output column name.
        value: The dftly expression string.
        input_schema: The dftly input schema.

    Returns:
        A tuple of (polars_expr, referenced_columns).

    Examples:
        >>> expr, cols = compile_field_expr_with_columns(
        ...     "code", "{test} // {unit}", {"test": "str", "unit": "str"}
        ... )
        >>> sorted(cols)
        ['test', 'unit']
    """
    parsed = parse({field_name: value}, input_schema=input_schema)
    node = parsed[field_name]
    return to_polars(node), get_referenced_columns(node)


def compile_transforms(
    transforms_dict: dict[str, str],
    input_schema: dict[str, str | None],
) -> dict[str, pl.Expr]:
    """Compiles a MESSY ``transforms:`` block into Polars expressions.

    Args:
        transforms_dict: A dict mapping derived column names to dftly expression strings.
        input_schema: The dftly input schema for the raw data file.

    Returns:
        A dict mapping column names to compiled Polars expressions.

    Examples:
        >>> exprs = compile_transforms(
        ...     {"total": "a + b"},
        ...     {"a": "int", "b": "int"}
        ... )
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        >>> df.select(**exprs)
        shape: (2, 1)
        ┌───────┐
        │ total │
        │ ---   │
        │ i64   │
        ╞═══════╡
        │ 4     │
        │ 6     │
        └───────┘
    """
    parsed = parse(transforms_dict, input_schema=input_schema)
    return map_to_polars(parsed)


def compile_subject_id_expr(
    expr_str: str,
    input_schema: dict[str, str | None],
) -> tuple[pl.Expr, set[str]]:
    """Compiles a ``subject_id_expr`` into a Polars expression and its referenced columns.

    If the expression uses ``hash()``, the result is reinterpreted as signed Int64
    (since MEDS subject IDs must be Int64).

    Args:
        expr_str: A dftly expression string (e.g., ``"hash(MRN)"``).
        input_schema: The dftly input schema.

    Returns:
        A tuple of (polars_expr, referenced_columns).

    Examples:
        >>> expr, cols = compile_subject_id_expr("hash(mrn)", {"mrn": "str"})
        >>> cols
        {'mrn'}
        >>> import polars as pl
        >>> df = pl.DataFrame({"mrn": ["ABC", "DEF"]})
        >>> result = df.select(subject_id=expr)
        >>> result.schema  # doctest: +SKIP
    """
    parsed = parse({"subject_id": expr_str}, input_schema=input_schema)
    node = parsed["subject_id"]
    expr = to_polars(node)
    cols = get_referenced_columns(node)

    # hash() produces UInt64 but MEDS needs Int64 — reinterpret bits
    if isinstance(node, Expression) and node.type == "HASH_TO_INT":
        expr = expr.reinterpret(signed=True)

    return expr, cols


# --- Column discovery utilities (used before file schema is available) ---

# dftly type keywords that should not be treated as column names
_DFTLY_TYPE_KEYWORDS = frozenset(
    {
        "int",
        "float",
        "bool",
        "str",
        "string",
        "date",
        "datetime",
        "duration",
    }
)

# dftly operator/syntax keywords
_DFTLY_SYNTAX_KEYWORDS = frozenset(
    {
        "as",
        "if",
        "else",
        "extract",
        "group",
        "of",
        "from",
        "match",
        "against",
        "not",
        "and",
        "or",
        "in",
        "hash",
        "coalesce",
        "null",
        "true",
        "false",
    }
)

_IDENTIFIER_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")
_INTERPOLATION_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
_QUOTED_STRING_RE = re.compile(r'"[^"]*"|\'[^\']*\'')


def extract_columns_schemaless(value: str) -> set[str]:
    """Extracts likely column references from a dftly expression string without a schema.

    Uses regex heuristics to find identifiers that are likely column references, filtering
    out dftly keywords, type names, and identifiers inside quoted strings (e.g., format
    strings like ``"%Y-%m-%d"``).

    Args:
        value: A dftly expression string.

    Returns:
        A set of likely column name strings.

    Examples:
        >>> sorted(extract_columns_schemaless("a + b"))
        ['a', 'b']
        >>> extract_columns_schemaless("hash(mrn)")
        {'mrn'}
        >>> sorted(extract_columns_schemaless("{test} // {unit}"))
        ['test', 'unit']
        >>> extract_columns_schemaless("col1 as float")
        {'col1'}
        >>> sorted(extract_columns_schemaless("date_col @ time_col"))
        ['date_col', 'time_col']
        >>> sorted(extract_columns_schemaless("val if flag else 0"))
        ['flag', 'val']
        >>> extract_columns_schemaless("ADMISSION")
        {'ADMISSION'}
        >>> extract_columns_schemaless('charttime as "%m/%d/%Y, %H:%M:%S"')
        {'charttime'}
    """
    return _extract_identifiers(value, include_bare=True)


def extract_columns_schemaless_code(value: str) -> set[str]:
    """Like ``extract_columns_schemaless`` but for code fields where bare identifiers are literals.

    For code fields, only ``{interpolation}`` references are treated as column references.
    Bare identifiers (e.g., ``ADMISSION``) are string literals, not column names.

    Args:
        value: A dftly expression string for a code field.

    Returns:
        A set of likely column name strings.

    Examples:
        >>> extract_columns_schemaless_code("ADMISSION")
        set()
        >>> sorted(extract_columns_schemaless_code("{test} // {unit}"))
        ['test', 'unit']
        >>> extract_columns_schemaless_code("test_name")
        set()
        >>> sorted(extract_columns_schemaless_code("{a} // {b}")  )
        ['a', 'b']
    """
    return _extract_identifiers(value, include_bare=False)


def _extract_identifiers(value: str, *, include_bare: bool) -> set[str]:
    """Shared implementation for schemaless column extraction."""
    columns: set[str] = set()

    # {interpolation} references are always column references
    for match in _INTERPOLATION_RE.finditer(value):
        columns.add(match.group(1))

    if include_bare:
        # Strip quoted strings to avoid format specifiers like %Y, %m, %d
        stripped = _QUOTED_STRING_RE.sub("", value)
        all_keywords = _DFTLY_TYPE_KEYWORDS | _DFTLY_SYNTAX_KEYWORDS
        for match in _IDENTIFIER_RE.finditer(stripped):
            name = match.group(1)
            if name not in all_keywords:
                columns.add(name)

    return columns
