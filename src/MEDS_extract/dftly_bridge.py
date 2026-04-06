"""Bridge module between MEDS_extract's MESSY config system and the dftly expression language.

This module translates MESSY config values into dftly expressions and then into Polars expressions,
enabling users to write dftly expressions inline in their MESSY YAML configs.
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

# Patterns that indicate a string is a dftly expression rather than a legacy MESSY value.
# These are checked against config values to decide whether to route through dftly parsing.
_DFTLY_OPERATOR_PATTERNS = re.compile(
    r"""
    (?:^|\s) as \s        |  # type casting: "col as float"
    \s @ \s               |  # timestamp resolution: "date @ time"
    \s if \s              |  # conditional: "a if b else c"
    \s else \s            |  # conditional (else branch)
    ^extract\s            |  # regex: "extract group 1 of ..."
    ^hash\(               |  # hash: "hash(col)"
    ^not\s                |  # boolean negation
    \s && \s              |  # boolean and
    \s \|\| \s            |  # boolean or
    \{[^}]+\}             |  # string interpolation: "{col1} // {col2}"
    \s [+\-] \s           |  # arithmetic: "col1 + col2"
    \s (?>|>=?|<=?) \s    |  # comparison operators
    \s in \s [\[\(]          # range/set membership: "col in [0, 10)"
    """,
    re.VERBOSE,
)

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
        >>> sorted(get_referenced_columns(Expression(
        ...     type="CONDITIONAL",
        ...     arguments={"condition": Column(name="flag", type="bool"),
        ...                "true_value": Column(name="a", type="int"),
        ...                "false_value": Literal(value=0)}
        ... )))
        ['a', 'flag']
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
    # Literal and other types have no column references


def is_dftly_expr(value: str) -> bool:
    """Detects whether a MESSY config value string is a dftly expression.

    Uses a heuristic based on the presence of dftly operator keywords/symbols.
    Returns False for legacy ``col(X)`` references, None, and plain identifiers.

    Args:
        value: A string value from a MESSY event config field.

    Returns:
        True if the value appears to be a dftly expression.

    Examples:
        >>> is_dftly_expr("col(timestamp)")
        False
        >>> is_dftly_expr("timestamp")
        False
        >>> is_dftly_expr("ADMISSION")
        False
        >>> is_dftly_expr("col1 + col2")
        True
        >>> is_dftly_expr("col1 as float")
        True
        >>> is_dftly_expr("hash(mrn)")
        True
        >>> is_dftly_expr("{test_name} // {units}")
        True
        >>> is_dftly_expr("date_col @ time_col")
        True
        >>> is_dftly_expr("val if flag else 0")
        True
        >>> is_dftly_expr("extract group 1 of abc from col")
        True
    """
    if not isinstance(value, str):
        return False
    return bool(_DFTLY_OPERATOR_PATTERNS.search(value))


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


def compile_field_expr(
    field_name: str,
    value: str,
    input_schema: dict[str, str | None],
) -> pl.Expr:
    """Compiles a single MESSY event field value into a Polars expression via dftly.

    Args:
        field_name: The MESSY field name (e.g., "time", "numeric_value").
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
    """
    parsed = parse({field_name: value}, input_schema=input_schema)
    return to_polars(parsed[field_name])


def compile_code_interpolation(
    code_str: str,
    input_schema: dict[str, str | None],
) -> tuple[pl.Expr, pl.Expr | None, set[str]]:
    """Compiles a dftly string interpolation code value into a Polars expression.

    Matches the return signature of ``get_code_expr()`` for drop-in compatibility.

    Args:
        code_str: A dftly string interpolation expression like ``"{test} // {units}"``.
        input_schema: The dftly input schema.

    Returns:
        A tuple of (code_expr, null_filter_expr, needed_columns).

    Examples:
        >>> expr, null_filter, cols = compile_code_interpolation(
        ...     "{test} // {units}",
        ...     {"test": "str", "units": "str"}
        ... )
        >>> sorted(cols)
        ['test', 'units']
        >>> import polars as pl
        >>> df = pl.DataFrame({"test": ["Lab", "Vital"], "units": ["mg", "mmHg"]})
        >>> df.select(code=expr)  # doctest: +SKIP
    """
    parsed = parse({"code": code_str}, input_schema=input_schema)
    node = parsed["code"]
    code_expr = to_polars(node)
    needed_cols = get_referenced_columns(node)

    # Build null filter on the first referenced column (matching get_code_expr behavior)
    null_filter = None
    if needed_cols:
        first_col = sorted(needed_cols)[0]
        null_filter = pl.col(first_col).is_not_null()

    return code_expr, null_filter, needed_cols


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


def extract_columns_from_dftly_value(
    value: str,
    input_schema: dict[str, str | None],
) -> set[str]:
    """Parses a dftly expression string and extracts all referenced column names.

    Args:
        value: A dftly expression string.
        input_schema: The dftly input schema (needed for column name resolution).

    Returns:
        A set of referenced column names.

    Examples:
        >>> sorted(extract_columns_from_dftly_value("a + b", {"a": "int", "b": "int"}))
        ['a', 'b']
        >>> extract_columns_from_dftly_value("hash(mrn)", {"mrn": "str"})
        {'mrn'}
        >>> sorted(extract_columns_from_dftly_value("{test} // {unit}", {"test": "str", "unit": "str"}))
        ['test', 'unit']
    """
    parsed = parse({"_": value}, input_schema=input_schema)
    return get_referenced_columns(parsed["_"])


# dftly type keywords that should not be treated as column names
_DFTLY_TYPE_KEYWORDS = frozenset({
    "int", "float", "bool", "str", "string", "date", "datetime", "duration",
})

# dftly operator/syntax keywords
_DFTLY_SYNTAX_KEYWORDS = frozenset({
    "as", "if", "else", "extract", "group", "of", "from", "match", "against",
    "not", "and", "or", "in", "hash", "coalesce", "null", "true", "false",
})

# Regex for bare identifiers (Python-style names)
_IDENTIFIER_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")

# Regex for string interpolation references: {col_name}
_INTERPOLATION_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def extract_columns_from_dftly_value_schemaless(value: str) -> set[str]:
    """Extracts likely column references from a dftly expression string without needing a schema.

    Uses regex heuristics to find identifiers that are likely column references.
    This is a best-effort approach for use during column discovery (before the file schema
    is available).

    Args:
        value: A dftly expression string.

    Returns:
        A set of likely column name strings.

    Examples:
        >>> sorted(extract_columns_from_dftly_value_schemaless("a + b"))
        ['a', 'b']
        >>> extract_columns_from_dftly_value_schemaless("hash(mrn)")
        {'mrn'}
        >>> sorted(extract_columns_from_dftly_value_schemaless("{test} // {unit}"))
        ['test', 'unit']
        >>> extract_columns_from_dftly_value_schemaless("col1 as float")
        {'col1'}
        >>> sorted(extract_columns_from_dftly_value_schemaless("date_col @ time_col"))
        ['date_col', 'time_col']
        >>> sorted(extract_columns_from_dftly_value_schemaless("val if flag else 0"))
        ['flag', 'val']
    """
    columns: set[str] = set()

    # Extract interpolation references first
    for match in _INTERPOLATION_RE.finditer(value):
        columns.add(match.group(1))

    # Extract bare identifiers, filtering out keywords and type names
    all_keywords = _DFTLY_TYPE_KEYWORDS | _DFTLY_SYNTAX_KEYWORDS
    for match in _IDENTIFIER_RE.finditer(value):
        name = match.group(1)
        if name not in all_keywords:
            columns.add(name)

    return columns
