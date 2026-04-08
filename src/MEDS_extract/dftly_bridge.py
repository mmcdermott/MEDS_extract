"""Bridge module between MEDS_extract's MESSY config system and the dftly expression language.

With dftly 0.1.0+, columns use ``$`` prefix syntax (e.g., ``$col``), string interpolation uses
f-string syntax (e.g., ``f"CODE//{$col}"``), and type casts use ``::`` (e.g., ``$ts::"%Y-%m-%d"``).
No ``input_schema`` is needed — the ``$`` prefix unambiguously identifies columns.
"""

from __future__ import annotations

import polars as pl
from dftly import Parser
from dftly.nodes.arithmetic import Hash

# Structural keys in the event config that are not event field definitions.
EVENT_META_KEYS = {"_metadata", "join", "transforms", "schema", "subject_id_expr", "subject_id_col"}


def compile_field_expr(field_name: str, value: str) -> pl.Expr:
    """Compiles a single MESSY event field value into a Polars expression via dftly.

    Args:
        field_name: The output column name (e.g., "code", "time", "numeric_value").
        value: The dftly expression string.

    Returns:
        A compiled Polars expression.

    Examples:
        >>> expr = compile_field_expr("time", '$ts::"%Y-%m-%d"')
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
    return Parser.expr_to_polars(value)


def compile_field_expr_with_columns(
    field_name: str,
    value: str,
) -> tuple[pl.Expr, set[str]]:
    """Like ``compile_field_expr`` but also returns the set of referenced column names.

    Args:
        field_name: The output column name.
        value: The dftly expression string.

    Returns:
        A tuple of (polars_expr, referenced_columns).

    Examples:
        >>> expr, cols = compile_field_expr_with_columns(
        ...     "code", 'f"{$test}//{$unit}"'
        ... )
        >>> sorted(cols)
        ['test', 'unit']
    """
    node = Parser()(value)
    return node.polars_expr, node.referenced_columns


def compile_transforms(
    transforms_dict: dict[str, str],
) -> dict[str, pl.Expr]:
    """Compiles a MESSY ``transforms:`` block into Polars expressions.

    Args:
        transforms_dict: A dict mapping derived column names to dftly expression strings.

    Returns:
        A dict mapping column names to compiled Polars expressions.

    Examples:
        >>> exprs = compile_transforms({"total": "$a + $b"})
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
    return Parser.to_polars(transforms_dict)


def compile_subject_id_expr(
    expr_str: str,
) -> tuple[pl.Expr, set[str]]:
    """Compiles a ``subject_id_expr`` into a Polars expression and its referenced columns.

    If the expression uses ``hash()``, the result is reinterpreted as signed Int64
    (since MEDS subject IDs must be Int64).

    Args:
        expr_str: A dftly expression string (e.g., ``"hash($MRN)"``).

    Returns:
        A tuple of (polars_expr, referenced_columns).

    Examples:
        >>> expr, cols = compile_subject_id_expr("hash($mrn)")
        >>> cols
        {'mrn'}
        >>> import polars as pl
        >>> df = pl.DataFrame({"mrn": ["ABC", "DEF"]})
        >>> result = df.select(subject_id=expr)
        >>> result.schema  # doctest: +SKIP
    """
    node = Parser()(expr_str)
    expr = node.polars_expr
    cols = node.referenced_columns

    # hash() produces UInt64 but MEDS needs Int64 — reinterpret bits
    if isinstance(node, Hash):
        expr = expr.reinterpret(signed=True)

    return expr, cols
