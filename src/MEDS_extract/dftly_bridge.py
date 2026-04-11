"""Bridge constants and helpers between MEDS_extract and dftly.

Most dftly functionality (``Parser.expr_to_polars``, ``Parser.to_polars``, ``extract_columns``)
should be called directly. This module provides only MEDS-specific constants and the subject-ID
compilation helper that handles the ``hash()`` → Int64 reinterpret.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dftly import Parser
from dftly.nodes.arithmetic import Hash

if TYPE_CHECKING:
    import polars as pl

# Structural keys in the event config that are not event field definitions.
# All use _ prefix to avoid namespace collisions with event names.
EVENT_META_KEYS = {"_metadata", "_table", "_defaults"}


def compile_subject_id_expr(expr_str: str) -> tuple[pl.Expr, set[str]]:
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
