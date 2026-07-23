"""MESSY event conversion config parsing.

The MESSY config is a small DSL for extracting MEDS events from raw source tables.
It resolves to a list of :class:`TableConfig` entries — each owning a list of
:class:`EventConfig` entries plus the subject_id expression, derived columns, and
join target that apply to the whole table.

All parsing, validation, and polars expression construction happen through
:class:`MessyConfig`. Stages should call :meth:`MessyConfig.load` once and then
consume the parsed class via its methods (``table.extract_events``,
``config.needed_source_columns``, etc.) — they should never traverse the raw
dict themselves.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl
from dftly import Parser
from dftly.nodes.arithmetic import Hash
from dftly.nodes.base import NodeBase
from omegaconf import DictConfig, OmegaConf

from .io import resolve_source_files, scan_source

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from upath import UPath

logger = logging.getLogger(__name__)


def _null_safe_code_expr(code_node: NodeBase) -> pl.Expr:
    """Compile a composite ``code`` so each null component renders as the literal ``"UNK"``.

    dftly string interpolation null-propagates, so a null in any referenced component would make
    the whole ``code`` null (which MEDS forbids). This rebuilds the interpolation from the dftly
    node's ordered pieces, casting each to a string and filling nulls with ``"UNK"`` — restoring
    the pre-dftly ``pl.col(c).cast(pl.Utf8).fill_null("UNK")`` behaviour (including non-string
    columns, which are cast to their string form). It is scoped to the code expression, so
    ``code_components`` keeps the raw, typed values. See issue #109.
    """
    if type(code_node).__name__ == "StringInterpolate":
        template = code_node.args[0].args[0]  # e.g. "LAB//{}//{}"
        pieces = [arg.polars_expr for arg in code_node.args[1:]]
        if isinstance(template, str) and template.count("{}") == len(pieces):
            filled = [p.cast(pl.Utf8, strict=False).fill_null(pl.lit("UNK")) for p in pieces]
            return pl.format(template, *filled)
    # Unexpected node shape: at least guarantee the whole code is never null.
    return code_node.polars_expr.cast(pl.Utf8, strict=False).fill_null(pl.lit("UNK"))


# ── JoinConfig ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class JoinConfig:
    """Parsed left-join configuration for a single table.

    A join must always pull in at least one column from the joined table
    (``cols`` is required) — a join without ``cols`` would be a no-op. The
    MESSY syntax has two forms:

    Short form, for the common case of a shared key column::

        join: {stays: {key: stay_id, cols: [patient_id, dischtime]}}

    Long form, when the key columns differ::

        join: {admissions: {left_on: hadm_id, right_on: admission_id, cols: [dischtime]}}

    Examples:
        >>> JoinConfig.parse({"stays": {"key": "stay_id", "cols": ["subject_id"]}})
        JoinConfig(input_prefix='stays', left_on='stay_id', right_on='stay_id', cols=('subject_id',))
        >>> JoinConfig.parse(
        ...     {"admissions": {"left_on": "hadm_id", "right_on": "adm_id", "cols": ["dischtime"]}}
        ... )
        JoinConfig(input_prefix='admissions', left_on='hadm_id', right_on='adm_id', cols=('dischtime',))
        >>> JoinConfig.parse({"a": {}, "b": {}})
        Traceback (most recent call last):
            ...
        ValueError: Join config must have exactly one key (the input prefix), got: ['a', 'b']
        >>> JoinConfig.parse({"stays": {"cols": ["dischtime"]}})
        Traceback (most recent call last):
            ...
        ValueError: Join config for 'stays' must specify either 'key' or both 'left_on' and 'right_on'.
        >>> JoinConfig.parse({"stays": {"key": "stay_id"}})
        Traceback (most recent call last):
            ...
        ValueError: Join config for 'stays' must pull in at least one column via 'cols'.
    """

    input_prefix: str
    left_on: str
    right_on: str
    cols: tuple[str, ...]

    def __post_init__(self):
        if not self.cols:
            raise ValueError(
                f"Join config for '{self.input_prefix}' must pull in at least one column via 'cols'."
            )

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> JoinConfig:
        if not isinstance(raw, dict) or len(raw) != 1:
            got = sorted(raw.keys()) if isinstance(raw, dict) else raw
            raise ValueError(f"Join config must have exactly one key (the input prefix), got: {got}")

        input_prefix, inner = next(iter(raw.items()))
        if not isinstance(inner, dict):
            raise ValueError(
                f"Join config for '{input_prefix}' must be a mapping with 'key'/'left_on'+'right_on' "
                f"and 'cols', got {type(inner).__name__}."
            )

        if "key" in inner:
            left_on = right_on = inner["key"]
        elif "left_on" in inner and "right_on" in inner:
            left_on = inner["left_on"]
            right_on = inner["right_on"]
        else:
            raise ValueError(
                f"Join config for '{input_prefix}' must specify either 'key' or both "
                f"'left_on' and 'right_on'."
            )

        cols_raw = inner.get("cols", ())
        if not isinstance(cols_raw, list | tuple) or any(not isinstance(c, str) for c in cols_raw):
            raise ValueError(
                f"Join config for '{input_prefix}' must specify 'cols' as a list of column-name "
                f"strings, got {type(cols_raw).__name__}: {cols_raw!r}. A bare string like "
                f"'cols: subject_id' would silently be treated as a tuple of characters — "
                f"use 'cols: [subject_id]' instead."
            )

        return cls(
            input_prefix=input_prefix,
            left_on=left_on,
            right_on=right_on,
            cols=tuple(cols_raw),
        )

    def apply(self, left: pl.LazyFrame, input_dir: Path | UPath) -> pl.LazyFrame:
        """Scan join-target files under ``input_dir`` and left-join them to ``left``.

        File resolution goes through :func:`MEDS_extract.io.resolve_source_files`,
        so every stage that applies a join uses the same layout-detection logic
        as the stages that read the main table.
        """
        right = scan_source(resolve_source_files(input_dir, self.input_prefix))
        return left.join(right, left_on=self.left_on, right_on=self.right_on, how="left")


# ── EventConfig ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class EventConfig:
    """A single MEDS event extraction config.

    ``columns`` maps each output column name to its parsed dftly node. ``"code"``
    is mandatory; ``"time"`` is optional and may be ``None`` (static event).
    Every non-None value is a :class:`dftly.nodes.base.NodeBase` instance —
    :meth:`parse` handles converting raw input (strings or expanded dftly
    dicts) into nodes.

    ``metadata`` holds the raw ``_metadata`` block, or ``{}`` if absent. It's
    consumed separately by ``extract_code_metadata``.

    Examples:
        >>> from dftly import Parser
        >>> p = Parser()
        >>> ev = EventConfig(
        ...     name="lab",
        ...     columns={"code": p('f"{$test}//{$units}"'),
        ...              "time": p('$ts::"%Y-%m-%d"'),
        ...              "numeric_value": p("$result")},
        ... )
        >>> ev.is_static
        False
        >>> sorted(ev.referenced_columns)
        ['result', 'test', 'ts', 'units']
        >>> static = EventConfig(name="eye_color", columns={"code": p('"EYE_COLOR"'), "time": None})
        >>> static.is_static
        True

        Direct construction validates the same invariants as :meth:`parse`:
        missing ``code``, per-event ``subject_id``, and non-node column values
        all raise on instantiation.

        >>> EventConfig(name="bad", columns={"time": None})
        Traceback (most recent call last):
            ...
        KeyError: "Event 'bad' must contain a 'code' key. Got: [time]."
        >>> EventConfig(name="bad", columns={"code": p("X"), "subject_id": p("$sid")})
        Traceback (most recent call last):
            ...
        ValueError: Event 'bad' contains a 'subject_id' key. subject_id is a table-level concept ...
        >>> EventConfig(name="bad", columns={"code": "X"})
        Traceback (most recent call last):
            ...
        TypeError: Event 'bad' column 'code' must be a parsed dftly node, got str ('X').
    """

    name: str
    columns: dict[str, NodeBase | None]
    metadata: dict = field(default_factory=dict)
    raw_code: str | None = None

    def __post_init__(self):
        if "code" not in self.columns:
            raise KeyError(
                f"Event '{self.name}' must contain a 'code' key. Got: [{', '.join(self.columns.keys())}]."
            )
        if "subject_id" in self.columns:
            raise ValueError(
                f"Event '{self.name}' contains a 'subject_id' key. subject_id is a table-level "
                f"concept and must be set in '_defaults', not per-event. See MEDS_extract #73."
            )
        for k, v in self.columns.items():
            if k == "time" and v is None:
                continue
            if not isinstance(v, NodeBase):
                raise TypeError(
                    f"Event '{self.name}' column '{k}' must be a parsed dftly node, "
                    f"got {type(v).__name__} ({v!r})."
                )

    @classmethod
    def parse(cls, name: str, raw: Mapping[str, Any]) -> EventConfig:
        """Parse a raw event block into an EventConfig.

        Each column value is compiled through :class:`dftly.Parser`, so raw
        input may be either a dftly expression string (``"$col"``,
        ``'f"PREFIX//{$col}"'``, ``"hash($col)"``) or an expanded dftly dict
        form. The time column is the only key that may be ``None`` — a
        ``None`` time produces a static event.

        Examples:
            Strings get parsed to nodes:

            >>> ev = EventConfig.parse("lab", {"code": "X", "time": None, "numeric_value": "$v"})
            >>> type(ev.columns["code"]).__name__
            'Literal'
            >>> type(ev.columns["numeric_value"]).__name__
            'Column'
            >>> ev.columns["time"] is None
            True

            Validation errors surface at parse time with the event name:

            >>> EventConfig.parse("bad", {"time": None})
            Traceback (most recent call last):
                ...
            KeyError: "Event 'bad' must contain a 'code' key. Got: [time]."
            >>> EventConfig.parse("bad", {"code": "X", "subject_id": "$sid"})
            Traceback (most recent call last):
                ...
            ValueError: Event 'bad' contains a 'subject_id' key. subject_id is a table-level concept ...
        """
        raw = dict(raw)
        metadata = dict(raw.pop("_metadata", {}))

        raw_code = raw["code"] if isinstance(raw.get("code"), str) else None

        columns: dict[str, NodeBase | None] = {}
        parser = Parser()
        for k, v in raw.items():
            if k == "time" and v is None:
                columns[k] = None
                continue
            try:
                columns[k] = v if isinstance(v, NodeBase) else parser(v)
            except Exception as e:
                raise ValueError(f"Event '{name}' column '{k}' failed to parse: {e}") from e

        return cls(name=name, columns=columns, metadata=metadata, raw_code=raw_code)

    @property
    def is_static(self) -> bool:
        """True if this event has no time column (time absent or ``None``)."""
        return self.columns.get("time") is None

    @cached_property
    def polars_exprs(self) -> dict[str, pl.Expr]:
        """Polars expression for each output column. Built once and reused.

        ``"time"`` resolves to a typed null literal for static events.
        """
        out: dict[str, pl.Expr] = {}
        for k, v in self.columns.items():
            if k == "time" and v is None:
                out[k] = pl.lit(None, dtype=pl.Datetime)
            else:
                out[k] = v.polars_expr
        return out

    @cached_property
    def code_source_columns(self) -> frozenset[str]:
        """Source columns referenced by the ``code`` expression."""
        return frozenset(self.columns["code"].referenced_columns)

    @cached_property
    def time_source_columns(self) -> frozenset[str]:
        """Source columns referenced by the ``time`` expression (empty if static)."""
        if self.is_static:
            return frozenset()
        return frozenset(self.columns["time"].referenced_columns)

    @cached_property
    def referenced_columns(self) -> frozenset[str]:
        """All source columns referenced by any output column expression.

        Aggregates across ``code``, ``time``, and all additional value columns.
        Used by :meth:`TableConfig.source_columns` to determine which columns
        must be read from the source parquet file.
        """
        cols: set[str] = set()
        for v in self.columns.values():
            if v is None:
                continue
            cols.update(v.referenced_columns)
        return frozenset(cols)

    def extract(
        self,
        df: pl.LazyFrame,
        source_block: str,
        do_dedup_text_and_numeric: bool = False,
    ) -> pl.LazyFrame:
        """Extract this event's rows from a dataframe already prepared by :meth:`TableConfig.prepare`.

        The input ``df`` must have a ``subject_id`` column and any source columns this
        event references. ``source_block`` tags each output row with its MESSY origin
        (e.g. ``"patients/eye_color"``) and is always included in the output schema.

        Examples:
            >>> _ = pl.Config.set_tbl_width_chars(600)
            >>> raw = pl.DataFrame({
            ...     "subject_id": [1, 2, 3],
            ...     "color": ["blue", "green", "brown"],
            ... })
            >>> ev = EventConfig.parse(
            ...     "eye_color",
            ...     {"code": "EYE_COLOR", "time": None, "eye_color": "$color"},
            ... )
            >>> ev.extract(raw.lazy(), "patients/eye_color").collect()
            shape: (3, 5)
            ┌────────────┬───────────┬──────────────┬───────────┬────────────────────┐
            │ subject_id ┆ code      ┆ time         ┆ eye_color ┆ source_block       │
            │ ---        ┆ ---       ┆ ---          ┆ ---       ┆ ---                │
            │ i64        ┆ str       ┆ datetime[μs] ┆ str       ┆ str                │
            ╞════════════╪═══════════╪══════════════╪═══════════╪════════════════════╡
            │ 1          ┆ EYE_COLOR ┆ null         ┆ blue      ┆ patients/eye_color │
            │ 2          ┆ EYE_COLOR ┆ null         ┆ green     ┆ patients/eye_color │
            │ 3          ┆ EYE_COLOR ┆ null         ┆ brown     ┆ patients/eye_color │
            └────────────┴───────────┴──────────────┴───────────┴────────────────────┘

            A null in a referenced ``time`` column, or in a **bare-column** ``code``
            (``code: $col``), filters the row out before selection:

            >>> raw = pl.DataFrame({
            ...     "subject_id": [1, 2, 3],
            ...     "name": ["A", None, "C"],
            ...     "ts": ["2021-01-01", "2021-01-02", None],
            ... })
            >>> ev = EventConfig.parse("e", {"code": "$name", "time": '$ts::"%Y-%m-%d"'})
            >>> ev.extract(raw.lazy(), "t/e").collect().select("subject_id", "code", "time")
            shape: (1, 3)
            ┌────────────┬──────┬────────────┐
            │ subject_id ┆ code ┆ time       │
            │ ---        ┆ ---  ┆ ---        │
            │ i64        ┆ str  ┆ date       │
            ╞════════════╪══════╪════════════╡
            │ 1          ┆ A    ┆ 2021-01-01 │
            └────────────┴──────┴────────────┘

            The rule above applies to a *bare column* code (``code: $col``): a null value is a
            meaningless event, so the row is dropped. A *composite / interpolated* code behaves
            differently — every null component is rendered as the literal ``"UNK"`` so a missing
            part (e.g. a unit) doesn't discard the whole event, and the code is never null. This
            applies to **every** referenced column, including the leading one even when the code
            has no literal prefix. ``$itemid`` and ``$valueuom`` below cover all four null/non-null
            combinations (see issue #109):

            >>> raw = pl.DataFrame({
            ...     "subject_id": [1, 2, 3, 4],
            ...     "itemid": ["GLU", "GLU", None, None],  # present, present, null, null
            ...     "valueuom": ["mg/dL", None, "mg/dL", None],  # present, null, present, null
            ... })
            >>> ev = EventConfig.parse("lab", {"code": 'f"{$itemid}//{$valueuom}"', "time": None})
            >>> ev.extract(raw.lazy(), "labs/lab").collect().select("subject_id", "code").sort(
            ...     "subject_id"
            ... )
            shape: (4, 2)
            ┌────────────┬────────────┐
            │ subject_id ┆ code       │
            │ ---        ┆ ---        │
            │ i64        ┆ str        │
            ╞════════════╪════════════╡
            │ 1          ┆ GLU//mg/dL │
            │ 2          ┆ GLU//UNK   │
            │ 3          ┆ UNK//mg/dL │
            │ 4          ┆ UNK//UNK   │
            └────────────┴────────────┘

            With ``do_dedup_text_and_numeric=True``, a ``text_value`` that
            numerically equals ``numeric_value`` is nulled out:

            >>> raw = pl.DataFrame({
            ...     "subject_id": [1, 2],
            ...     "ts": ["2021-01-01", "2021-01-02"],
            ...     "val": [1.5, 2.0],
            ...     "text": ["1.5", "other"],
            ... })
            >>> ev = EventConfig.parse("m", {
            ...     "code": "MEAS",
            ...     "time": '$ts::"%Y-%m-%d"',
            ...     "numeric_value": "$val",
            ...     "text_value": "$text",
            ... })
            >>> ev.extract(raw.lazy(), "t/m", do_dedup_text_and_numeric=True).collect().select(
            ...     "numeric_value", "text_value"
            ... )
            shape: (2, 2)
            ┌───────────────┬────────────┐
            │ numeric_value ┆ text_value │
            │ ---           ┆ ---        │
            │ f64           ┆ str        │
            ╞═══════════════╪════════════╡
            │ 1.5           ┆ null       │
            │ 2.0           ┆ other      │
            └───────────────┴────────────┘
        """
        exprs: dict[str, pl.Expr] = {"subject_id": pl.col("subject_id")}

        # A composite code renders each null component as "UNK" (see issue #109); a bare-column
        # or literal code is used as-is.
        if self.code_source_columns and type(self.columns["code"]).__name__ != "Column":
            exprs["code"] = _null_safe_code_expr(self.columns["code"])
        else:
            exprs["code"] = self.polars_exprs["code"]
        if self.code_source_columns:
            # Raw, typed component values (unaffected by the "UNK" rendering above).
            exprs["code_components"] = pl.struct(
                **{col: pl.col(col) for col in sorted(self.code_source_columns)}
            )

        exprs["time"] = self.polars_exprs["time"]

        for k in self.columns:
            if k in ("code", "time"):
                continue
            exprs[k] = self.polars_exprs[k]

        if do_dedup_text_and_numeric and "numeric_value" in exprs and "text_value" in exprs:
            text_expr = exprs["text_value"]
            num_expr = exprs["numeric_value"]
            exprs["text_value"] = (
                pl.when(text_expr.cast(pl.Float32, strict=False) == num_expr.cast(pl.Float32))
                .then(pl.lit(None, pl.String))
                .otherwise(text_expr)
            )

        exprs["source_block"] = pl.lit(source_block)

        if self.code_source_columns and type(self.columns["code"]).__name__ == "Column":
            # A bare-column code is a bare identifier: a null value is a meaningless event (no
            # identifier), so drop the row. A composite code instead renders each null component
            # as "UNK" in the code expression above (restoring the pre-dftly behavior that the
            # dftly migration regressed — dftly interpolation null-propagates). See issue #109.
            (only_col,) = self.code_source_columns
            df = df.filter(pl.col(only_col).is_not_null())

        if not self.is_static:
            # Filter on source columns being non-null/non-empty rather than on the parsed time
            # expression, to avoid a polars predicate-pushdown bug where strptime(strict=True)
            # is evaluated during parquet scanning before nulls are filtered.
            if self.time_source_columns:
                schema = df.collect_schema()
                ts_filters = []
                for c in sorted(self.time_source_columns):
                    col_filter = pl.col(c).is_not_null()
                    if schema.get(c) == pl.String or schema.get(c) is None:
                        col_filter = col_filter & (pl.col(c) != pl.lit(""))
                    ts_filters.append(col_filter)
                df = df.filter(pl.all_horizontal(*ts_filters))
            else:
                df = df.filter(exprs["time"].is_not_null())

        return df.select(**exprs).unique(maintain_order=True)


# ── TableConfig ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class TableConfig:
    """Fully-resolved config for one source table block.

    Global ``_defaults`` are merged with file-level ``_defaults`` at parse time,
    so :attr:`subject_id_node` already reflects the final inherited value.
    ``_table`` sub-keys (``cols``, ``join``) are lifted to top-level fields.

    Examples:
        >>> tc = TableConfig.parse("patients", {
        ...     "_defaults": {"subject_id": "$MRN"},
        ...     "_table": {"join": {"stays": {"key": "stay_id", "cols": ["dischtime"]}}},
        ...     "dob": {"code": "BIRTH", "time": "$dob"},
        ... })
        >>> tc.input_prefix
        'patients'
        >>> type(tc.subject_id_node).__name__
        'Column'
        >>> sorted(tc.subject_id_node.referenced_columns)
        ['MRN']
        >>> tc.join.input_prefix
        'stays'
        >>> [e.name for e in tc.events]
        ['dob']
    """

    input_prefix: str
    subject_id_node: NodeBase | None = None
    cols: dict[str, NodeBase] = field(default_factory=dict)
    join: JoinConfig | None = None
    events: tuple[EventConfig, ...] = ()

    def __post_init__(self):
        if not self.events:
            raise ValueError(
                f"Table '{self.input_prefix}' defines no events. Every table must contain at "
                f"least one event (a dict with at minimum a 'code' key)."
            )
        if self.subject_id_node is not None and not isinstance(self.subject_id_node, NodeBase):
            raise TypeError(
                f"Table '{self.input_prefix}' subject_id_node must be a parsed dftly node, "
                f"got {type(self.subject_id_node).__name__}."
            )
        for k, v in self.cols.items():
            if not isinstance(v, NodeBase):
                raise TypeError(
                    f"Table '{self.input_prefix}' derived column '{k}' must be a parsed dftly "
                    f"node, got {type(v).__name__}."
                )

    @classmethod
    def parse(
        cls,
        input_prefix: str,
        raw: Mapping[str, Any],
        global_defaults: Mapping[str, Any] | None = None,
    ) -> TableConfig:
        raw = dict(raw)
        global_defaults = dict(global_defaults or {})

        file_defaults = dict(raw.pop("_defaults", {}))
        merged_defaults = {**global_defaults, **file_defaults}

        table_cfg = dict(raw.pop("_table", {}))
        unknown_table_keys = set(table_cfg) - {"cols", "join"}
        if unknown_table_keys:
            raise ValueError(
                f"Table '{input_prefix}' has unknown keys under '_table': "
                f"{sorted(unknown_table_keys)}. Allowed keys: 'cols', 'join'."
            )
        parser = Parser()

        raw_cols = dict(table_cfg.get("cols", {}))
        cols = {k: v if isinstance(v, NodeBase) else parser(v) for k, v in raw_cols.items()}

        join = JoinConfig.parse(dict(table_cfg["join"])) if "join" in table_cfg else None

        subject_id_raw = merged_defaults.get("subject_id")
        subject_id_node: NodeBase | None
        if subject_id_raw is None:
            subject_id_node = None
        elif isinstance(subject_id_raw, NodeBase):
            subject_id_node = subject_id_raw
        else:
            subject_id_node = parser(subject_id_raw)

        events = tuple(EventConfig.parse(event_name, event_raw) for event_name, event_raw in raw.items())

        return cls(
            input_prefix=input_prefix,
            subject_id_node=subject_id_node,
            cols=cols,
            join=join,
            events=events,
        )

    @cached_property
    def subject_id_polars_expr(self) -> pl.Expr:
        """Polars expression producing the MEDS ``subject_id`` column.

        Always produces ``Int64`` output, per the MEDS schema. When no explicit
        subject_id expression is configured, the expression reads the existing
        ``subject_id`` column from the source table and casts it to Int64.
        Non-hash expressions are cast via :meth:`polars.Expr.cast` in **strict
        mode** — values that can't be converted (e.g., unparsable strings)
        raise at query time rather than silently becoming nulls. ``hash()``
        outputs (UInt64) are reinterpreted to preserve bits (tracked upstream
        in dftly#57; the hash value is a bit-reinterpret of polars' hash, not
        a fresh signed hash, so external systems computing hashes won't get
        bit-compatible values).

        The returned expression is never ``None`` — stages can apply it
        unconditionally.

        Examples:
            No explicit subject_id expression: read the existing ``subject_id``
            column (cast to Int64 for schema compliance).

            >>> import polars as pl
            >>> tc = TableConfig.parse("t", {"e": {"code": "X", "time": None}})
            >>> df = pl.DataFrame({"subject_id": [1, 2]}, schema={"subject_id": pl.Int32})
            >>> df.select(subject_id=tc.subject_id_polars_expr).schema["subject_id"]
            Int64

            Column-reference expression: Int32 source → Int64 output.

            >>> tc = TableConfig.parse("t", {"_defaults": {"subject_id": "$patient_id"},
            ...                              "e": {"code": "X", "time": None}})
            >>> df = pl.DataFrame({"patient_id": [1, 2]}, schema={"patient_id": pl.Int32})
            >>> df.select(subject_id=tc.subject_id_polars_expr).schema["subject_id"]
            Int64

            ``hash()`` produces UInt64; we reinterpret to Int64.

            >>> tc = TableConfig.parse("t", {"_defaults": {"subject_id": "hash($mrn)"},
            ...                              "e": {"code": "X", "time": None}})
            >>> df = pl.DataFrame({"mrn": ["ABC", "DEF"]})
            >>> df.select(subject_id=tc.subject_id_polars_expr).schema["subject_id"]
            Int64
        """
        if self.subject_id_node is None:
            return pl.col("subject_id").cast(pl.Int64, strict=True)
        expr = self.subject_id_node.polars_expr
        if isinstance(self.subject_id_node, Hash):
            return expr.reinterpret(signed=True)
        return expr.cast(pl.Int64, strict=True)

    @property
    def col_outputs(self) -> set[str]:
        """Column names produced by ``_table.cols`` — derived, not read from source."""
        return set(self.cols.keys())

    @property
    def joined_columns(self) -> set[str]:
        """Column names that come from the joined table, not the source file."""
        return set(self.join.cols) if self.join is not None else set()

    def source_columns(self) -> set[str]:
        """Columns that must be read from this table's source parquet file.

        Aggregates the subject_id source columns, the join left key, derived-column
        inputs, and all event-referenced columns — minus columns produced by
        ``_table.cols`` (derived) or pulled in by the join (come from elsewhere).

        Examples:
            >>> tc = TableConfig.parse("labs", {
            ...     "_defaults": {"subject_id": "$patient_id"},
            ...     "_table": {
            ...         "cols": {"year": "$anchor_year - $anchor_age"},
            ...         "join": {"stays": {"key": "stay_id", "cols": ["dischtime"]}},
            ...     },
            ...     "lab": {"code": "$test", "time": "$dischtime"},
            ... })
            >>> sorted(tc.source_columns())
            ['anchor_age', 'anchor_year', 'patient_id', 'stay_id', 'test']
        """
        cols: set[str] = set()
        if self.subject_id_node is not None:
            cols.update(self.subject_id_node.referenced_columns)
        else:
            cols.add("subject_id")

        for node in self.cols.values():
            cols.update(node.referenced_columns)

        if self.join is not None:
            cols.add(self.join.left_on)

        for event in self.events:
            cols.update(event.referenced_columns)

        return cols - self.col_outputs - self.joined_columns

    def source_files(self, dir: Path | UPath) -> list[Path | UPath]:
        """Resolve the list of source files for this table under ``dir``.

        Delegates to :func:`MEDS_extract.io.resolve_source_files`. Returns a
        non-empty list; raises ``FileNotFoundError`` or ``ValueError`` (on
        layout ambiguity) otherwise.
        """
        return resolve_source_files(dir, self.input_prefix)

    def scan(self, dir: Path | UPath, **scan_kwargs: Any) -> pl.LazyFrame:
        """Scan every source file for this table under ``dir``, apply the join.

        The unified entry point that every stage should use to read a table's data. Auto-detects the layout
        (bare file vs sub-sharded directory), dispatches on format, and applies the join if configured.
        """
        df = scan_source(self.source_files(dir), **scan_kwargs)
        if self.join is not None:
            df = self.join.apply(df, dir)
        return df

    def prepare(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Apply subject_id materialization and derived columns to a raw dataframe.

        The result has an ``Int64`` ``subject_id`` column and any ``_table.cols``
        derived columns. It does **not** apply the join — joins need a
        stage-specific input directory for the join target and are applied by
        the caller via :meth:`JoinConfig.apply`.

        Derived columns are applied in insertion order via a chain of
        ``with_columns`` calls, so a later entry may reference a name defined
        above it in the same ``_table.cols`` block. A forward reference (a
        column referencing a name defined below it) fails at query time with
        polars' standard missing-column error — intentionally not special-cased.

        Examples:
            >>> _ = pl.Config.set_tbl_width_chars(600)
            >>> tc = TableConfig.parse("t", {
            ...     "_defaults": {"subject_id": "$MRN"},
            ...     "_table": {"cols": {"year": "$anchor_year - $anchor_age"}},
            ...     "e": {"code": "X", "time": None},
            ... })
            >>> raw = pl.DataFrame({
            ...     "MRN": [100, 200],
            ...     "anchor_year": [2020, 2021],
            ...     "anchor_age": [30, 25],
            ... })
            >>> tc.prepare(raw.lazy()).collect().select("subject_id", "year")
            shape: (2, 2)
            ┌────────────┬──────┐
            │ subject_id ┆ year │
            │ ---        ┆ ---  │
            │ i64        ┆ i64  │
            ╞════════════╪══════╡
            │ 100        ┆ 1990 │
            │ 200        ┆ 1996 │
            └────────────┴──────┘

            Later ``_table.cols`` entries can reference earlier ones — the
            motivating case is offset-time schemas (eICU, HIRID, ...) where
            pseudotimes chain off each other. Here ``age_at_admit`` uses the
            ``year_of_birth`` defined on the line above:

            >>> tc = TableConfig.parse("t", {
            ...     "_defaults": {"subject_id": "$MRN"},
            ...     "_table": {"cols": {
            ...         "year_of_birth": "$anchor_year - $anchor_age",
            ...         "age_at_admit":  "$admit_year - $year_of_birth",
            ...     }},
            ...     "e": {"code": "X", "time": None},
            ... })
            >>> raw = pl.DataFrame({
            ...     "MRN": [100, 200],
            ...     "anchor_year": [2020, 2021],
            ...     "anchor_age": [30, 25],
            ...     "admit_year": [2024, 2024],
            ... })
            >>> tc.prepare(raw.lazy()).collect().select(
            ...     "subject_id", "year_of_birth", "age_at_admit"
            ... )
            shape: (2, 3)
            ┌────────────┬───────────────┬──────────────┐
            │ subject_id ┆ year_of_birth ┆ age_at_admit │
            │ ---        ┆ ---           ┆ ---          │
            │ i64        ┆ i64           ┆ i64          │
            ╞════════════╪═══════════════╪══════════════╡
            │ 100        ┆ 1990          ┆ 34           │
            │ 200        ┆ 1996          ┆ 28           │
            └────────────┴───────────────┴──────────────┘
        """
        df = df.with_columns(subject_id=self.subject_id_polars_expr)
        for name, node in self.cols.items():
            df = df.with_columns(node.polars_expr.alias(name))
        return df

    def extract_events(
        self,
        df: pl.LazyFrame,
        do_dedup_text_and_numeric: bool = False,
    ) -> pl.LazyFrame:
        """Prepare ``df`` and extract every event in this table, concatenated.

        Each event's output rows are tagged with a ``source_block`` column derived
        from ``f"{input_prefix}/{event.name}"``.

        Raises:
            ValueError: if extracting any individual event fails (the table + event
                name are included in the error).

        Examples:
            >>> _ = pl.Config.set_tbl_width_chars(600)
            >>> tc = TableConfig.parse("data", {
            ...     "admit": {"code": 'f"ADMIT//{$dept}"', "time": '$ts::"%Y-%m-%d"'},
            ...     "color": {"code": "EYE_COLOR", "time": None, "eye_color": "$color"},
            ... })
            >>> raw = pl.DataFrame({
            ...     "subject_id": [1, 2],
            ...     "dept": ["CARDIAC", "PULM"],
            ...     "ts": ["2021-01-01", "2021-01-02"],
            ...     "color": ["blue", "green"],
            ... })
            >>> tc.extract_events(raw.lazy()).collect().select(
            ...     "subject_id", "code", "source_block"
            ... ).sort("source_block", "subject_id")
            shape: (4, 3)
            ┌────────────┬────────────────┬──────────────┐
            │ subject_id ┆ code           ┆ source_block │
            │ ---        ┆ ---            ┆ ---          │
            │ i64        ┆ str            ┆ str          │
            ╞════════════╪════════════════╪══════════════╡
            │ 1          ┆ ADMIT//CARDIAC ┆ data/admit   │
            │ 2          ┆ ADMIT//PULM    ┆ data/admit   │
            │ 1          ┆ EYE_COLOR      ┆ data/color   │
            │ 2          ┆ EYE_COLOR      ┆ data/color   │
            └────────────┴────────────────┴──────────────┘
        """
        df = self.prepare(df)

        event_dfs = []
        for event in self.events:
            source_block = f"{self.input_prefix}/{event.name}"
            try:
                logger.info(f"Building extraction plan for {source_block}")
                event_dfs.append(
                    event.extract(
                        df, source_block=source_block, do_dedup_text_and_numeric=do_dedup_text_and_numeric
                    )
                )
            except Exception as e:
                raise ValueError(f"Error extracting event {source_block}: {e}") from e

        return pl.concat(event_dfs, how="diagonal_relaxed")


# ── MessyConfig ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class MessyConfig:
    """Top-level parsed MESSY event-conversion config.

    Single entry point for stages: call :meth:`load` once with the config file
    path, then query the class methods instead of re-traversing the raw dict.

    Examples:
        >>> cfg = MessyConfig.parse({
        ...     "_defaults": {"subject_id": "$MRN"},
        ...     "patients": {"dob": {"code": "BIRTH", "time": "$dob"}},
        ...     "labs": {
        ...         "_defaults": {"subject_id": "$patient_id"},
        ...         "_table": {"join": {"stays": {"key": "stay_id", "cols": ["patient_id"]}}},
        ...         "lab": {"code": "$test", "time": "$ts"},
        ...     },
        ... })
        >>> cfg.table_prefixes
        ['patients', 'labs']
        >>> sorted(cfg.tables[0].subject_id_node.referenced_columns)
        ['MRN']
        >>> sorted(cfg.tables[1].subject_id_node.referenced_columns)
        ['patient_id']
        >>> [e.name for e in cfg.iter_events()]
        ['dob', 'lab']
    """

    tables: tuple[TableConfig, ...]
    source_fp: Path | None = None

    # Top-level keys that are NOT event-table definitions and should be ignored here.
    # ``_defaults`` is consumed separately below as the global defaults; this set is
    # strictly for siblings that the ``meds-extract-download`` CLI (or future adjacent
    # tools) drops into the same MESSY file so one file carries everything for a
    # dataset. Currently just ``sources``; new entries join the set without a code
    # change below.
    _IGNORED_TOP_LEVEL_KEYS: ClassVar[frozenset[str]] = frozenset({"sources"})

    @classmethod
    def parse(cls, raw: Mapping[str, Any] | DictConfig) -> MessyConfig:
        """Parse a raw MESSY mapping into a :class:`MessyConfig`.

        Reserved sibling keys (``sources``, consumed only by
        ``meds-extract-download``) are stripped before interpolation resolution
        and before table parsing. A config with no event tables left after
        stripping is an error — most commonly a sources-only file passed to the
        event-conversion pipeline by mistake:

        >>> MessyConfig.parse({"sources": {"dataset": []}})
        Traceback (most recent call last):
            ...
        ValueError: MESSY config defines no event tables ...
        """
        if OmegaConf.is_config(raw):
            # Strip ignored reserved keys BEFORE ``resolve=True`` so ``${oc.env:...}``
            # interpolations inside a ``sources:`` block (only needed by
            # ``meds-extract-download``) don't require those env vars to be set just
            # to load the event-conversion config.
            raw = OmegaConf.create(raw)
            for key in cls._IGNORED_TOP_LEVEL_KEYS:
                if key in raw:
                    del raw[key]
            raw = OmegaConf.to_container(raw, resolve=True)
        raw_dict = dict(raw)
        global_defaults = dict(raw_dict.pop("_defaults", {}))
        # Non-DictConfig (plain dict) callers still need the ignored-key filter.
        for key in cls._IGNORED_TOP_LEVEL_KEYS:
            raw_dict.pop(key, None)

        if not raw_dict:
            # A sources-only (or _defaults-only) file would otherwise parse to an
            # empty config: shard_events no-ops "successfully" and the pipeline
            # dies two stages later inside polars with no hint of the real
            # mistake. Fail here, where the cause is nameable.
            raise ValueError(
                "MESSY config defines no event tables (found only reserved keys: "
                f"{sorted({'_defaults', *cls._IGNORED_TOP_LEVEL_KEYS})}). A file carrying only a "
                "'sources:' block can drive `meds-extract-download`, but the event-conversion "
                "pipeline needs a MESSY file with event-table definitions."
            )

        tables = tuple(
            TableConfig.parse(prefix, block, global_defaults) for prefix, block in raw_dict.items()
        )
        return cls(tables=tables)

    @classmethod
    def load(cls, fp: Path | str) -> MessyConfig:
        """Load, validate, and parse a MESSY config file.

        Handles existence check, OmegaConf loading, logging, and parsing in one
        call. All stages should use this rather than calling ``OmegaConf.load``
        directly, so logging stays consistent. The source file path is
        remembered so :meth:`save` can copy the original YAML verbatim.

        Examples:
            >>> yaml = '''
            ... _defaults: {subject_id: $MRN}
            ... patients:
            ...   dob: {code: BIRTH, time: null}
            ... '''
            >>> cfg_fp = getfixture("tmp_path") / "cfg.yaml"
            >>> _ = cfg_fp.write_text(yaml)
            >>> cfg = MessyConfig.load(cfg_fp)
            >>> cfg.table_prefixes
            ['patients']
            >>> type(cfg.tables[0].subject_id_node).__name__
            'Column'

            Loading from a missing file raises with a clear error:

            >>> MessyConfig.load(cfg_fp.parent / "missing.yaml")
            Traceback (most recent call last):
                ...
            FileNotFoundError: Event conversion config file not found: ...missing.yaml
        """
        fp = Path(fp)
        if not fp.exists():
            raise FileNotFoundError(f"Event conversion config file not found: {fp}")
        logger.info(f"Reading event conversion config from {fp}")
        raw = OmegaConf.load(fp)
        # Log with reserved keys stripped: a combined-MESSY ``sources:`` block can
        # carry credentials (literal API keys / passwords), which must not land in
        # every stage's log output.
        loggable = OmegaConf.create(raw)
        for key in cls._IGNORED_TOP_LEVEL_KEYS:
            if key in loggable:
                del loggable[key]
        logger.info(f"Event conversion config:\n{OmegaConf.to_yaml(loggable)}")
        parsed = cls.parse(raw)
        # Attach the source path so `.save()` can verbatim-copy the original.
        object.__setattr__(parsed, "source_fp", fp)
        return parsed

    def save(self, fp: Path | UPath | str) -> None:
        """Copy the original MESSY config file to ``fp``, minus reserved keys.

        Only valid on instances produced by :meth:`load` (which remembers the
        source path). Instances built via :meth:`parse` directly don't have a
        source file to copy and will raise. Uses ``read_bytes`` / ``write_bytes``
        so UPath-backed cloud destinations work as well as local paths.

        When the source file carries reserved sibling blocks (``sources:``), the
        copy is re-serialized with those blocks stripped — a combined-MESSY
        ``sources:`` block can carry credentials, and this copy lands inside the
        (often shared) pipeline output tree. Comment formatting is preserved only
        for files with no reserved blocks, where a verbatim byte-copy suffices.

        Examples:
            >>> yaml = '''
            ... sources:
            ...   dataset:
            ...     - type: http
            ...       headers: {X-Dataverse-key: super-secret-token}
            ...       urls: [https://example.com/x.csv]
            ... patients:
            ...   dob: {code: BIRTH, time: null}
            ... '''
            >>> cfg_fp = getfixture("tmp_path") / "cfg.yaml"
            >>> _ = cfg_fp.write_text(yaml)
            >>> out_fp = getfixture("tmp_path") / "copy.yaml"
            >>> MessyConfig.load(cfg_fp).save(out_fp)
            >>> print(out_fp.read_text().strip())
            patients:
              dob:
                code: BIRTH
                time: null
            >>> "super-secret-token" in out_fp.read_text()
            False
        """
        if self.source_fp is None:
            raise ValueError("MessyConfig.save requires a source file path (only available after .load()).")
        if not self.source_fp.exists():
            raise FileNotFoundError(
                f"MessyConfig source file no longer exists at {self.source_fp}; cannot copy to {fp}."
            )
        dest = Path(fp) if isinstance(fp, str) else fp
        raw = OmegaConf.load(self.source_fp)
        reserved_present = [k for k in self._IGNORED_TOP_LEVEL_KEYS if k in raw]
        if not reserved_present:
            dest.write_bytes(self.source_fp.read_bytes())
            return
        for key in reserved_present:
            del raw[key]
        # ``to_yaml`` does not resolve interpolations, so symbolic ``${oc.env:...}``
        # references in the event-conversion sections survive the round-trip.
        dest.write_bytes(OmegaConf.to_yaml(raw).encode("utf-8"))

    def iter_tables(self) -> Iterator[TableConfig]:
        return iter(self.tables)

    def iter_events(self) -> Iterator[EventConfig]:
        for table in self.tables:
            yield from table.events

    def shuffled_tables(self, seed: int | None = None) -> list[TableConfig]:
        """Return tables in randomized order.

        Used by stages that iterate tables to spread parallel worker load — without shuffling, every worker
        would contend on the same first table.
        """
        rng = random.Random(seed)
        tables = list(self.tables)
        rng.shuffle(tables)
        return tables

    @property
    def table_prefixes(self) -> list[str]:
        return [t.input_prefix for t in self.tables]

    def needed_source_columns(self) -> dict[str, list[str]]:
        """Map each source prefix to the sorted list of columns that must be read.

        Aggregates each table's own :meth:`TableConfig.source_columns` plus the
        columns that any table pulls in from a join target. The returned dict is
        the input to ``shard_events`` — it tells that stage which columns to
        project when subsharding raw inputs.

        Examples:
            >>> cfg = MessyConfig.parse({
            ...     "_defaults": {"subject_id": "$subject_id_global"},
            ...     "hosp/patients": {
            ...         "eye_color": {"code": "EYE_COLOR", "time": None},
            ...         "height": {"code": "HEIGHT", "time": None, "numeric_value": "$height"},
            ...     },
            ...     "icu/chartevents": {
            ...         "_defaults": {"subject_id": "$subject_id_icu"},
            ...         "heart_rate": {
            ...             "code": "HEART_RATE", "time": "$charttime", "numeric_value": "$HR"
            ...         },
            ...     },
            ... })
            >>> cfg.needed_source_columns()
            {'hosp/patients': ['height', 'subject_id_global'],
             'icu/chartevents': ['HR', 'charttime', 'subject_id_icu']}

            Derived columns and joined columns are excluded from the source file's
            needed-column list (they come from elsewhere); the join target gets
            its own entry:

            >>> cfg = MessyConfig.parse({
            ...     "labs": {
            ...         "_defaults": {"subject_id": "$patient_id"},
            ...         "_table": {"join": {"stays": {"key": "stay_id", "cols": ["dischtime"]}}},
            ...         "lab": {"code": "$test", "time": "$dischtime"},
            ...     },
            ... })
            >>> cfg.needed_source_columns()
            {'labs': ['patient_id', 'stay_id', 'test'], 'stays': ['dischtime', 'stay_id']}
        """
        out: dict[str, set[str]] = {}
        for table in self.tables:
            out.setdefault(table.input_prefix, set()).update(table.source_columns())
            if table.join is not None:
                jt = out.setdefault(table.join.input_prefix, set())
                jt.add(table.join.right_on)
                jt.update(table.join.cols)
        return {k: sorted(v) for k, v in out.items()}

    def events_by_metadata_prefix(self) -> dict[str, list[dict]]:
        """Invert the events → metadata mapping.

        Each event's ``_metadata`` block maps metadata-file prefixes to
        per-prefix metadata config dicts. This returns the reverse: each
        metadata prefix gets the list of ``{code, _metadata, source_block}``
        entries that reference it. The ``code`` value is the original raw
        dftly expression string when available (so downstream
        ``code_template`` columns stay human-readable), falling back to the
        parsed node otherwise. The ``source_block`` value is the
        ``{input_prefix}/{event_name}`` tag that :meth:`EventConfig.extract`
        stamps on every output row — ``extract_code_metadata`` uses it to
        scope partial-match (``_match_on``) expansions to the event that
        declared the ``_metadata`` block (see issue #134).

        Used by ``extract_code_metadata``.

        Examples:
            >>> cfg = MessyConfig.parse({
            ...     "_defaults": {"subject_id": "$MRN"},
            ...     "icu/procedureevents": {
            ...         "_defaults": {"subject_id": "$subject_id"},
            ...         "start": {
            ...             "code": 'f"PROC//START//{$itemid}"',
            ...             "_metadata": {
            ...                 "proc_datetimeevents": {"desc": ["omop_concept_name", "label"]},
            ...             },
            ...         },
            ...     },
            ... })
            >>> grouped = cfg.events_by_metadata_prefix()
            >>> sorted(grouped.keys())
            ['proc_datetimeevents']
            >>> entry = grouped["proc_datetimeevents"][0]
            >>> entry["code"]
            'f"PROC//START//{$itemid}"'
            >>> entry["source_block"]
            'icu/procedureevents/start'
            >>> MessyConfig.parse({"t": {"e": {"code": "X", "time": None}}}).events_by_metadata_prefix()
            {}
        """
        out: dict[str, list[dict]] = {}
        for table in self.tables:
            for event in table.events:
                code: str | NodeBase = event.raw_code if event.raw_code is not None else event.columns["code"]
                source_block = f"{table.input_prefix}/{event.name}"
                for metadata_prefix, metadata_cfg in event.metadata.items():
                    out.setdefault(metadata_prefix, []).append(
                        {"code": code, "_metadata": metadata_cfg, "source_block": source_block}
                    )
        return out
