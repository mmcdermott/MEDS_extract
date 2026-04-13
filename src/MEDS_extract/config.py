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
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from dftly import Parser, extract_columns
from dftly.nodes.arithmetic import Hash
from meds import DataSchema
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from upath import UPath

logger = logging.getLogger(__name__)

# Structural keys in the YAML config that are not event names. All use the ``_``
# prefix to avoid colliding with user-chosen event names in the same dict.
TABLE_META_KEYS = frozenset({"_table", "_defaults"})
EVENT_META_KEYS = frozenset({"_metadata"})


def _to_plain(obj: Any) -> Any:
    """Convert an OmegaConf container to a plain Python dict/list, else pass through."""
    if OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


# ── JoinConfig ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class JoinConfig:
    """Parsed left-join configuration for a single table.

    The MESSY syntax supports three equivalent forms:

    Minimal (shared key, no extra columns)::

        join: {stays: stay_id}

    Short form (shared key, with extra columns to pull in)::

        join: {stays: {key: stay_id, cols: [patient_id, dischtime]}}

    Long form (different key names on each side)::

        join: {admissions: {left_on: hadm_id, right_on: admission_id}}

    Examples:
        >>> JoinConfig.parse({"stays": "stay_id"})
        JoinConfig(input_prefix='stays', left_on='stay_id', right_on='stay_id', cols=())
        >>> JoinConfig.parse({"stays": {"key": "stay_id", "cols": ["subject_id"]}})
        JoinConfig(input_prefix='stays', left_on='stay_id', right_on='stay_id', cols=('subject_id',))
        >>> JoinConfig.parse({"admissions": {"left_on": "hadm_id", "right_on": "admission_id"}})
        JoinConfig(input_prefix='admissions', left_on='hadm_id', right_on='admission_id', cols=())
        >>> JoinConfig.parse({"a": {}, "b": {}})
        Traceback (most recent call last):
            ...
        ValueError: Join config must have exactly one key (the input prefix), got: ['a', 'b']
        >>> JoinConfig.parse({"stays": {"cols": ["dischtime"]}})
        Traceback (most recent call last):
            ...
        ValueError: Join config for 'stays' must specify either 'key' or both 'left_on' and 'right_on'.
    """

    input_prefix: str
    left_on: str
    right_on: str
    cols: tuple[str, ...] = ()

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> JoinConfig:
        raw = _to_plain(raw)
        if not isinstance(raw, dict) or len(raw) != 1:
            got = sorted(raw.keys()) if isinstance(raw, dict) else raw
            raise ValueError(f"Join config must have exactly one key (the input prefix), got: {got}")

        input_prefix, inner = next(iter(raw.items()))
        if isinstance(inner, str):
            return cls(input_prefix=input_prefix, left_on=inner, right_on=inner)

        inner = _to_plain(inner)
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
        return cls(
            input_prefix=input_prefix,
            left_on=left_on,
            right_on=right_on,
            cols=tuple(inner.get("cols", ())),
        )

    def apply(
        self,
        left: pl.LazyFrame,
        input_dir: Path | UPath,
        glob: str = "**/*.parquet",
    ) -> pl.LazyFrame:
        """Scan join target files under ``input_dir`` and left-join them to ``left``.

        Used by every stage that needs to resolve a join: both ``split_and_shard_subjects``
        (which scans from a subsharded directory with ``**/*.parquet``) and
        ``convert_to_subject_sharded`` (which scans from a flat directory with
        ``*.parquet``).
        """
        join_fps = list((input_dir / self.input_prefix).glob(glob))
        right = pl.concat(
            [pl.scan_parquet(fp, glob=False) for fp in join_fps],
            how="vertical_relaxed",
        )
        return left.join(right, left_on=self.left_on, right_on=self.right_on, how="left")


# ── EventConfig ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class EventConfig:
    """A single MEDS event extraction config.

    The ``columns`` dict holds every output column for this event. ``"code"`` is
    mandatory; ``"time"`` is optional and may be ``None`` (static event). All
    other keys are additional output columns (e.g. ``"numeric_value"``,
    ``"text_value"``). Every value is a dftly expression string, which gets
    turned into a polars expression once and cached.

    ``metadata`` holds the raw ``_metadata`` block, or ``{}`` if absent. It's
    consumed separately by ``extract_code_metadata``.

    Examples:
        >>> ev = EventConfig(
        ...     name="lab",
        ...     columns={"code": 'f"{$test}//{$units}"', "time": '$ts::"%Y-%m-%d"',
        ...              "numeric_value": "$result"},
        ... )
        >>> ev.is_static
        False
        >>> sorted(ev.referenced_columns)
        ['result', 'test', 'ts', 'units']
        >>> static = EventConfig(name="eye_color", columns={"code": "EYE_COLOR", "time": None})
        >>> static.is_static
        True
    """

    name: str
    columns: dict[str, str | None]
    metadata: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, name: str, raw: Mapping[str, Any]) -> EventConfig:
        """Validate a raw event block and build an EventConfig.

        Examples:
            >>> EventConfig.parse("lab", {"code": "X", "time": None, "numeric_value": "$v"})
            EventConfig(name='lab', columns={'code': 'X', 'time': None, 'numeric_value': '$v'}, metadata={})

            Missing ``code`` is an error (``time`` is optional):

            >>> EventConfig.parse("lab", {"time": None})
            Traceback (most recent call last):
                ...
            KeyError: "Event 'lab' must contain a 'code' key. Got: [time]."

            Per-event ``subject_id`` is rejected because subject_id is a table-level
            concept (multiple subject IDs within one table would silently break
            subject sharding):

            >>> EventConfig.parse("bad", {"code": "X", "subject_id": "$sid"})
            Traceback (most recent call last):
                ...
            ValueError: Event 'bad' contains a 'subject_id' key. subject_id is a table-level concept ...

            Non-string output values are rejected (every value must be a dftly
            expression string, except ``time`` which may also be ``None``):

            >>> EventConfig.parse("lab", {"code": "X", "numeric_value": 42})
            Traceback (most recent call last):
                ...
            ValueError: Event 'lab' field 'numeric_value' must be a dftly expression string, got int: 42
        """
        raw = dict(_to_plain(raw))
        if "code" not in raw:
            raise KeyError(f"Event '{name}' must contain a 'code' key. Got: [{', '.join(raw.keys())}].")
        if "subject_id" in raw:
            raise ValueError(
                f"Event '{name}' contains a 'subject_id' key. subject_id is a table-level "
                f"concept and must be set in '_defaults', not per-event. See MEDS_extract #73."
            )

        metadata = dict(raw.pop("_metadata", {}))
        columns: dict[str, str | None] = {}
        for k, v in raw.items():
            if k == "time" and v is None:
                columns[k] = None
                continue
            if not isinstance(v, str):
                raise ValueError(
                    f"Event '{name}' field '{k}' must be a dftly expression string, "
                    f"got {type(v).__name__}: {v!r}"
                )
            columns[k] = v

        return cls(name=name, columns=columns, metadata=metadata)

    @property
    def is_static(self) -> bool:
        """True if this event has no time column (time absent or ``None``)."""
        return self.columns.get("time") is None

    @cached_property
    def polars_exprs(self) -> dict[str, pl.Expr]:
        """Polars expression for each output column, built once and reused.

        ``"time"`` resolves to a typed null literal for static events.
        """
        out: dict[str, pl.Expr] = {}
        for k, v in self.columns.items():
            if k == "time" and v is None:
                out[k] = pl.lit(None, dtype=pl.Datetime)
            else:
                out[k] = Parser.expr_to_polars(str(v))
        return out

    @cached_property
    def code_source_columns(self) -> frozenset[str]:
        """Source columns referenced by the ``code`` expression."""
        return frozenset(extract_columns(self.columns["code"]))

    @cached_property
    def time_source_columns(self) -> frozenset[str]:
        """Source columns referenced by the ``time`` expression (empty if static)."""
        if self.is_static:
            return frozenset()
        return frozenset(extract_columns(self.columns["time"]))

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
            cols.update(extract_columns(v))
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
        """
        exprs: dict[str, pl.Expr] = {"subject_id": pl.col("subject_id")}

        exprs["code"] = self.polars_exprs["code"]
        if self.code_source_columns:
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

        if self.code_source_columns:
            first_col = sorted(self.code_source_columns)[0]
            df = df.filter(pl.col(first_col).is_not_null())

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
    so :attr:`subject_id_expr` already reflects the final inherited value.
    ``_table`` sub-keys (``cols``, ``join``) are lifted to top-level fields.

    Examples:
        >>> tc = TableConfig.parse("patients", {
        ...     "_defaults": {"subject_id": "$MRN"},
        ...     "_table": {"join": {"stays": {"key": "stay_id", "cols": ["dischtime"]}}},
        ...     "dob": {"code": "BIRTH", "time": "$dob"},
        ... })
        >>> tc.input_prefix
        'patients'
        >>> tc.subject_id_expr
        '$MRN'
        >>> tc.join.input_prefix
        'stays'
        >>> [e.name for e in tc.events]
        ['dob']
    """

    input_prefix: str
    subject_id_expr: str | None = None
    cols: dict[str, str] = field(default_factory=dict)
    join: JoinConfig | None = None
    events: tuple[EventConfig, ...] = ()

    @classmethod
    def parse(
        cls,
        input_prefix: str,
        raw: Mapping[str, Any],
        global_defaults: Mapping[str, Any] | None = None,
    ) -> TableConfig:
        raw = dict(_to_plain(raw))
        global_defaults = dict(_to_plain(global_defaults or {}))

        file_defaults = dict(_to_plain(raw.pop("_defaults", {})))
        merged_defaults = {**global_defaults, **file_defaults}

        table_cfg = dict(_to_plain(raw.pop("_table", {})))
        cols = dict(_to_plain(table_cfg.get("cols", {})))
        join = JoinConfig.parse(table_cfg["join"]) if "join" in table_cfg else None

        subject_id_expr = merged_defaults.get("subject_id")
        if subject_id_expr is not None:
            subject_id_expr = str(subject_id_expr)

        events = tuple(EventConfig.parse(event_name, event_raw) for event_name, event_raw in raw.items())

        return cls(
            input_prefix=input_prefix,
            subject_id_expr=subject_id_expr,
            cols=cols,
            join=join,
            events=events,
        )

    @cached_property
    def subject_id_polars_expr(self) -> pl.Expr | None:
        """Polars expression producing the MEDS ``subject_id`` column.

        Always produces ``Int64`` output, per the MEDS schema. Most expressions
        are cast via :meth:`polars.Expr.cast`; expressions that return ``UInt64``
        (such as ``hash($col)``, which delegates to polars' hash) are reinterpreted
        instead to preserve the full bit pattern — casting would null-out half the
        hash space.

        Examples:
            >>> import polars as pl
            >>> tc = TableConfig.parse("t", {"_defaults": {"subject_id": "$patient_id"},
            ...                              "e": {"code": "X", "time": None}})
            >>> df = pl.DataFrame({"patient_id": [1, 2]}, schema={"patient_id": pl.Int32})
            >>> df.select(subject_id=tc.subject_id_polars_expr).schema
            Schema({'subject_id': Int64})
            >>> tc = TableConfig.parse("t", {"_defaults": {"subject_id": "hash($mrn)"},
            ...                              "e": {"code": "X", "time": None}})
            >>> df = pl.DataFrame({"mrn": ["ABC", "DEF"]})
            >>> df.select(subject_id=tc.subject_id_polars_expr).schema
            Schema({'subject_id': Int64})
        """
        if self.subject_id_expr is None:
            return None
        node = Parser()(self.subject_id_expr)
        expr = node.polars_expr
        if isinstance(node, Hash):
            return expr.reinterpret(signed=True)
        return expr.cast(pl.Int64, strict=False)

    @cached_property
    def derived_column_exprs(self) -> dict[str, pl.Expr]:
        """Polars expressions for each ``_table.cols`` derived column."""
        return Parser.to_polars(dict(self.cols)) if self.cols else {}

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
        if self.subject_id_expr is not None:
            cols.update(extract_columns(self.subject_id_expr))
        else:
            cols.add(DataSchema.subject_id_name)

        for expr in self.cols.values():
            if isinstance(expr, str):
                cols.update(extract_columns(expr))

        if self.join is not None:
            cols.add(self.join.left_on)

        for event in self.events:
            cols.update(event.referenced_columns)

        return cols - self.col_outputs - self.joined_columns

    def prepare(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Apply subject_id materialization and derived columns to a raw dataframe.

        The result has a ``subject_id`` column (cast to Int64) and any ``_table.cols``
        derived columns. It does **not** apply the join — joins need a stage-specific
        input directory for the join target and are applied by the caller via
        :meth:`JoinConfig.apply`.
        """
        if self.subject_id_polars_expr is not None:
            df = df.with_columns(subject_id=self.subject_id_polars_expr)
        if self.derived_column_exprs:
            df = df.with_columns(**self.derived_column_exprs)
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
            ValueError: if this table has no events, or if extracting any individual
                event fails (the table + event name are included in the error).
        """
        if not self.events:
            raise ValueError(f"Table '{self.input_prefix}' has no events to extract.")

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
        >>> cfg.tables[0].subject_id_expr
        '$MRN'
        >>> cfg.tables[1].subject_id_expr
        '$patient_id'
        >>> [e.name for e in cfg.iter_events()]
        ['dob', 'lab']
    """

    tables: tuple[TableConfig, ...]

    @classmethod
    def parse(cls, raw: Mapping[str, Any] | DictConfig) -> MessyConfig:
        raw_dict = dict(_to_plain(raw))
        global_defaults = dict(_to_plain(raw_dict.pop("_defaults", {})))

        tables = tuple(
            TableConfig.parse(prefix, block, global_defaults) for prefix, block in raw_dict.items()
        )
        return cls(tables=tables)

    @classmethod
    def load(cls, fp: Path | str) -> MessyConfig:
        """Load, validate, and parse a MESSY config file.

        Handles existence check, OmegaConf loading, logging, and parsing in one
        call. All stages should use this rather than calling ``OmegaConf.load``
        directly, so logging stays consistent.
        """
        fp = Path(fp)
        if not fp.exists():
            raise FileNotFoundError(f"Event conversion config file not found: {fp}")
        logger.info(f"Reading event conversion config from {fp}")
        raw = OmegaConf.load(fp)
        logger.info(f"Event conversion config:\n{OmegaConf.to_yaml(raw)}")
        return cls.parse(raw)

    def iter_tables(self) -> Iterator[TableConfig]:
        return iter(self.tables)

    def iter_events(self) -> Iterator[EventConfig]:
        for table in self.tables:
            yield from table.events

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

        Each event's ``_metadata`` block maps metadata-file prefixes to per-prefix
        metadata config dicts. This method returns the reverse: each metadata
        prefix gets the list of ``{code, _metadata}`` entries that reference it.
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
            ...         "end": {
            ...             "code": 'f"PROC//END//{$itemid}"',
            ...             "_metadata": {
            ...                 "proc_datetimeevents": {"desc": ["omop_concept_name", "label"]},
            ...             },
            ...         },
            ...     },
            ... })
            >>> cfg.events_by_metadata_prefix()
            {'proc_datetimeevents': [{'code': 'f"PROC//START//{$itemid}"',
                                      '_metadata': {'desc': ['omop_concept_name', 'label']}},
                                     {'code': 'f"PROC//END//{$itemid}"',
                                      '_metadata': {'desc': ['omop_concept_name', 'label']}}]}
            >>> MessyConfig.parse({"t": {"e": {"code": "X", "time": None}}}).events_by_metadata_prefix()
            {}
        """
        out: dict[str, list[dict]] = {}
        for event in self.iter_events():
            for metadata_prefix, metadata_cfg in event.metadata.items():
                out.setdefault(metadata_prefix, []).append(
                    {"code": event.columns["code"], "_metadata": metadata_cfg}
                )
        return out
