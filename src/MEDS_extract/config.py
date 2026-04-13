"""MESSY event config parsing.

The MESSY config is a DSL for specifying how to extract MEDS events from a set of raw source
tables. It resolves to a list of ``(input_prefix → TableConfig)`` entries, each of which owns
a resolved ``subject_id`` expression, optional derived columns and joins, and a list of
``EventConfig`` entries.

All parsing and validation happens through :class:`MessyConfig`. Stages consume the parsed
class directly — they should not traverse the raw dict themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

from dftly import Parser, extract_columns
from dftly.nodes.arithmetic import Hash
from meds import DataSchema
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    import polars as pl

# Structural keys in the event config that are not event field definitions.
# All use the ``_`` prefix to avoid namespace collisions with event names.
EVENT_META_KEYS = {"_metadata", "_table", "_defaults"}
TABLE_META_KEYS = {"_table", "_defaults"}
EVENT_VALUE_KEYS_RESERVED = {"code", "time", "subject_id"}


def _to_plain_dict(obj: Any) -> Any:
    """Recursively convert OmegaConf containers to plain Python dicts/lists."""
    if isinstance(obj, DictConfig) or OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


@dataclass(frozen=True)
class JoinConfig:
    """Parsed join configuration for a single table join.

    The MESSY syntax supports three equivalent forms, in order of preference:

    - **Minimal** (join key is the same column in both tables, no extra columns needed)::

          join: {stays: stay_id}

    - **Short form** (shared key, extra columns)::

          join: {stays: {key: stay_id, cols: [patient_id, dischtime]}}

    - **Long form** (different key names)::

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
        raw = _to_plain_dict(raw)
        if not isinstance(raw, dict) or len(raw) != 1:
            got = sorted(raw.keys()) if isinstance(raw, dict) else raw
            raise ValueError(f"Join config must have exactly one key (the input prefix), got: {got}")

        input_prefix, cfg = next(iter(raw.items()))

        if isinstance(cfg, str):
            return cls(input_prefix=input_prefix, left_on=cfg, right_on=cfg)

        cfg = _to_plain_dict(cfg)
        if "key" in cfg:
            left_on = right_on = cfg["key"]
        elif "left_on" in cfg and "right_on" in cfg:
            left_on = cfg["left_on"]
            right_on = cfg["right_on"]
        else:
            raise ValueError(
                f"Join config for '{input_prefix}' must specify either 'key' or both "
                f"'left_on' and 'right_on'."
            )

        cols = tuple(cfg.get("cols", ()))
        return cls(input_prefix=input_prefix, left_on=left_on, right_on=right_on, cols=cols)


@dataclass(frozen=True)
class EventConfig:
    """A single resolved MEDS event within a table.

    Event configs are always constructed via :meth:`TableConfig.parse`. The ``name`` and
    ``table_prefix`` fields record the origin of the event (for ``source_block`` tracking),
    while ``code``, ``time``, and ``extras`` hold dftly expressions that will be compiled
    at extraction time. ``metadata`` holds the raw ``_metadata`` block, or ``{}`` if absent.

    Examples:
        >>> ec = EventConfig(
        ...     name="eye_color",
        ...     table_prefix="patients",
        ...     code='f"EYE//{$color}"',
        ...     time=None,
        ...     extras={"eye_color": "$color"},
        ...     metadata={},
        ... )
        >>> ec.source_block
        'patients/eye_color'
        >>> sorted(ec.referenced_columns)
        ['color']
    """

    name: str
    table_prefix: str
    code: str
    time: str | None
    extras: dict[str, str] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def source_block(self) -> str:
        return f"{self.table_prefix}/{self.name}"

    @property
    def referenced_columns(self) -> set[str]:
        """All source columns referenced by this event's code, time, and extras."""
        cols = set(extract_columns(self.code))
        if self.time is not None:
            cols.update(extract_columns(self.time))
        for v in self.extras.values():
            if isinstance(v, str):
                cols.update(extract_columns(v))
        return cols

    def as_legacy_dict(self) -> dict[str, Any]:
        """Render this event config back to the legacy dict form that extract_event consumes.

        Used as a bridge during the stage migration. Will be removed once extract_event
        takes ``EventConfig`` directly.
        """
        out: dict[str, Any] = {"code": self.code, "time": self.time, **self.extras}
        if self.metadata:
            out["_metadata"] = self.metadata
        return out


@dataclass(frozen=True)
class TableConfig:
    """Parsed, fully-resolved config for one file/table block.

    Global ``_defaults`` are merged into file-level ``_defaults`` at parse time, so
    :attr:`subject_id_expr` reflects the final, inherited value. ``_table`` sub-keys
    (``cols``, ``join``) are lifted to top-level fields.

    Examples:
        >>> tc = TableConfig.parse(
        ...     "patients",
        ...     {
        ...         "_defaults": {"subject_id": "$MRN"},
        ...         "_table": {"join": {"stays": {"key": "stay_id", "cols": ["dischtime"]}}},
        ...         "dob": {"code": "BIRTH", "time": None},
        ...     },
        ... )
        >>> tc.input_prefix
        'patients'
        >>> tc.subject_id_expr
        '$MRN'
        >>> tc.subject_id_column
        'MRN'
        >>> tc.join.input_prefix
        'stays'
        >>> [e.name for e in tc.events]
        ['dob']
    """

    input_prefix: str
    subject_id_expr: str | None
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
        raw = dict(_to_plain_dict(raw))
        global_defaults = dict(_to_plain_dict(global_defaults or {}))

        file_defaults = dict(_to_plain_dict(raw.pop("_defaults", {})))
        merged_defaults = {**global_defaults, **file_defaults}

        table_cfg = dict(_to_plain_dict(raw.pop("_table", {})))
        cols = dict(_to_plain_dict(table_cfg.get("cols", {})))
        join = JoinConfig.parse(table_cfg["join"]) if "join" in table_cfg else None

        subject_id_expr = merged_defaults.get("subject_id")
        if subject_id_expr is not None:
            subject_id_expr = str(subject_id_expr)
            cls._validate_subject_id_expr(input_prefix, subject_id_expr)

        events: list[EventConfig] = []
        for event_name, event_raw in raw.items():
            if event_name in TABLE_META_KEYS:
                # Already popped above, but guard for unexpected meta keys.
                continue
            event_raw = _to_plain_dict(event_raw)
            if not isinstance(event_raw, dict):
                raise TypeError(
                    f"Event '{input_prefix}.{event_name}' must be a mapping, got {type(event_raw).__name__}."
                )
            events.append(cls._parse_event(input_prefix, event_name, event_raw))

        return cls(
            input_prefix=input_prefix,
            subject_id_expr=subject_id_expr,
            cols=cols,
            join=join,
            events=tuple(events),
        )

    @staticmethod
    def _parse_event(input_prefix: str, name: str, raw: dict[str, Any]) -> EventConfig:
        loc = f"{input_prefix}.{name}"
        if "code" not in raw:
            raise KeyError(f"Event '{loc}' must contain a 'code' key. Got: [{', '.join(raw.keys())}].")
        if "subject_id" in raw:
            raise ValueError(
                f"Event '{loc}' contains a 'subject_id' key. subject_id is a table-level "
                f"concept and must be set in '_defaults', not per-event. See MEDS_extract #73."
            )

        code = str(raw["code"])
        # ``time`` is optional; missing is equivalent to ``time: null`` (static event). Some
        # metadata-only event blocks in extract_code_metadata legitimately omit time.
        time = raw.get("time")
        if time is not None:
            time = str(time)

        metadata = dict(raw.get("_metadata", {}))
        extras: dict[str, str] = {}
        for k, v in raw.items():
            if k in {"code", "time", "_metadata", "subject_id"}:
                continue
            if not isinstance(v, str):
                raise ValueError(
                    f"Event '{loc}' field '{k}' must be a string (dftly expression). "
                    f"Got {type(v).__name__}: {v!r}"
                )
            extras[k] = v

        return EventConfig(
            name=name, table_prefix=input_prefix, code=code, time=time, extras=extras, metadata=metadata
        )

    @staticmethod
    def _validate_subject_id_expr(input_prefix: str, expr: str) -> None:
        """Enforce the single-source-column invariant for subject_id expressions.

        Subject sharding filters on the source column (or joined-in column) directly, not
        on the compiled expression. For that to produce correct output, ``subject_id_expr``
        must reference exactly one column. ``hash($col)`` is the one allowed wrapper; other
        multi-column expressions are rejected.
        """
        cols = extract_columns(expr)
        if len(cols) != 1:
            raise ValueError(
                f"Table '{input_prefix}' subject_id expression {expr!r} must reference "
                f"exactly one source column, got {sorted(cols)}. Subject sharding requires "
                f"a single-column subject_id (optionally wrapped in hash())."
            )

    @cached_property
    def subject_id_polars_expr(self) -> pl.Expr | None:
        """Compiled Polars expression for subject_id.

        Wraps ``hash()`` outputs with ``reinterpret(signed=True)`` for MEDS compliance
        (Int64 subject IDs).

        Examples:
            >>> import polars as pl
            >>> tc = TableConfig.parse("t", {"_defaults": {"subject_id": "$MRN"},
            ...                              "e": {"code": "X", "time": None}})
            >>> df = pl.DataFrame({"MRN": [1, 2]})
            >>> df.select(subject_id=tc.subject_id_polars_expr)["subject_id"].to_list()
            [1, 2]
        """
        if self.subject_id_expr is None:
            return None
        node = Parser()(self.subject_id_expr)
        expr = node.polars_expr
        if isinstance(node, Hash):
            expr = expr.reinterpret(signed=True)
        return expr

    @property
    def subject_id_column(self) -> str:
        """Source column that ``subject_id_expr`` reads from.

        For simple refs like ``$MRN``, returns ``"MRN"``. For ``hash($mrn)``, returns
        ``"mrn"``. When ``subject_id_expr`` is None, falls back to the literal MEDS schema
        name ``"subject_id"`` (the source file is expected to already have that column).

        Examples:
            >>> TableConfig("t", "$MRN").subject_id_column
            'MRN'
            >>> TableConfig("t", "hash($mrn)").subject_id_column
            'mrn'
            >>> TableConfig("t", None).subject_id_column
            'subject_id'
        """
        if self.subject_id_expr is None:
            return DataSchema.subject_id_name
        cols = extract_columns(self.subject_id_expr)
        # Invariant checked at parse time; safe to pop.
        return next(iter(cols))

    @property
    def col_outputs(self) -> set[str]:
        """Column names produced by ``_table.cols`` — derived, not read from source."""
        return {k for k, v in self.cols.items() if isinstance(v, str)}

    @property
    def joined_columns(self) -> set[str]:
        """Column names that come from the joined table, not the source file."""
        return set(self.join.cols) if self.join is not None else set()

    def source_columns(self) -> set[str]:
        """All columns that must be read from this table's **source** parquet file.

        Includes the subject_id source column, join left-key, derived-column inputs, and
        event code/time/extras references — minus columns that are produced by the
        ``_table.cols`` block or joined in from another table.

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


@dataclass(frozen=True)
class MessyConfig:
    """Fully-parsed MESSY event-conversion config.

    Single entry point for all stages. Call :meth:`parse` once at the start of a stage,
    then query the class methods instead of re-traversing the raw dict.

    Examples:
        >>> cfg = MessyConfig.parse({
        ...     "_defaults": {"subject_id": "$MRN"},
        ...     "patients": {
        ...         "dob": {"code": "BIRTH", "time": "$dob::\\"%Y-%m-%d\\""},
        ...     },
        ...     "labs": {
        ...         "_defaults": {"subject_id": "$patient_id"},
        ...         "_table": {"join": {"stays": {"key": "stay_id", "cols": ["patient_id"]}}},
        ...         "lab": {"code": "$test", "time": "$ts"},
        ...     },
        ... })
        >>> [t.input_prefix for t in cfg.tables]
        ['patients', 'labs']
        >>> cfg.tables[0].subject_id_expr
        '$MRN'
        >>> cfg.tables[1].subject_id_expr  # file-level overrides global
        '$patient_id'
        >>> [(e.table_prefix, e.name) for e in cfg.iter_events()]
        [('patients', 'dob'), ('labs', 'lab')]
        >>> MessyConfig.parse({
        ...     "bad": {"e": {"code": "X", "time": None, "subject_id": "$sid"}},
        ... })
        Traceback (most recent call last):
            ...
        ValueError: Event 'bad.e' contains a 'subject_id' key. ...
    """

    tables: tuple[TableConfig, ...]

    @classmethod
    def parse(cls, raw: Mapping[str, Any] | DictConfig) -> MessyConfig:
        raw_dict = dict(_to_plain_dict(raw))
        global_defaults = dict(_to_plain_dict(raw_dict.pop("_defaults", {})))

        tables: list[TableConfig] = []
        for prefix, block in raw_dict.items():
            block = _to_plain_dict(block)
            if not isinstance(block, dict):
                raise TypeError(f"Table block '{prefix}' must be a mapping, got {type(block).__name__}.")
            tables.append(TableConfig.parse(prefix, block, global_defaults))

        return cls(tables=tuple(tables))

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

        Aggregates both the tables' own :meth:`TableConfig.source_columns` and the columns
        that any table pulls in from a join target. The returned dict is the input to
        :mod:`shard_events` — it tells that stage which columns to project when subsharding
        raw inputs.

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

            Derived columns and joined columns are excluded from the source file's needed
            column list (they come from elsewhere); the join target gets its own entry:

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
        """Group events by the metadata file prefixes their ``_metadata`` blocks reference.

        Each event's ``_metadata`` block maps metadata-file prefixes to per-prefix metadata
        config dicts. This method inverts that mapping: for each metadata prefix, return
        the list of ``{code, _metadata}`` entries that reference it.

        Examples:
            >>> cfg = MessyConfig.parse({
            ...     "_defaults": {"subject_id": "$MRN"},
            ...     "icu/procedureevents": {
            ...         "_defaults": {"subject_id": "$subject_id"},
            ...         "start": {
            ...             "code": 'f"PROCEDURE//START//{$itemid}"',
            ...             "time": None,
            ...             "_metadata": {
            ...                 "proc_datetimeevents": {"desc": ["omop_concept_name", "label"]},
            ...                 "proc_itemid": {"desc": ["omop_concept_name", "label"]},
            ...             },
            ...         },
            ...         "end": {
            ...             "code": 'f"PROCEDURE//END//{$itemid}"',
            ...             "time": None,
            ...             "_metadata": {
            ...                 "proc_datetimeevents": {"desc": ["omop_concept_name", "label"]},
            ...                 "proc_itemid": {"desc": ["omop_concept_name", "label"]},
            ...             },
            ...         },
            ...     },
            ...     "icu/inputevents": {
            ...         "event": {
            ...             "code": 'f"INFUSION//{$itemid}"',
            ...             "time": None,
            ...             "_metadata": {
            ...                 "inputevents_to_rxnorm": {"desc": 'f"{$label}"', "itemid": 'f"{$foo}"'}
            ...             },
            ...         },
            ...     },
            ... })
            >>> cfg.events_by_metadata_prefix()
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
            >>> MessyConfig.parse({
            ...     "icu/procedureevents": {
            ...         "start": {"code": 'f"PROCEDURE//START//{$itemid}"', "time": None},
            ...     },
            ... }).events_by_metadata_prefix()
            {}
        """
        out: dict[str, list[dict]] = {}
        for event in self.iter_events():
            for metadata_prefix, metadata_cfg in event.metadata.items():
                out.setdefault(metadata_prefix, []).append({"code": event.code, "_metadata": metadata_cfg})
        return out
