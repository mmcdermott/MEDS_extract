"""MESSY event config parsing.

Provides structured parsing of the MESSY event conversion config format. Each file block
can contain ``_defaults`` (inherited event field defaults), ``_table`` (table modifications
like joins and derived columns), and event definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dftly import extract_columns

# Structural keys in the event config that are not event field definitions.
# All use _ prefix to avoid namespace collisions with event names.
EVENT_META_KEYS = {"_metadata", "_table", "_defaults"}


@dataclass
class JoinConfig:
    """Parsed join configuration for a single table join.

    Examples:
        >>> JoinConfig.parse({"stays": {"key": "stay_id", "cols": ["subject_id"]}})
        JoinConfig(input_prefix='stays', left_on='stay_id', right_on='stay_id', cols=['subject_id'])
        >>> JoinConfig.parse({"admissions": {"left_on": "hadm_id", "right_on": "admission_id"}})
        JoinConfig(input_prefix='admissions', left_on='hadm_id', right_on='admission_id', cols=[])
        >>> JoinConfig.parse({"stays": "stay_id"})
        JoinConfig(input_prefix='stays', left_on='stay_id', right_on='stay_id', cols=[])
    """

    input_prefix: str
    left_on: str
    right_on: str
    cols: list[str] = field(default_factory=list)

    @classmethod
    def parse(cls, join_cfg: dict) -> JoinConfig:
        """Parse a join config dict into a JoinConfig.

        Supports three forms:

        Minimal (just the join key):
            ``{prefix: key_col}``

        Short form (matching join keys):
            ``{prefix: {key: key_col, cols: [col1, col2]}}``

        Long form (different join keys):
            ``{prefix: {left_on: l, right_on: r, cols: [col1]}}``

        Raises:
            ValueError: If the join config has multiple prefixes or missing keys.

        Examples:
            >>> JoinConfig.parse({"stays": {"key": "stay_id"}})
            JoinConfig(input_prefix='stays', left_on='stay_id', right_on='stay_id', cols=[])
            >>> JoinConfig.parse({"stays": {"left_on": "a", "right_on": "b", "cols": ["x"]}})
            JoinConfig(input_prefix='stays', left_on='a', right_on='b', cols=['x'])
            >>> JoinConfig.parse({"a": {}, "b": {}})
            Traceback (most recent call last):
                ...
            ValueError: Join config must have exactly one key (the input prefix), got: ['a', 'b']
        """
        if len(join_cfg) != 1:
            raise ValueError(
                f"Join config must have exactly one key (the input prefix), got: {sorted(join_cfg.keys())}"
            )

        input_prefix, cfg = next(iter(join_cfg.items()))

        if isinstance(cfg, str):
            return cls(input_prefix=input_prefix, left_on=cfg, right_on=cfg)

        if "key" in cfg:
            left_on = right_on = cfg["key"]
        else:
            left_on = cfg["left_on"]
            right_on = cfg["right_on"]

        cols = list(cfg.get("cols", cfg.get("columns", [])))
        return cls(input_prefix=input_prefix, left_on=left_on, right_on=right_on, cols=cols)


@dataclass
class FileConfig:
    """Parsed configuration for a single file block in the event conversion config.

    Examples:
        >>> fc = FileConfig.parse({
        ...     "_defaults": {"subject_id": "$MRN"},
        ...     "_table": {
        ...         "cols": {"year": "$anchor_year - $anchor_age"},
        ...         "join": {"stays": {"key": "stay_id", "cols": ["dischtime"]}},
        ...     },
        ...     "dob": {"code": "BIRTH", "time": "$year::year"},
        ... })
        >>> fc.defaults
        {'subject_id': '$MRN'}
        >>> fc.cols
        {'year': '$anchor_year - $anchor_age'}
        >>> fc.join.input_prefix
        'stays'
        >>> sorted(fc.events.keys())
        ['dob']
    """

    defaults: dict[str, str] = field(default_factory=dict)
    cols: dict[str, str] | None = None
    join: JoinConfig | None = None
    events: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def parse(cls, event_cfgs: dict, global_defaults: dict | None = None) -> FileConfig:
        """Parse a file-level config block into a FileConfig.

        Args:
            event_cfgs: The raw config dict for one file block.
            global_defaults: Global defaults to merge with file-level defaults.

        Returns:
            A FileConfig with parsed defaults, table config, and event definitions.

        Examples:
            >>> fc = FileConfig.parse({"dob": {"code": "DOB", "time": None}})
            >>> fc.events
            {'dob': {'code': 'DOB', 'time': None}}
            >>> fc.defaults
            {}
            >>> fc = FileConfig.parse(
            ...     {"_defaults": {"subject_id": "$patient_id"}},
            ...     global_defaults={"subject_id": "$MRN"},
            ... )
            >>> fc.defaults
            {'subject_id': '$patient_id'}
        """
        event_cfgs = dict(event_cfgs)
        global_defaults = global_defaults or {}

        file_defaults = {**global_defaults, **dict(event_cfgs.pop("_defaults", {}))}
        table_cfg = dict(event_cfgs.pop("_table", {}))

        cols = dict(table_cfg["cols"]) if "cols" in table_cfg else None
        join = JoinConfig.parse(dict(table_cfg["join"])) if "join" in table_cfg else None

        events = {k: dict(v) for k, v in event_cfgs.items() if k not in EVENT_META_KEYS}

        return cls(defaults=file_defaults, cols=cols, join=join, events=events)

    @property
    def subject_id_expr(self) -> str | None:
        """The subject_id expression from defaults, or None if not set.

        Examples:
            >>> FileConfig(defaults={"subject_id": "hash($mrn)"}).subject_id_expr
            'hash($mrn)'
            >>> FileConfig(defaults={}).subject_id_expr is None
            True
        """
        return self.defaults.get("subject_id")

    @property
    def subject_id_column(self) -> str:
        """The source column name for subject_id, extracted from the defaults expression.

        For simple column references like ``$MRN``, returns ``"MRN"``.
        For complex expressions like ``hash($mrn)``, returns ``"mrn"`` (the single referenced column).
        Falls back to ``"subject_id"`` if no expression is set or multiple columns are referenced.

        Examples:
            >>> FileConfig(defaults={"subject_id": "$MRN"}).subject_id_column
            'MRN'
            >>> FileConfig(defaults={"subject_id": "hash($mrn)"}).subject_id_column
            'mrn'
            >>> FileConfig(defaults={}).subject_id_column
            'subject_id'
        """
        expr = self.subject_id_expr
        if expr is None:
            return "subject_id"
        cols = extract_columns(expr)
        return cols.pop() if len(cols) == 1 else "subject_id"

    @property
    def col_outputs(self) -> set[str]:
        """Column names produced by _table.cols (derived columns, not in source file).

        Examples:
            >>> FileConfig(cols={"year": "$a - $b"}).col_outputs
            {'year'}
            >>> FileConfig(cols=None).col_outputs
            set()
        """
        if self.cols is None:
            return set()
        return {k for k, v in self.cols.items() if isinstance(v, str)}

    @property
    def joined_columns(self) -> set[str]:
        """Column names that come from the joined table (not from the source file).

        Examples:
            >>> from dataclasses import dataclass
            >>> fc = FileConfig(join=JoinConfig("stays", "id", "id", ["subject_id", "dischtime"]))
            >>> sorted(fc.joined_columns)
            ['dischtime', 'subject_id']
            >>> FileConfig(join=None).joined_columns
            set()
        """
        if self.join is None:
            return set()
        return set(self.join.cols)


def parse_global_defaults(event_conversion_cfg: dict) -> dict[str, str]:
    """Extract and remove global _defaults from the top-level event conversion config.

    Args:
        event_conversion_cfg: The full event conversion config. Modified in place — _defaults is removed.

    Returns:
        The global defaults dict.

    Examples:
        >>> cfg = {"_defaults": {"subject_id": "$MRN"}, "patients": {"dob": {"code": "DOB"}}}
        >>> parse_global_defaults(cfg)
        {'subject_id': '$MRN'}
        >>> "_defaults" in cfg
        False
    """
    return dict(event_conversion_cfg.pop("_defaults", {}))
