"""Pre-stage: find the codes whose rows mix units, and stage them as side tables.

``extract_code_metadata`` writes one ``unit`` per code and fails on a code whose rows record
several. Most items of a dataset are recorded in one unit, but a few are recorded in more (an
infusion given in ``mg/hour`` and ``mcg/kg/min``, a drug ordered in ``mg`` and ``g``). The usual fix
is to make the unit part of only those codes (``INFUSION//221906//mg/hour``) and leave every other
code as it is. Which items those are depends on the data, so a spec cannot name them; this stage
finds them on every run and writes them as **side tables** that the table's ``_table.joins`` reads.
A ``_table.cols`` entry then appends the unit to the code only where the joined flag is true.

It is configured entirely from the spec (``etl.pre_stages``), with one entry per side table:

.. code-block:: yaml

    etl:
      pre_stages:
        - mixed_unit_keys:
            side_tables:
              inputevents_unit_in_code:       # the side table's prefix, which a join names
                table: icu/inputevents        # the spec table whose rows are compared
                keys: [itemid]                # the side table's key columns
                flags:                        # one Boolean column per code family
                  start_unit_in_code: {unit: $rateuom, time: $starttime}
                  end_unit_in_code: {unit: $amountuom, time: $endtime}

    icu/inputevents:
      _table:
        joins:
          inputevents_unit_in_code:
            key: itemid
            cols: [start_unit_in_code, end_unit_in_code]
        cols:
          _start_unit: '(f"//{$rateuom ?? ''UNK''}" if $start_unit_in_code else "")'
      input_start:
        code: 'f"INFUSION_START//{$itemid}{$_start_unit}"'
        ...

Each side table entry takes:

- ``table``: an event table of the spec. Its rows are read as the pipeline reads them: typed as
  ``convert_to_parquet`` types them (so a key's dtype matches the join's left side), with the
  table's ``_table.joins`` and ``_table.cols`` available to the expressions below. Joins on the side
  tables this stage writes are left out, since they do not exist yet; an expression may not use
  their columns.
- ``keys``: the side table's key columns, as a list of column names of the table (a raw,
  ``_table.cols`` or joined column), or as a mapping of side-table column name to a dftly
  expression. Rows with a null key are not compared.
- ``group_by`` (optional): one dftly expression, or a list of them, whose rows must share one unit
  because they share one code. Defaults to the keys. Set it when the code is built from something
  other than the join key, e.g. the item's name when several item ids share a name.
- ``flags``: a mapping of flag column name to a code family: ``unit``, a dftly expression giving a
  row's unit, and optionally ``time``, the family's event time; rows where it is null make no event
  and are not compared.

For each flag, units are compared as ``extract_code_metadata`` compares them when it writes a
code's one unit: as Strings, stripped, with nulls and blanks left out. A group with more than one
distinct unit is *mixed*, and every key of its compared rows gets the flag ``true``. The side table
has one row per key with at least one flag; a flag is ``true`` or null, never ``false``, so a joined
row of another key gets null, which an ``if`` treats as false. It is written as
``<output_dir>/<side table>.parquet``, sorted by its keys, and ``convert_to_parquet`` reads it from
there like any table (see the README's "Extension stages").
"""

from __future__ import annotations

import dataclasses
import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from dftly import Parser
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig, OmegaConf
from upath import UPath

from ..config import MessyConfig, TableConfig
from ..encoding import AUTO_ENCODING, source_encoding
from ..io import DEFAULT_CSV_SEPARATOR, resolve_source_files, source_separator

if TYPE_CHECKING:
    from dftly.nodes.base import NodeBase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlagSpec:
    """One code family of a side table: the unit of its rows, and its event time.

    Examples:
        >>> flag = FlagSpec.parse("t", "start_unit_in_code", {"unit": "$rateuom", "time": "$starttime"})
        >>> flag.name, sorted(flag.referenced_columns)
        ('start_unit_in_code', ['rateuom', 'starttime'])
        >>> FlagSpec.parse("t", "f", {"unit": "$u"}).time is None
        True
        >>> FlagSpec.parse("t", "f", {"time": "$t"})
        Traceback (most recent call last):
            ...
        ValueError: mixed_unit_keys side table 't', flag 'f': needs a 'unit' expression.
        >>> FlagSpec.parse("t", "f", {"unit": "$u", "when": "$t"})
        Traceback (most recent call last):
            ...
        ValueError: mixed_unit_keys side table 't', flag 'f': unknown key(s) ['when']; allowed: time,
        unit.
    """

    name: str
    unit: NodeBase
    time: NodeBase | None = None

    @classmethod
    def parse(cls, side_table: str, name: str, raw: Any) -> FlagSpec:
        where = f"mixed_unit_keys side table '{side_table}', flag '{name}'"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{where}: must be a mapping with 'unit' (and optionally 'time'), got {raw!r}.")
        if unknown := sorted(set(raw) - {"unit", "time"}):
            raise ValueError(f"{where}: unknown key(s) {unknown}; allowed: time, unit.")
        if raw.get("unit") is None:
            raise ValueError(f"{where}: needs a 'unit' expression.")
        parser = Parser()
        time = raw.get("time")
        return cls(name=name, unit=parser(raw["unit"]), time=None if time is None else parser(time))

    @property
    def referenced_columns(self) -> set[str]:
        cols = set(self.unit.referenced_columns)
        if self.time is not None:
            cols |= self.time.referenced_columns
        return cols


@dataclass(frozen=True)
class SideTableSpec:
    """One side table of the ``mixed_unit_keys`` pre-stage, parsed from its ``side_tables`` entry.

    Examples:
        Key columns by name, the group defaulting to the keys:

        >>> spec = SideTableSpec.parse("inputevents_unit_in_code", {
        ...     "table": "icu/inputevents",
        ...     "keys": ["itemid"],
        ...     "flags": {
        ...         "start_unit_in_code": {"unit": "$rateuom", "time": "$starttime"},
        ...         "end_unit_in_code": {"unit": "$amountuom", "time": "$endtime"},
        ...     },
        ... })
        >>> spec.table, list(spec.keys), spec.group_by, list(spec.flags)
        ('icu/inputevents', ['itemid'], None, ['start_unit_in_code', 'end_unit_in_code'])
        >>> sorted(spec.referenced_columns)
        ['amountuom', 'endtime', 'itemid', 'rateuom', 'starttime']

        Keys as expressions, and a group other than the keys:

        >>> spec = SideTableSpec.parse("pharma_unit_in_code", {
        ...     "table": "pharma",
        ...     "keys": {"pharmaid": "$pharmaid::int64"},
        ...     "group_by": "$name ?? $pharmaid::str",
        ...     "flags": {"unit_in_code": {"unit": "$doseunit"}},
        ... })
        >>> len(spec.group_by), sorted(spec.referenced_columns)
        (1, ['doseunit', 'name', 'pharmaid'])

        Malformed entries are rejected, naming the side table:

        >>> SideTableSpec.parse("s", {"table": "t", "keys": [], "flags": {"f": {"unit": "$u"}}})
        Traceback (most recent call last):
            ...
        ValueError: mixed_unit_keys side table 's': 'keys' must name at least one key column (a list of
        column names, or a mapping of column name to expression), got [].
        >>> SideTableSpec.parse("s", {"table": "t", "keys": ["k"], "flags": {}})
        Traceback (most recent call last):
            ...
        ValueError: mixed_unit_keys side table 's': 'flags' must map at least one flag column name to
        {unit: ..., time: ...}, got {}.
        >>> SideTableSpec.parse("s", {"table": "t", "keys": ["f"], "flags": {"f": {"unit": "$u"}}})
        Traceback (most recent call last):
            ...
        ValueError: mixed_unit_keys side table 's': flag column(s) ['f'] are also key columns.
        >>> SideTableSpec.parse("s", {"table": "t", "key": ["k"], "flags": {"f": {"unit": "$u"}}})
        Traceback (most recent call last):
            ...
        ValueError: mixed_unit_keys side table 's': unknown key(s) ['key']; allowed: flags, group_by,
        keys, table.
    """

    name: str
    table: str
    keys: dict[str, NodeBase]
    flags: dict[str, FlagSpec]
    group_by: tuple[NodeBase, ...] | None = None

    ALLOWED: tuple[str, ...] = ("flags", "group_by", "keys", "table")

    @classmethod
    def parse(cls, name: str, raw: Any) -> SideTableSpec:
        where = f"mixed_unit_keys side table '{name}'"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{where}: must be a mapping with 'table', 'keys' and 'flags', got {raw!r}.")
        if unknown := sorted(set(raw) - set(cls.ALLOWED)):
            raise ValueError(f"{where}: unknown key(s) {unknown}; allowed: {', '.join(cls.ALLOWED)}.")
        table = raw.get("table")
        if not isinstance(table, str) or not table:
            raise ValueError(f"{where}: 'table' must name an event table of the spec, got {table!r}.")

        parser = Parser()
        raw_keys = raw.get("keys")
        if isinstance(raw_keys, str):
            raw_keys = [raw_keys]
        if isinstance(raw_keys, Sequence) and raw_keys and all(isinstance(k, str) for k in raw_keys):
            keys = {k: parser({"column": k}) for k in raw_keys}
        elif isinstance(raw_keys, Mapping) and raw_keys:
            keys = {str(k): parser(v) for k, v in raw_keys.items()}
        else:
            raise ValueError(
                f"{where}: 'keys' must name at least one key column (a list of column names, or a "
                f"mapping of column name to expression), got {raw_keys!r}."
            )

        raw_flags = raw.get("flags")
        if not isinstance(raw_flags, Mapping) or not raw_flags:
            raise ValueError(
                f"{where}: 'flags' must map at least one flag column name to {{unit: ..., time: ...}}, "
                f"got {raw_flags!r}."
            )
        flags = {str(f): FlagSpec.parse(name, str(f), v) for f, v in raw_flags.items()}
        if clash := sorted(set(flags) & set(keys)):
            raise ValueError(f"{where}: flag column(s) {clash} are also key columns.")

        raw_group = raw.get("group_by")
        if raw_group is None:
            group_by = None
        else:
            items = (
                raw_group
                if isinstance(raw_group, Sequence) and not isinstance(raw_group, str)
                else [raw_group]
            )
            if not items:
                raise ValueError(f"{where}: 'group_by' must be an expression or a non-empty list of them.")
            group_by = tuple(parser(g) for g in items)
        return cls(name=name, table=table, keys=keys, flags=flags, group_by=group_by)

    @property
    def referenced_columns(self) -> set[str]:
        """Every column the keys, groups, units and times read."""
        cols: set[str] = set()
        for node in (*self.keys.values(), *(self.group_by or ())):
            cols |= node.referenced_columns
        for flag in self.flags.values():
            cols |= flag.referenced_columns
        return cols


def parse_side_tables(raw: Any) -> list[SideTableSpec]:
    """Parse the stage's ``side_tables`` option.

    Examples:
        >>> [s.name for s in parse_side_tables({
        ...     "a": {"table": "t", "keys": ["k"], "flags": {"f": {"unit": "$u"}}},
        ...     "b": {"table": "t", "keys": ["k"], "flags": {"g": {"unit": "$v"}}},
        ... })]
        ['a', 'b']
        >>> parse_side_tables({})
        Traceback (most recent call last):
            ...
        ValueError: mixed_unit_keys needs at least one entry under 'side_tables' (side table name ->
        {table, keys, flags}); got {}.
    """
    if isinstance(raw, DictConfig):
        raw = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(
            f"mixed_unit_keys needs at least one entry under 'side_tables' (side table name -> "
            f"{{table, keys, flags}}); got {raw!r}."
        )
    return [SideTableSpec.parse(str(name), spec) for name, spec in raw.items()]


def mixed_unit_keys(rows: pl.LazyFrame, spec: SideTableSpec) -> pl.DataFrame:
    """The keys of ``rows`` whose groups mix units, one row per key with its flags.

    Args:
        rows: The table's rows, with every column the spec's expressions read.
        spec: The side table.

    Returns:
        The side table: the key columns, then one Boolean column per flag, ``true`` or null.
        Sorted by the keys.

    Examples:
        Units are compared stripped, with nulls and blanks left out, and only on rows with a time:

        >>> rows = pl.LazyFrame({
        ...     "itemid": [1, 1, 1, 2, 2, 3, 3, 4],
        ...     "starttime": ["t"] * 7 + [None],
        ...     "endtime": ["t"] * 8,
        ...     "rateuom": ["mg/hour", "mcg/hour", None, "mL/hour", " mL/hour", None, "mL/hour", "mg/hour"],
        ...     "amountuom": ["mg", "mg", "mg", "mL", "mL", "mL", "L", "mL"],
        ... })
        >>> spec = SideTableSpec.parse("s", {
        ...     "table": "t", "keys": ["itemid"],
        ...     "flags": {
        ...         "start_unit_in_code": {"unit": "$rateuom", "time": "$starttime"},
        ...         "end_unit_in_code": {"unit": "$amountuom", "time": "$endtime"},
        ...     },
        ... })
        >>> mixed_unit_keys(rows, spec)
        shape: (2, 3)
        ┌────────┬────────────────────┬──────────────────┐
        │ itemid ┆ start_unit_in_code ┆ end_unit_in_code │
        │ ---    ┆ ---                ┆ ---              │
        │ i64    ┆ bool               ┆ bool             │
        ╞════════╪════════════════════╪══════════════════╡
        │ 1      ┆ true               ┆ null             │
        │ 3      ┆ null               ┆ true             │
        └────────┴────────────────────┴──────────────────┘

        Key expressions, and a group other than the keys: here ids 1 and 2 share the name ``A`` but not
        its unit, so both get the flag, while id 3 is alone with its name. Id 5 has no name; its group is
        its id as a String:

        >>> rows = pl.LazyFrame({
        ...     "pharmaid": [1, 2, 2, 3, 5, 5],
        ...     "name": ["A", "A", "A", "B", None, None],
        ...     "doseunit": ["mg", "ml", " ml", "mg", "mg", "g"],
        ... })
        >>> spec = SideTableSpec.parse("s", {
        ...     "table": "t", "keys": {"id": "$pharmaid"}, "group_by": "$name ?? $pharmaid::str",
        ...     "flags": {"unit_in_code": {"unit": "$doseunit"}},
        ... })
        >>> mixed_unit_keys(rows, spec).rows()
        [(1, True), (2, True), (5, True)]

        A table without mixed units gives an empty side table of the right schema:

        >>> dict(mixed_unit_keys(rows.filter(pl.col("pharmaid") == 3), spec).schema)
        {'id': Int64, 'unit_in_code': Boolean}
    """
    key_names = list(spec.keys)
    if spec.group_by is None:
        group_names, group_exprs = key_names, []
    else:
        group_names = [f"_group_{i}" for i in range(len(spec.group_by))]
        group_exprs = [g.polars_expr.alias(n) for n, g in zip(group_names, spec.group_by, strict=True)]
    flag_exprs = []
    for f in spec.flags.values():
        flag_exprs.append(f.unit.polars_expr.cast(pl.String).str.strip_chars().alias(f"_unit_{f.name}"))
        if f.time is not None:
            flag_exprs.append(f.time.polars_expr.is_not_null().alias(f"_time_{f.name}"))
    keyed = rows.select(
        *(node.polars_expr.alias(name) for name, node in spec.keys.items()), *group_exprs, *flag_exprs
    ).filter(pl.all_horizontal(pl.col(c).is_not_null() for c in dict.fromkeys([*key_names, *group_names])))

    per_flag = []
    for flag in spec.flags.values():
        unit = pl.col(f"_unit_{flag.name}")
        compared = keyed.filter(unit.is_not_null() & (unit != ""))
        if flag.time is not None:
            compared = compared.filter(pl.col(f"_time_{flag.name}"))
        mixed = (
            compared.group_by(group_names)
            .agg(unit.n_unique().alias("_n_units"))
            .filter(pl.col("_n_units") > 1)
            .select(group_names)
        )
        if spec.group_by is not None:
            mixed = compared.join(mixed, on=group_names, how="semi", nulls_equal=False).select(key_names)
        per_flag.append(mixed.unique().with_columns(pl.lit(True).alias(flag.name)))

    table = per_flag[0]
    for other in per_flag[1:]:
        table = table.join(other, on=key_names, how="full", coalesce=True, nulls_equal=False)
    return table.collect(engine="streaming").sort(key_names)


def _needed(table: TableConfig, spec: SideTableSpec, side_tables: set[str]) -> tuple[TableConfig, set[str]]:
    """The part of ``table``'s ``_table`` block that ``spec``'s expressions need, and its raw columns.

    Follows each referenced column to the ``_table.cols`` entry or join that produces it, transitively,
    and keeps only those (in their original order), so that a ``_table.cols`` entry reading a side
    table's flag is never evaluated.

    Examples:
        >>> table = TableConfig.parse("pharma", {
        ...     "_table": {
        ...         "joins": {
        ...             "ref": {"left_on": ["_id", "_kind"], "right_on": ["ID", "Kind"], "cols": ["Name"]},
        ...             "pharma_unit_in_code": {"left_on": "_id", "right_on": "pharmaid", "cols": ["flag"]},
        ...         },
        ...         "cols": {
        ...             "_kind": "'Pharma'",
        ...             "_id": "$pharmaid::int64",
        ...             "name": {"column": "Name"},
        ...             "_suffix": '("//x" if $flag else "")',
        ...         },
        ...     },
        ...     "med": {"code": 'f"MED//{$name}{$_suffix}"', "time": "$givenat"},
        ... })
        >>> spec = SideTableSpec.parse("pharma_unit_in_code", {
        ...     "table": "pharma", "keys": {"pharmaid": "$_id"}, "group_by": "$name ?? $pharmaid::str",
        ...     "flags": {"flag": {"unit": "$doseunit"}},
        ... })
        >>> reduced, raw = _needed(table, spec, {"pharma_unit_in_code"})
        >>> list(reduced.cols), [j.input_prefix for j in reduced.joins], sorted(raw)
        (['_kind', '_id', 'name'], ['ref'], ['doseunit', 'pharmaid'])

        An expression may not read a side table this stage writes:

        >>> bad = dataclasses.replace(spec, group_by=(Parser()("$_suffix"),))
        >>> _needed(table, bad, {"pharma_unit_in_code"})
        Traceback (most recent call last):
            ...
        ValueError: mixed_unit_keys side table 'pharma_unit_in_code' reads column 'flag' of table 'pharma',
        which comes from the side table 'pharma_unit_in_code' this stage writes.
    """
    delivered_by = {c: j for j in table.joins for c in j.delivered_columns}
    cols: set[str] = set()
    joins: list = []
    raw: set[str] = set()
    frontier = list(spec.referenced_columns)
    seen: set[str] = set()
    while frontier:
        c = frontier.pop()
        if c in seen:
            continue
        seen.add(c)
        if c in table.cols:
            cols.add(c)
            frontier.extend(table.cols[c].referenced_columns)
        elif c in delivered_by:
            join = delivered_by[c]
            if join.input_prefix in side_tables:
                raise ValueError(
                    f"mixed_unit_keys side table '{spec.name}' reads column '{c}' of table "
                    f"'{table.input_prefix}', which comes from the side table '{join.input_prefix}' "
                    f"this stage writes."
                )
            if join not in joins:
                joins.append(join)
                frontier.extend(join.left_on)
        else:
            raw.add(c)
    reduced = dataclasses.replace(
        table,
        cols={k: v for k, v in table.cols.items() if k in cols},
        joins=tuple(j for j in table.joins if j in joins),
    )
    return reduced, raw


def _stage_source(
    input_dir: Path | UPath,
    prefix: str,
    columns: list[str],
    staging_dir: Path,
    *,
    encoding: str,
    separator: str,
) -> None:
    """Write ``prefix``'s raw files under ``staging_dir`` as ``convert_to_parquet`` writes them.

    The same conversion (csv typing, encoding, separator, projection) as the pipeline's first stage, so every
    column has the dtype it will have when the side table is joined.
    """
    from ..convert_to_parquet.convert_to_parquet import _convert_one, _dest_for

    for fp in resolve_source_files(input_dir, prefix):
        _convert_one(
            fp,
            _dest_for(fp, input_dir, staging_dir),
            prefix=prefix,
            columns=columns,
            encoding=encoding,
            separator=separator,
        )


def write_side_tables(
    messy_cfg: MessyConfig, input_dir: Path | UPath, output_dir: Path, raw_side_tables: Any
) -> dict[str, Path]:
    """Compute every configured side table from the raw data in ``input_dir``; write them to ``output_dir``.

    Returns:
        The written files, by side table name, each replaced atomically.
    """
    specs = parse_side_tables(raw_side_tables)
    tables = {t.input_prefix: t for t in messy_cfg.event_tables}
    names = {s.name for s in specs}
    for spec in specs:
        if spec.table not in tables:
            raise ValueError(
                f"mixed_unit_keys side table '{spec.name}' names table '{spec.table}', which is not an "
                f"event table of the spec (tables: {sorted(tables)})."
            )
        if spec.name in tables:
            raise ValueError(
                f"mixed_unit_keys side table '{spec.name}' has the name of an event table of the spec; "
                f"a staged table replaces the raw table of the same prefix."
            )

    etl_options = messy_cfg.etl.stage_options
    encoding = etl_options.get("encoding", AUTO_ENCODING)
    separator = etl_options.get("csv_separator", DEFAULT_CSV_SEPARATOR)
    needed_columns = messy_cfg.needed_source_columns()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(dir=output_dir, prefix=".staging-") as tmp:
        staging_dir = Path(tmp)
        staged: dict[str, set[str]] = {}
        for spec in specs:
            table, raw_columns = _needed(tables[spec.table], spec, names)
            left_keys = {k for j in table.joins for k in j.left_on} - set(table.cols)
            columns = {spec.table: raw_columns | left_keys}
            for join in table.joins:
                columns[join.input_prefix] = set(needed_columns[join.input_prefix])
            for prefix, cols in columns.items():
                if not cols <= staged.get(prefix, set()):
                    cols = cols | staged.get(prefix, set())
                    _stage_source(
                        input_dir,
                        prefix,
                        sorted(cols),
                        staging_dir,
                        encoding=source_encoding(encoding, prefix),
                        separator=source_separator(separator, prefix),
                    )
                    staged[prefix] = cols
            rows = table.scan(staging_dir)
            for name, node in table.cols.items():
                rows = rows.with_columns(node.polars_expr.alias(name))
            side = mixed_unit_keys(rows, spec)

            out = output_dir / f"{spec.name}.parquet"
            tmp_out = out.with_name(f".{out.name}.tmp")
            side.write_parquet(tmp_out)
            os.replace(tmp_out, out)
            written[spec.name] = out
            counts = ", ".join(f"{side[f].sum()} {f}" for f in spec.flags)
            logger.info(f"{spec.name}: {side.height} keys of {spec.table} whose rows mix units ({counts}).")
    return written


@Stage.register(is_metadata=False)
def main(cfg: DictConfig):
    """Write the configured side tables from the raw data in ``input_dir`` (worker 0 only)."""
    if cfg.get("worker", 0) != 0:
        return
    messy_cfg = MessyConfig.load(cfg.MESSY_config_fp)
    write_side_tables(
        messy_cfg, UPath(cfg.input_dir), Path(cfg.stage_cfg.output_dir), cfg.stage_cfg.side_tables
    )
