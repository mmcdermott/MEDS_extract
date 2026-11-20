"""End-to-end tests for the ``mixed_unit_keys`` pre-stage.

The stage's parsing and its grouping logic are doctested in its module. These tests run it the way a
dataset spec does, from ``etl.pre_stages`` in one ``meds-extract-run``, over a tiny raw dataset with:

- ``infusions.csv``: integer item ids (so the side table's key must come out ``Int64``, as
  ``convert_to_parquet`` types the csv, or the join fails). Item 1 is recorded in two rate units and
  gets the unit in its code; item 2's units differ only by whitespace, and item 3's second unit is on
  a row without a start time (no event), so both keep their plain code. Each infusion's end has its
  own unit (``amountuom``), and only item 2 mixes those.
- ``pharma.parquet`` plus ``drug_reference.csv``: drugs whose code is the reference's name, not the id.
  Ids 10 and 11 share the name ``A`` in different dose units, so both get the unit in their code
  (``group_by``); id 12 is alone with its name, and id 13 has no reference entry.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import polars as pl
import pytest

from MEDS_extract.config import EtlConfig, MessyConfig
from MEDS_extract.mixed_unit_keys.mixed_unit_keys import write_side_tables

if TYPE_CHECKING:
    from pathlib import Path

SPEC = """\
etl:
  dataset_name: mixed_units
  raw_dataset_version: "1.0"
  n_subjects_per_shard: 10
  split_fracs: {train: 0.5, tuning: 0.25, held_out: 0.25}
  pre_stages:
    - mixed_unit_keys:
        side_tables:
          infusion_unit_in_code:
            table: infusions
            keys: [itemid]
            flags:
              start_unit_in_code: {unit: $rateuom, time: $starttime}
              end_unit_in_code: {unit: $amountuom, time: $endtime}
          pharma_unit_in_code:
            table: pharma
            keys: {pharmaid: $_ref_id}
            group_by: $name ?? $pharmaid::str
            flags:
              unit_in_code: {unit: $doseunit}

_defaults:
  subject_id: $patient_id

infusions:
  _table:
    joins:
      infusion_unit_in_code:
        key: itemid
        cols: [start_unit_in_code, end_unit_in_code]
    cols:
      _start_unit: '(f"//{$rateuom ?? ''UNK''}" if $start_unit_in_code else "")'
      _end_unit: '(f"//{$amountuom ?? ''UNK''}" if $end_unit_in_code else "")'
  start:
    code: 'f"INFUSION_START//{$itemid}{$_start_unit}"'
    time: '$starttime::"%Y-%m-%d %H:%M:%S"'
    numeric_value: $rate
    _metadata:
      _self:
        unit: $rateuom
  end:
    code: 'f"INFUSION_END//{$itemid}{$_end_unit}"'
    time: '$endtime::"%Y-%m-%d %H:%M:%S"'
    numeric_value: $amount
    _metadata:
      _self:
        unit: $amountuom

pharma:
  _table:
    joins:
      drug_reference:
        left_on: _ref_id
        right_on: ID
        cols: [Name]
      pharma_unit_in_code:
        left_on: _ref_id
        right_on: pharmaid
        cols: [unit_in_code]
    cols:
      _ref_id: $pharmaid::int64
      name: {column: Name}
      _unit: '(f"//{$doseunit ?? ''UNK''}" if $unit_in_code else "")'
  medication:
    code: 'f"MEDICATION//{$name ?? $pharmaid::str}{$_unit}"'
    time: '$givenat::"%Y-%m-%d %H:%M:%S"'
    numeric_value: $dose
    _metadata:
      _self:
        unit: $doseunit
"""

INFUSIONS = """\
patient_id,itemid,starttime,endtime,rate,rateuom,amount,amountuom
1,1,2020-01-01 00:00:00,2020-01-01 01:00:00,1.0,mg/hour,1.0,mg
2,1,2020-01-02 00:00:00,2020-01-02 01:00:00,2.0,mcg/hour,2.0,mg
3,2,2020-01-03 00:00:00,2020-01-03 01:00:00,3.0,mL/hour,3.0,mL
4,2,2020-01-04 00:00:00,2020-01-04 01:00:00,4.0, mL/hour,4.0,L
1,3,2020-01-05 00:00:00,2020-01-05 01:00:00,5.0,mg/hour,5.0,mg
2,3,,2020-01-06 01:00:00,,units/hour,6.0,mg
3,3,2020-01-07 00:00:00,2020-01-07 01:00:00,,,7.0,mg
"""

DRUG_REFERENCE = """\
ID,Name
10,A
11,A
12,B
"""

PHARMA = {
    "patient_id": [1, 2, 3, 4, 1, 2],
    "pharmaid": [10, 11, 11, 12, 13, 13],
    "givenat": [f"2020-02-0{d} 00:00:00" for d in range(1, 7)],
    "dose": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "doseunit": ["mg", "ml", "ml", "mg", "mg", "mg"],
}


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "infusions.csv").write_text(INFUSIONS)
    (raw / "drug_reference.csv").write_text(DRUG_REFERENCE)
    pl.DataFrame(PHARMA, schema_overrides={"pharmaid": pl.Int32}).write_parquet(raw / "pharma.parquet")
    spec_fp = tmp_path / "messy.yaml"
    spec_fp.write_text(SPEC)
    return spec_fp, raw


def _run(spec_fp: Path, raw: Path, out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "meds-extract-run",
            f"spec={spec_fp}",
            f"output_dir={out_dir}",
            "do_download=false",
            f"input_dir={raw}",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )


def test_only_codes_whose_rows_mix_units_end_in_the_unit(dataset, tmp_path):
    spec_fp, raw = dataset
    out_dir = tmp_path / "meds"
    result = _run(spec_fp, raw, out_dir)
    logs = "\n".join(fp.read_text()[-3000:] for fp in sorted(out_dir.glob("**/.logs/**/*.log")))
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}\n{logs}"

    # The side tables, typed as the joins' left sides: the csv's itemid as Int64, the pharma key as the
    # `$pharmaid::int64` it is joined on. The raw input is untouched.
    staged = out_dir / "mixed_unit_keys"
    assert sorted(p.name for p in staged.iterdir() if not p.name.startswith(".")) == [
        "infusion_unit_in_code.parquet",
        "pharma_unit_in_code.parquet",
    ]
    infusion = pl.read_parquet(staged / "infusion_unit_in_code.parquet")
    assert infusion.schema == {
        "itemid": pl.Int64,
        "start_unit_in_code": pl.Boolean,
        "end_unit_in_code": pl.Boolean,
    }
    assert infusion.rows() == [(1, True, None), (2, None, True)]
    pharma = pl.read_parquet(staged / "pharma_unit_in_code.parquet")
    assert pharma.schema == {"pharmaid": pl.Int64, "unit_in_code": pl.Boolean}
    assert pharma.rows() == [(10, True), (11, True)]
    assert sorted(p.name for p in raw.iterdir()) == ["drug_reference.csv", "infusions.csv", "pharma.parquet"]

    codes = pl.read_parquet(out_dir / "metadata" / "codes.parquet")
    units = dict(zip(codes["code"], codes["unit"], strict=True))
    assert units == {
        "INFUSION_START//1//mg/hour": "mg/hour",
        "INFUSION_START//1//mcg/hour": "mcg/hour",
        "INFUSION_START//2": "mL/hour",
        "INFUSION_START//3": "mg/hour",
        "INFUSION_END//1": "mg",
        "INFUSION_END//2//mL": "mL",
        "INFUSION_END//2//L": "L",
        "INFUSION_END//3": "mg",
        "MEDICATION//A//mg": "mg",
        "MEDICATION//A//ml": "ml",
        "MEDICATION//B": "mg",
        "MEDICATION//13": "mg",
    }


def test_configuration_errors_name_the_side_table(dataset, tmp_path):
    spec_fp, raw = dataset
    messy = MessyConfig.load(spec_fp)

    with pytest.raises(ValueError, match="names table 'no_such_table', which is not an event table"):
        write_side_tables(
            messy,
            raw,
            tmp_path / "out",
            {"s": {"table": "no_such_table", "keys": ["k"], "flags": {"f": {"unit": "$u"}}}},
        )
    with pytest.raises(ValueError, match="has the name of an event table"):
        write_side_tables(
            messy,
            raw,
            tmp_path / "out",
            {"pharma": {"table": "infusions", "keys": ["itemid"], "flags": {"f": {"unit": "$rateuom"}}}},
        )
    with pytest.raises(ValueError, match="comes from the side table 'infusion_unit_in_code' this stage"):
        write_side_tables(
            messy,
            raw,
            tmp_path / "out",
            {
                "infusion_unit_in_code": {
                    "table": "infusions",
                    "keys": ["itemid"],
                    "flags": {"start_unit_in_code": {"unit": "$_start_unit"}},
                }
            },
        )


def test_the_spec_may_set_only_side_tables():
    etl = EtlConfig.parse({"pre_stages": [{"mixed_unit_keys": {"side_table": {}}}]})
    with pytest.raises(
        ValueError, match=r"unknown option\(s\) \['side_table'\]; its default config declares"
    ):
        etl.check_extension_stages()
    EtlConfig.parse({"pre_stages": [{"mixed_unit_keys": {"side_tables": {}}}]}).check_extension_stages()
