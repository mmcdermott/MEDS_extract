"""Event outputs must not replace extraction bookkeeping columns."""

import polars as pl
import pytest

from MEDS_extract.config import EventConfig


@pytest.mark.parametrize("name", ["source_block", "code_components", "metadata_components"])
def test_generated_column_names_are_rejected_at_construction(name):
    with pytest.raises(ValueError, match=name) as error:
        EventConfig.parse("measurement", {"code": "SYNTHETIC", "time": None, name: "$provided"})
    assert "measurement" in str(error.value)
    assert "reserved" in str(error.value)


def test_same_named_source_columns_can_be_preserved_under_other_output_names():
    event = EventConfig.parse(
        "measurement",
        {
            "code": 'f"SYNTHETIC//{$itemid}"',
            "time": None,
            "original_source": "$source_block",
            "original_components": "$code_components",
            "original_metadata": "$metadata_components",
            "_metadata": {"_self": {"description": "$description"}},
        },
    )
    raw = pl.LazyFrame(
        {
            "subject_id": [1],
            "itemid": ["example"],
            "source_block": ["original source"],
            "code_components": ["original components"],
            "metadata_components": ["original metadata"],
            "description": ["invented description"],
        }
    )
    actual = event.extract(raw, "synthetic/measurement").collect()
    assert actual["original_source"].to_list() == ["original source"]
    assert actual["original_components"].to_list() == ["original components"]
    assert actual["original_metadata"].to_list() == ["original metadata"]
    assert actual["source_block"].to_list() == ["synthetic/measurement"]
    assert actual["code_components"].to_list() == [{"itemid": "example"}]
    assert actual["metadata_components"].to_list() == [{"description": "invented description"}]
