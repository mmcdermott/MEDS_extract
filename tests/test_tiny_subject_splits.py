"""Small cohorts can populate every requested split without losing subjects."""

import numpy as np
import pytest

from MEDS_extract.split_and_shard_subjects.split_and_shard_subjects import shard_subjects


@pytest.mark.parametrize("n_subjects", [3, 4, 5])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_repair_multiple_empty_splits(n_subjects, seed):
    subjects = np.arange(n_subjects)
    kwargs = {"split_fracs_dict": {"train": 0.94, "tuning": 0.03, "held_out": 0.03}, "seed": seed}
    shards = shard_subjects(subjects, 10, **kwargs)
    assert {name: len(ids) for name, ids in shards.items()} == {
        "train/0": n_subjects - 2,
        "tuning/0": 1,
        "held_out/0": 1,
    }
    assert sorted(subject for ids in shards.values() for subject in ids) == subjects.tolist()
    assert shard_subjects(subjects, 10, **kwargs) == shards


@pytest.mark.parametrize("n_subjects", [1, 2])
def test_impossible_split_still_raises(n_subjects):
    with pytest.raises(ValueError, match="Unable to adjust splits"):
        shard_subjects(np.arange(n_subjects), 10)
