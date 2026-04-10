"""Auto-discovered stage example tests.

The MEDS-transforms pytest plugin discovers stages registered via entry points, loads their examples/
directories, and parametrizes this test across all (stage, scenario) pairs.
"""

from MEDS_transforms.stages import StageExample


def test_stage_scenario(stage_example: StageExample):
    stage_example.test()
