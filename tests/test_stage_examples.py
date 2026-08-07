"""Run every registered MEDS_extract stage example end-to-end.

The `stage_example` fixture is auto-parametrized by `MEDS_transforms.pytest_plugin` across all
`(stage, scenario)` pairs in the `MEDS_extract` package, so this file only needs to stand up one
test function. A stage migrates onto this machinery by placing `in.yaml` plus either
`out_data.yaml` (for data stages) or `out_metadata.yaml` (for metadata stages) under
`<stage>/examples/<scenario>/`; `Stage.register` auto-infers `examples_dir` from the stage's
own directory.
"""


def test_stage_example(stage_example):
    stage_example.test()
