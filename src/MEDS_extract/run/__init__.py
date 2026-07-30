"""``meds-extract-run`` — the generic dataset-ETL runner (issue #170).

A dataset ETL package under this design is *pure config*: a ``pyproject.toml`` that
registers a name in the ``MEDS_extract.pipelines`` entry-point group, plus one MESSY
YAML carrying ``sources:`` (where the raw data lives, including its
``dataset_version``), event blocks (what to extract), and an optional ``etl:`` block
(identity fallbacks + curated stage options). This package supplies the runner those
packages no longer need to clone:

- :mod:`.registry` — the spec-resolution ladder (registered name → ``pkg://`` →
  filesystem path) and the bundled-MESSY-file convention.
- :mod:`.pipeline` — synthesis of the MEDS-transforms pipeline config from the
  ``etl:`` block, and healed-PATH subprocess spawning (the
  console-script-resolution fix for MEDS_transforms#398).
- :mod:`.cli` — the ``meds-extract-run`` Hydra entry point: it shells out to
  ``meds-extract-download`` and ``MEDS_transform-pipeline`` in turn.
"""

from .registry import PIPELINES_ENTRY_POINT_GROUP, ResolvedSpec, resolve_spec
