"""``meds-extract-run`` — the generic dataset-ETL runner (issue #170).

A dataset ETL package under this design is *pure config*: a ``pyproject.toml`` that
registers a name in the ``MEDS_extract.pipelines`` entry-point group, plus one MESSY
YAML carrying ``sources:`` (where the raw data lives), event blocks (what to
extract), and ``etl:`` (how to run it). This package supplies the runner those
packages no longer need to clone:

- :mod:`.registry` — the spec-resolution ladder (registered name → ``pkg://`` →
  filesystem path) and the bundled-MESSY-file convention.
- :mod:`.pipeline` — synthesis of the MEDS-transforms pipeline config from the
  ``etl:`` block and the in-process pipeline invocation (with the
  console-script-PATH healing for MEDS_transforms#398).
- :mod:`.cli` — the ``meds-extract-run`` Hydra entry point tying it together with
  the download layer's :func:`~MEDS_extract.download.api.stage_sources`.
"""

from .registry import PIPELINES_ENTRY_POINT_GROUP, ResolvedSpec, resolve_spec
