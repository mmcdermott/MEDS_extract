"""``StageExample`` subclass for MEDS_extract's raw-ingestion stages.

The upstream :class:`MEDS_transforms.stages.examples.StageExample` validates outputs as either
a :class:`MEDSDataset` (``data/*.parquet`` shards) or a single ``metadata/codes.parquet`` file.
MEDS_extract's early-pipeline stages don't fit either mold:

* ``shard_events`` writes raw row-chunk parquets to ``data/<prefix>/[start-end).parquet``.
* ``split_and_shard_subjects`` writes a JSON file at ``metadata/.shards.json``.
* ``convert_to_subject_sharded``, ``convert_to_MEDS_events``, and ``extract_code_metadata`` all
  write intermediate parquet files whose schemas aren't MEDS-format.

``MEDSExtractStageExample`` handles all of those by declaring inputs AND outputs as
:class:`~pathlib.Path` references to ``yaml_to_disk`` spec files. :meth:`check_outputs`
materializes ``out_data.yaml`` into a temp directory and compares every file in there to the
stage's actual output — ``.parquet`` via :func:`polars.testing.assert_frame_equal`, ``.json``
via parsed-dict equality, ``.yaml`` via parsed-dict equality. Subclasses of this class can add
more suffix handlers as needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import polars as pl
from MEDS_transforms.stages.examples import StageExample
from omegaconf import OmegaConf
from polars.testing import assert_frame_equal
from yaml_to_disk import yaml_disk


@dataclass
class MEDSExtractStageExample(StageExample):
    """StageExample variant that reads inputs/outputs as ``yaml_to_disk`` specs on disk.

    Overrides:

    - :meth:`is_example_dir`: an example dir carries an ``out_data.yaml`` or ``out_metadata.yaml`` file.
    - :meth:`from_dir`: loads ``in.yaml`` and ``out_data.yaml`` / ``out_metadata.yaml`` as
      :class:`~pathlib.Path` references (never as parsed :class:`MEDSDataset` objects), loads
      optional ``cfg.yaml`` + ``pipeline_cfg.yaml`` as plain dicts, and optional ``_test_cfg.yaml``
      as keyword overrides.
    - :meth:`check_outputs`: materializes ``want_data`` / ``want_metadata`` via ``yaml_to_disk``
      and compares each expected file to the actual output by suffix (``.parquet`` →
      :func:`polars.testing.assert_frame_equal`; ``.json``, ``.yaml``, ``.yml`` → parsed-dict
      equality).

    Wiring ``event_conversion_config_fp`` and ``shards_map_fp`` happens through
    ``pipeline_cfg.yaml``: e.g. setting ``event_conversion_config_fp: ${input_dir}/event_cfg.yaml``
    there flows into the auto-generated pipeline YAML that ``MEDS_transform-stage`` reads.
    """

    want_data: Path | None = None
    want_metadata: Path | None = None

    _COMPARE_SUFFIXES: ClassVar[frozenset[str]] = frozenset({".parquet", ".json", ".yaml", ".yml"})
    # Directories emitted by Hydra / pipeline runners that we never want to diff against.
    _SKIP_DIRS: ClassVar[frozenset[str]] = frozenset({".logs", ".hydra"})
    # Stage byproducts that aren't part of what we validate — config copies and the
    # timestamp-bearing dataset.json (created_at changes each run).
    _SKIP_FILES: ClassVar[frozenset[str]] = frozenset({"event_conversion_config.yaml", "dataset.json"})

    def __post_init__(self):
        if self.want_data is None and self.want_metadata is None:
            raise ValueError("Either want_data or want_metadata must be provided.")
        if self.want_data is not None and self.want_metadata is not None:
            raise ValueError("Either want_data or want_metadata must be provided, but not both.")
        if self.scenario_name == ".":
            self.scenario_name = None
        if self.pipeline_cfg:
            self.do_use_config_yaml = True
        if self.df_check_kwargs is None:
            self.df_check_kwargs = {"check_dtypes": False, "check_column_order": False}

    @classmethod
    def is_example_dir(cls, path: Path) -> bool:
        return (path / "out_data.yaml").is_file() or (path / "out_metadata.yaml").is_file()

    @classmethod
    def from_dir(
        cls,
        stage_name: str,
        scenario_name: str,
        example_dir: Path,
        **schema_updates,
    ) -> MEDSExtractStageExample:
        """Load an example from ``<stage>/examples/<scenario>/``.

        Expected files (all optional except at least one of ``out_data.yaml`` / ``out_metadata.yaml``):

        * ``in.yaml`` — yaml_to_disk spec materializing the stage's input tree
        * ``out_data.yaml`` — yaml_to_disk spec describing expected ``data/`` tree
        * ``out_metadata.yaml`` — yaml_to_disk spec describing expected ``metadata/`` tree
        * ``cfg.yaml`` — stage-specific config (maps onto ``stage_cfg``)
        * ``pipeline_cfg.yaml`` — top-level pipeline overrides (e.g. ``event_conversion_config_fp``)
        * ``_test_cfg.yaml`` — kwargs forwarded to the dataclass constructor (e.g.
          ``df_check_kwargs``)
        """
        in_fp = example_dir / "in.yaml"
        want_data_fp = example_dir / "out_data.yaml"
        want_metadata_fp = example_dir / "out_metadata.yaml"
        cfg_fp = example_dir / "cfg.yaml"
        pipeline_cfg_fp = example_dir / "pipeline_cfg.yaml"
        test_cfg_fp = example_dir / "_test_cfg.yaml"

        stage_cfg = OmegaConf.to_container(OmegaConf.load(cfg_fp)) if cfg_fp.is_file() else {}
        pipeline_cfg = (
            OmegaConf.to_container(OmegaConf.load(pipeline_cfg_fp)) if pipeline_cfg_fp.is_file() else {}
        )
        test_kwargs = OmegaConf.to_container(OmegaConf.load(test_cfg_fp)) if test_cfg_fp.is_file() else {}

        return cls(
            stage_name=stage_name,
            scenario_name=scenario_name,
            stage_cfg=stage_cfg,
            pipeline_cfg=pipeline_cfg,
            in_data=in_fp if in_fp.is_file() else None,
            want_data=want_data_fp if want_data_fp.is_file() else None,
            want_metadata=want_metadata_fp if want_metadata_fp.is_file() else None,
            **test_kwargs,
        )

    def check_outputs(self, output_dir: Path, is_resolved_dir: bool = False) -> None:
        """Compare the stage's actual output tree to the expected tree from ``want_data`` /
        ``want_metadata``.

        When ``is_resolved_dir`` is set (the ``pipeline_tester`` case), the caller has already
        resolved ``output_dir`` to ``${cohort}/<stage_name>/``, so any leading ``data/`` /
        ``metadata/`` segments in the expected paths are absorbed by that resolution and
        must be stripped before the file-by-file diff.
        """
        want_fp = self.want_data if self.want_data is not None else self.want_metadata
        strip_top = {"data", "metadata"} if is_resolved_dir else set()

        with yaml_disk(want_fp) as expected_root:
            expected = {}
            for fp in expected_root.rglob("*"):
                if not fp.is_file() or fp.suffix not in self._COMPARE_SUFFIXES:
                    continue
                rel = fp.relative_to(expected_root)
                if rel.parts and rel.parts[0] in strip_top:
                    rel = Path(*rel.parts[1:])
                expected[rel] = fp

            # Scan only the top-level directories the expected spec references — keeps the diff
            # from tripping on upstream-emitted metadata/hydra artifacts.
            top_dirs = {rel.parts[0] for rel in expected if rel.parts}
            actual = {}
            for top in top_dirs:
                search_root = output_dir / top if (output_dir / top).is_dir() else output_dir
                for fp in search_root.rglob("*"):
                    if not fp.is_file() or fp.suffix not in self._COMPARE_SUFFIXES:
                        continue
                    rel = fp.relative_to(output_dir)
                    # Drop hydra/log directories and stage-byproduct files from the actual-set
                    # so extraneous framework artifacts don't blow up the diff.
                    if any(part in self._SKIP_DIRS for part in rel.parts):
                        continue
                    if fp.name in self._SKIP_FILES:
                        continue
                    actual[rel] = fp

            missing = sorted(expected.keys() - actual.keys())
            if missing:
                raise AssertionError(
                    f"Output file set mismatch under {output_dir}:\n"
                    f"  Missing:    {missing}\n"
                    f"  Found:      {sorted(actual.keys())}"
                )
            # Unexpected extras (mapper intermediates like `{prefix}_0.parquet`, scratch files, etc.)
            # are intentionally tolerated so stages can freely emit byproducts.

            for rel in sorted(expected):
                _compare(expected[rel], actual[rel], rel, self.df_check_kwargs or {})

    def render_content(self, example_dir: Path | None = None) -> list[str]:
        """Override ``StageExample.render_content`` to render ``want_metadata`` as a YAML
        code block when it's a :class:`~pathlib.Path`.

        The upstream default assumes ``want_metadata`` is a :class:`polars.DataFrame` and calls
        ``df_to_markdown``, which blows up on a path. For MEDS_extract stages whose outputs are
        declared as ``out_metadata.yaml`` specs, we emit the YAML source under an
        ``**Expected output metadata:**`` heading — mirroring the built-in ``want_data``-as-path
        handling for data stages.
        """
        if isinstance(self.want_metadata, Path):
            saved = self.want_metadata
            self.want_metadata = None
            lines = super().render_content(example_dir)
            self.want_metadata = saved
            insert_idx = next(
                (i for i, line in enumerate(lines) if line.startswith("**Run this stage:**")),
                len(lines),
            )
            lines[insert_idx:insert_idx] = [
                "**Expected output metadata:**",
                "",
                "```yaml",
                saved.read_text().strip(),
                "```",
                "",
            ]
            return lines
        return super().render_content(example_dir)


def _compare(expected_fp: Path, actual_fp: Path, rel: Path, df_check_kwargs: dict) -> None:
    """Per-file comparison dispatcher keyed off the expected file's suffix.

    ``.parquet`` uses :func:`polars.testing.assert_frame_equal`; ``.json`` / ``.yaml`` /
    ``.yml`` parse both sides and compare the resulting Python objects. The caller is
    responsible for filtering to supported suffixes; reaching the ``_`` branch is a
    programming error.
    """
    match expected_fp.suffix:
        case ".parquet":
            kwargs = {"check_column_order": False, "check_dtypes": False, **df_check_kwargs}
            # `glob=False` because shard_events names files `[start-end).parquet` — the
            # brackets are glob metacharacters in polars' default reader.
            got = pl.read_parquet(actual_fp, glob=False)
            want = pl.read_parquet(expected_fp, glob=False)
            try:
                assert_frame_equal(
                    got.sort(want.columns) if set(got.columns) == set(want.columns) else got,
                    want.sort(want.columns),
                    **kwargs,
                )
            except AssertionError as e:
                raise AssertionError(f"Parquet {rel} differs.\nGot:\n{got}\nWant:\n{want}") from e
        case ".json":
            got = json.loads(actual_fp.read_text())
            want = json.loads(expected_fp.read_text())
            _assert_struct_equal(got, want, rel)
        case ".yaml" | ".yml":
            got = OmegaConf.to_container(OmegaConf.load(actual_fp))
            want = OmegaConf.to_container(OmegaConf.load(expected_fp))
            _assert_struct_equal(got, want, rel)
        case _:
            raise AssertionError(f"Unsupported output suffix '{expected_fp.suffix}' for {rel}.")


def _assert_struct_equal(got: object, want: object, rel: Path) -> None:
    """Compare two structured blobs (dicts/lists), treating top-level dict values as sets when
    they are lists of hashables.

    Motivated by ``.shards.json`` — ``split_and_shard_subjects`` output where per-split subject
    lists aren't deterministically ordered run-to-run. Nested dicts recurse; anything else is a
    strict ``==`` comparison.
    """
    if isinstance(got, dict) and isinstance(want, dict):
        got_keys = set(got.keys())
        want_keys = set(want.keys())
        if got_keys != want_keys:
            raise AssertionError(
                f"{rel}: key mismatch.\n  Missing: {sorted(want_keys - got_keys)}\n"
                f"  Extra:   {sorted(got_keys - want_keys)}"
            )
        for k in want_keys:
            gv, wv = got[k], want[k]
            if _is_hashable_list(gv) and _is_hashable_list(wv):
                if set(gv) != set(wv):
                    raise AssertionError(
                        f"{rel}[{k!r}]: contents differ (unordered).\n  Got:  {sorted(gv)}\n"
                        f"  Want: {sorted(wv)}"
                    )
            else:
                _assert_struct_equal(gv, wv, rel)
        return
    if got != want:
        raise AssertionError(f"{rel}: differs.\n  Got:  {got!r}\n  Want: {want!r}")


def _is_hashable_list(v: object) -> bool:
    if not isinstance(v, list):
        return False
    try:
        set(v)
    except TypeError:
        return False
    return True
