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

    Examples:
        Construction requires exactly one of ``want_data`` / ``want_metadata``:

        >>> MEDSExtractStageExample(stage_name="ex")
        Traceback (most recent call last):
            ...
        ValueError: Either want_data or want_metadata must be provided.
        >>> MEDSExtractStageExample(stage_name="ex", want_data=Path("a"), want_metadata=Path("b"))
        Traceback (most recent call last):
            ...
        ValueError: Either want_data or want_metadata must be provided, but not both.

        The ``"."`` scenario name (used by singleton example dirs) is canonicalized to ``None``,
        and setting a ``pipeline_cfg`` implicitly sets ``do_use_config_yaml``:

        >>> ex = MEDSExtractStageExample(
        ...     stage_name="ex", scenario_name=".", want_data=Path("a.yaml"),
        ...     pipeline_cfg={"event_conversion_config_fp": "x"},
        ... )
        >>> print(ex.scenario_name)
        None
        >>> ex.do_use_config_yaml
        True

        The default :attr:`df_check_kwargs` is loose about dtypes and column order so that
        ``yaml_to_disk``-materialized expected frames (which can't carry polars dtypes) diff
        cleanly against actual parquet outputs:

        >>> ex = MEDSExtractStageExample(stage_name="ex", want_data=Path("a.yaml"))
        >>> ex.df_check_kwargs
        {'check_dtypes': False, 'check_column_order': False}
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
        """Compare the stage's actual output tree to the expected tree from ``want_data`` / ``want_metadata``.

        When ``is_resolved_dir`` is set (the ``pipeline_tester`` case), the caller has already
        resolved ``output_dir`` to ``${cohort}/<stage_name>/``, so any leading ``data/`` /
        ``metadata/`` segments in the expected paths are absorbed by that resolution and must
        be stripped before the file-by-file diff.

        In pipeline mode, some stages write to **globally-shared pipeline paths** rather than
        their own ``stage_cfg.output_dir`` — e.g. ``split_and_shard_subjects`` writes
        ``metadata/.shards.json`` via the pipeline-level ``shards_map_fp`` config, which by
        convention lives at ``${output_dir}/metadata/.shards.json`` (the cohort root). When
        ``is_resolved_dir`` is set and a file isn't found under ``output_dir``, the search
        falls back to ``output_dir.parent`` (the cohort root) with the full unstripped
        relative path — matching the canonical MEDS_extract pipeline layout.
        """
        want_fp = self.want_data if self.want_data is not None else self.want_metadata
        strip_top = {"data", "metadata"} if is_resolved_dir else set()

        with yaml_disk(want_fp) as expected_root:
            stage_expected: dict[Path, Path] = {}
            global_expected: dict[Path, Path] = {}
            for fp in expected_root.rglob("*"):
                if not fp.is_file() or fp.suffix not in self._COMPARE_SUFFIXES:
                    continue
                full_rel = fp.relative_to(expected_root)
                stripped = Path(*full_rel.parts[1:]) if full_rel.parts[0] in strip_top else full_rel
                stage_expected[stripped] = fp
                global_expected[full_rel] = fp

            stage_top_dirs = {rel.parts[0] for rel in stage_expected if rel.parts}
            stage_actual = self._collect_actual(output_dir, stage_top_dirs)
            global_actual: dict[Path, Path] = {}
            if is_resolved_dir:
                global_actual = self._collect_actual(
                    output_dir.parent,
                    {rel.parts[0] for rel in global_expected if rel.parts},
                )

            found: dict[Path, tuple[Path, Path]] = {}  # rel_in_expected → (expected_fp, actual_fp)
            missing: list[Path] = []
            for rel, exp_fp in stage_expected.items():
                if rel in stage_actual:
                    found[rel] = (exp_fp, stage_actual[rel])
                    continue
                # Fallback: this file may live at the pipeline root (e.g. shards_map_fp).
                full_rel = next(r for r, f in global_expected.items() if f == exp_fp)
                if full_rel in global_actual:
                    found[rel] = (exp_fp, global_actual[full_rel])
                else:
                    missing.append(rel)

            if missing:
                raise AssertionError(
                    f"Output file set mismatch under {output_dir}:\n"
                    f"  Missing:    {sorted(missing)}\n"
                    f"  Found in stage dir:  {sorted(stage_actual.keys())}\n"
                    f"  Found in cohort root: {sorted(global_actual.keys())}"
                )
            # Unexpected extras (mapper intermediates like `{prefix}_0.parquet`, scratch files, etc.)
            # are intentionally tolerated so stages can freely emit byproducts.

            for rel in sorted(found):
                exp_fp, act_fp = found[rel]
                _compare(exp_fp, act_fp, rel, self.df_check_kwargs or {})

    def _collect_actual(self, root: Path, top_dirs: set[str]) -> dict[Path, Path]:
        """Walk ``top_dirs`` under ``root`` and return ``{rel: fp}`` for comparable files.

        Skips :attr:`_SKIP_DIRS` and :attr:`_SKIP_FILES` so hydra logs and stage-byproduct
        config copies don't blow up the diff.
        """
        actual: dict[Path, Path] = {}
        for top in top_dirs:
            search_root = root / top if (root / top).is_dir() else root
            for fp in search_root.rglob("*"):
                if not fp.is_file() or fp.suffix not in self._COMPARE_SUFFIXES:
                    continue
                rel = fp.relative_to(root)
                if any(part in self._SKIP_DIRS for part in rel.parts):
                    continue
                if fp.name in self._SKIP_FILES:
                    continue
                actual[rel] = fp
        return actual

    def render_content(self, example_dir: Path | None = None) -> list[str]:
        """Override ``StageExample.render_content`` to render ``want_metadata`` as a YAML code block when it's
        a :class:`~pathlib.Path`.

        The upstream default assumes ``want_metadata`` is a :class:`polars.DataFrame` and calls
        ``df_to_markdown``, which blows up on a path. For MEDS_extract stages whose outputs are
        declared as ``out_metadata.yaml`` specs, we emit the YAML source under an
        ``**Expected output metadata:**`` heading — mirroring the built-in ``want_data``-as-path
        handling for data stages.

        Examples:
            Metadata-Path examples get an ``**Expected output metadata:**`` block rendered
            inline above the shell-invocation hint:

            >>> with yaml_disk({"out_metadata.yaml": "metadata/codes.parquet:\\n  code: [HR]\\n"}) as d:
            ...     ex = MEDSExtractStageExample(stage_name="demo", want_metadata=d / "out_metadata.yaml")
            ...     rendered = "\\n".join(ex.render_content())
            >>> "**Expected output metadata:**" in rendered
            True
            >>> "metadata/codes.parquet:" in rendered
            True
            >>> "MEDS_transform-stage" in rendered
            True

            When ``want_metadata`` isn't a Path the upstream default applies unchanged:

            >>> import polars as pl
            >>> df = pl.DataFrame({"code": ["HR"], "description": ["Heart Rate"]})
            >>> ex = MEDSExtractStageExample(stage_name="demo", want_metadata=df)
            >>> "**Expected output metadata:**" in "\\n".join(ex.render_content())
            True
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

    Examples:
        Matching parquets pass silently:

        >>> with yaml_disk({"a.parquet": {"x": [1, 2]}, "b.parquet": {"x": [1, 2]}}) as d:
        ...     _compare(d / "a.parquet", d / "b.parquet", Path("x.parquet"), {})

        Differing parquets raise with the rel path and both frames in the message:

        >>> with yaml_disk({"a.parquet": {"x": [1, 2]}, "b.parquet": {"x": [1, 999]}}) as d:
        ...     _compare(d / "a.parquet", d / "b.parquet", Path("shard/0.parquet"), {})
        Traceback (most recent call last):
            ...
        AssertionError: Parquet shard/0.parquet differs...

        JSON comparison uses the recursive struct-equal helper (so e.g. ``.shards.json``
        hashable-list values diff order-independently):

        >>> import json, tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     a, b = Path(d) / "a.json", Path(d) / "b.json"
        ...     _ = a.write_text(json.dumps({"train/0": [1, 2]}))
        ...     _ = b.write_text(json.dumps({"train/0": [2, 1]}))
        ...     _compare(a, b, Path("shards.json"), {})

        YAML files dispatch through the same struct-equal helper:

        >>> with yaml_disk({"a.yaml": "foo: 1\\n", "b.yaml": "foo: 1\\n"}) as d:
        ...     _compare(d / "a.yaml", d / "b.yaml", Path("cfg.yaml"), {})
        >>> with yaml_disk({"a.yaml": "foo: 1\\n", "b.yaml": "foo: 2\\n"}) as d:
        ...     _compare(d / "a.yaml", d / "b.yaml", Path("cfg.yaml"), {})
        Traceback (most recent call last):
            ...
        AssertionError: cfg.yaml: differs.
          Got:  2
          Want: 1

        Unsupported suffixes raise — the caller should have filtered these out:

        >>> with yaml_disk({"a.txt": "hello", "b.txt": "hello"}) as d:
        ...     _compare(d / "a.txt", d / "b.txt", Path("x.txt"), {})
        Traceback (most recent call last):
            ...
        AssertionError: Unsupported output suffix '.txt' for x.txt.
    """
    match expected_fp.suffix:
        case ".parquet":
            # Defaults below can be overridden via the caller's `df_check_kwargs`. In
            # particular, `check_row_order=False` is the package-level default because most
            # MEDS_extract stages pass input through `rwlock_wrap` (which shuffles) or
            # concatenate multiple source shards non-deterministically; callers that DO
            # need strict ordering can opt in by passing `check_row_order: True` in
            # `_test_cfg.yaml`. Polars' own implementation of `check_row_order=False` sorts
            # both frames internally before diffing — no local sort hack needed.
            defaults = {"check_column_order": False, "check_dtypes": False, "check_row_order": False}
            kwargs = {**defaults, **df_check_kwargs}
            # `glob=False` because shard_events names files `[start-end).parquet` — the
            # brackets are glob metacharacters in polars' default reader.
            got = pl.read_parquet(actual_fp, glob=False)
            want = pl.read_parquet(expected_fp, glob=False)
            try:
                assert_frame_equal(got, want, **kwargs)
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
    """Compare two structured blobs (dicts/lists) recursively, treating dict values as unordered sets whenever
    they are lists of hashables at **any** nesting level.

    Motivated by ``.shards.json`` — ``split_and_shard_subjects`` output where per-split subject
    lists aren't deterministically ordered run-to-run. The unordered-list semantics apply
    recursively rather than only at the top level because nested pipeline-output YAMLs can
    carry similarly non-deterministic lists (e.g. a future ``{split: {shard: [subjects]}}``
    layout). Plain dicts recurse; anything else is a strict ``==`` comparison.

    Examples:
        Hashable-list dict values compare order-independently:

        >>> _assert_struct_equal({"x": [1, 2, 3]}, {"x": [3, 2, 1]}, Path("shards.json"))

        Missing/extra keys raise:

        >>> _assert_struct_equal({"a": 1, "b": 2}, {"a": 1}, Path("shards.json"))
        Traceback (most recent call last):
            ...
        AssertionError: shards.json: key mismatch.
          Missing: []
          Extra:   ['b']

        Differing hashable-list contents raise with both sides sorted for readability:

        >>> _assert_struct_equal({"x": [1, 2]}, {"x": [1, 3]}, Path("shards.json"))
        Traceback (most recent call last):
            ...
        AssertionError: shards.json['x']: contents differ (unordered).
          Got:  [1, 2]
          Want: [1, 3]

        Nested dicts recurse:

        >>> _assert_struct_equal({"a": {"b": [1, 2]}}, {"a": {"b": [2, 1]}}, Path("x.json"))

        Non-dict scalar mismatch raises:

        >>> _assert_struct_equal("foo", "bar", Path("x.json"))
        Traceback (most recent call last):
            ...
        AssertionError: x.json: differs.
          Got:  'foo'
          Want: 'bar'
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
    """True iff ``v`` is a :class:`list` of hashable elements.

    Used by :func:`_assert_struct_equal` to decide whether a dict value can be compared
    order-independently via a set.

    Examples:
        >>> _is_hashable_list([1, 2, 3])
        True
        >>> _is_hashable_list(["a", "b"])
        True
        >>> _is_hashable_list([])
        True
        >>> _is_hashable_list([[1, 2], [3]])  # inner lists are unhashable
        False
        >>> _is_hashable_list("not a list")
        False
        >>> _is_hashable_list({"x": 1})
        False
    """
    if not isinstance(v, list):
        return False
    try:
        set(v)
    except TypeError:
        return False
    return True
