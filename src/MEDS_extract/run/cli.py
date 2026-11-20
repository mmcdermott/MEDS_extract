"""``meds-extract-run`` — CLI entry point for the generic dataset-ETL runner.

One command runs a whole dataset ETL from its MESSY spec::

    meds-extract-run spec=MIMIC-IV output_dir=/data/mimic_meds dataset_key=demo
    meds-extract-run spec=pkg://MIMIC_IV_MEDS.configs.event_configs.yaml output_dir=...
    meds-extract-run spec=/path/to/messy.yaml output_dir=... do_download=false input_dir=...

The CLI itself just shuttles commands: it loads the spec into the one MESSY config
object (:meth:`~MEDS_extract.config.MessyConfig.load` — resolution ladder,
validation, and identity defaulting all live there), then spawns
``meds-extract-download`` and ``MEDS_transform-pipeline`` in turn via
:func:`run_command`, propagating child exit codes. The synthesized pipeline config
(written under ``<output_dir>/.meds_extract_run/``, every value inlined) is the
only channel through which the computed identity reaches the pipeline. Its stage
list is the canonical sequence plus the spec's extension stages (``etl.pre_stages`` /
``etl.post_stages``) minus ``skip_stages=``; before spawning the pipeline, the CLI
fits the pipeline runner's done markers to that list (:func:`reconcile_done_markers`), and
after a successful run it puts back the markers the runner's ``cleanup_intermediates`` deleted
(:func:`restore_cleaned_markers`), so a cleaned-up run still counts as finished.

Both children are spawned as ``sys.executable -m <module>`` — the canonical way to
pin a subprocess to the calling interpreter's environment, with no console-script
``PATH`` resolution to heal or mis-resolve (mmcdermott/MEDS_transforms#398 made the
pipeline runner ``python -m``-runnable in MEDS-transforms 0.7.0; the download CLI
has its own ``__main__`` guard).

Exits ``0`` on full success; config errors exit ``1``, child failures propagate the
child's exit code — via explicit :func:`sys.exit`, since Hydra discards the task
function's return value.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import field, fields
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from hydra.core.override_parser.types import Quote, QuotedString
from MEDS_transforms.configs.utils import hydra_registered_dataclass
from omegaconf import MISSING, DictConfig, OmegaConf

from .._cli import require_dotlist_args
from ..config import MessyConfig, user_local_path

if TYPE_CHECKING:
    from collections.abc import Collection

logger = logging.getLogger(__name__)


def _hydra_quote(value: str) -> str:
    r"""Render ``value`` as a single-quoted string literal in Hydra's override grammar.

    Hydra's override grammar gives bare ``,`` and ``=`` structural meaning (a choice
    sweep and the key/value separator), so a filesystem path interpolated verbatim
    into a dotlist override breaks the parse. Inside the grammar's single-quoted
    form every character is literal except the quote itself: an embedded ``'`` is
    escaped as ``\'``, and a run of backslashes immediately before a quote is
    doubled (backslashes elsewhere stay literal). Hydra's own
    :class:`~hydra.core.override_parser.types.QuotedString` implements exactly
    those rules, so quoting here always matches what the child's parser accepts.

    Examples:
        >>> _hydra_quote("/data/comma,dir/spec.yaml")
        "'/data/comma,dir/spec.yaml'"
        >>> _hydra_quote("/data/eq=dir/out")
        "'/data/eq=dir/out'"
        >>> _hydra_quote("/data/it's here")
        "'/data/it\\'s here'"
    """
    return QuotedString(text=value, quote=Quote.single).with_quotes()


def run_command(argv: list[str]) -> int:
    """Spawn ``sys.executable -m argv[0]`` with the remaining args; return the exit code.

    ``argv[0]`` is a module name, not a console-script name: ``python -m`` pins the
    child to this interpreter's environment with no ``PATH`` resolution to heal or
    mis-resolve (the pattern pip's docs recommend for exactly this). Output streams
    straight through to this process's stdout/stderr.
    """
    cmd = [sys.executable, "-m", *argv]
    logger.info(f"Running: {cmd}")
    return subprocess.run(cmd, check=False).returncode


def stage_name(entry: str | dict) -> str:
    """The name of one ``stages:`` entry (a bare name or a one-key ``{name: config}`` mapping).

    Examples:
        >>> stage_name("convert_to_parquet"), stage_name({"enrich_codes": {"rules": "X"}})
        ('convert_to_parquet', 'enrich_codes')
    """
    return entry if isinstance(entry, str) else next(iter(entry))


# Kept in ``<output_dir>/.logs``: the stages whose output dirs the pipeline runner's
# ``cleanup_intermediates`` deleted, and whose done markers this CLI then restored. See
# :func:`restore_cleaned_markers` and :func:`reconcile_done_markers`.
CLEANED_STAGES_FILE = "_cleaned_stages.json"


def read_cleaned_stages(log_dir: Path) -> list[str]:
    """The recorded stages whose output dir is still missing (see :func:`restore_cleaned_markers`).

    Examples:
        >>> log_dir = getfixture("tmp_path") / ".logs"
        >>> read_cleaned_stages(log_dir)
        []
        >>> log_dir.mkdir()
        >>> _ = (log_dir / CLEANED_STAGES_FILE).write_text('["convert_to_parquet", "merge_to_MEDS_cohort"]')
        >>> (log_dir.parent / "merge_to_MEDS_cohort").mkdir()  # it ran again since, and kept its dir
        >>> read_cleaned_stages(log_dir)
        ['convert_to_parquet']
    """
    fp = log_dir / CLEANED_STAGES_FILE
    if not fp.exists():
        return []
    return [s for s in json.loads(fp.read_text()) if not (log_dir.parent / s).is_dir()]


def restore_cleaned_markers(log_dir: Path, stages: list[str]) -> None:
    """After a successful pipeline run, mark the stages cleanup removed as done again, and record them.

    The pipeline runner marks every stage it runs as done, so after a successful run a stage
    of this run without a marker is one whose ``<output_dir>/<stage>`` dir and marker the
    runner's ``cleanup_intermediates`` deleted. Without its marker, the next
    :func:`reconcile_done_markers` would take the stage for one that never ran and resume
    the whole pipeline. So its marker comes back, and the stage is added to
    :data:`CLEANED_STAGES_FILE`: a later run that re-runs an earlier stage knows its output is
    gone. A recorded stage whose dir exists again (it ran again without cleanup) is dropped.

    Examples:
        >>> log_dir = getfixture("tmp_path") / ".logs"
        >>> log_dir.mkdir()
        >>> stages = ["convert_to_parquet", "finalize_MEDS_data"]
        >>> (log_dir / "finalize_MEDS_data.done").touch()  # writes to data/, so cleanup leaves it
        >>> restore_cleaned_markers(log_dir, stages)
        >>> sorted(p.name for p in log_dir.iterdir())
        ['_cleaned_stages.json', 'convert_to_parquet.done', 'finalize_MEDS_data.done']
        >>> read_cleaned_stages(log_dir)
        ['convert_to_parquet']

        A run without cleanup leaves no record:

        >>> (log_dir.parent / "convert_to_parquet").mkdir()
        >>> restore_cleaned_markers(log_dir, stages)
        >>> (log_dir / CLEANED_STAGES_FILE).exists()
        False
    """
    cleaned = read_cleaned_stages(log_dir)
    for stage in stages:
        if not (marker := log_dir / f"{stage}.done").exists():
            logger.info(f"Marking {stage} done again: cleanup removed its output after the run succeeded.")
            marker.touch()
            if stage not in cleaned:
                cleaned.append(stage)
    fp = log_dir / CLEANED_STAGES_FILE
    if cleaned:
        fp.write_text(json.dumps(cleaned))
    else:
        fp.unlink(missing_ok=True)


def reconcile_done_markers(log_dir: Path, stages: list[str], post_stages: Collection[str]) -> None:
    """Make the pipeline runner's done markers fit this run's stage list before it starts.

    The MEDS-transforms runner skips a stage whose ``<log_dir>/<stage>.done`` exists,
    and exits at once when ``_all_stages.done`` does. Three situations need help:

    - The global marker exists, but a stage of this run has no marker of its own: a
      post-stage was added to (or a ``skip_stages`` run is being completed on) a
      finished cohort, or a stage's marker was deleted so it re-runs. The global
      marker is removed, so the runner resumes and runs exactly the unmarked stages.
    - A stage that runs before a post-stage rewrites the cohort that post-stage
      edited in place, so every post-stage after the first unmarked stage loses its
      marker too and re-applies its edit.
    - A stage other than a post-stage runs again, but cleanup (``cleanup_intermediates``)
      deleted the output of stages before it (:func:`restore_cleaned_markers` recorded
      them). Those stages lose their markers too and run again, so it has its inputs.
      Post-stages read only the cohort, so a post-stage alone needs no such help.

    A finished run with nothing to add is left untouched, so it still exits at once.

    Examples:
        >>> log_dir = getfixture("tmp_path")
        >>> stages = ["convert_to_parquet", "finalize_MEDS_metadata", "enrich_codes"]
        >>> def markers():
        ...     return sorted(p.stem for p in log_dir.glob("*.done"))

        A finished run is left as it is:

        >>> for s in [*stages, "_all_stages"]:
        ...     (log_dir / f"{s}.done").touch()
        >>> reconcile_done_markers(log_dir, stages, post_stages={"enrich_codes"})
        >>> markers()
        ['_all_stages', 'convert_to_parquet', 'enrich_codes', 'finalize_MEDS_metadata']

        A post-stage added to a finished cohort runs alone:

        >>> (log_dir / "enrich_codes.done").unlink()
        >>> reconcile_done_markers(log_dir, stages, post_stages={"enrich_codes"})
        >>> markers()
        ['convert_to_parquet', 'finalize_MEDS_metadata']

        A canonical stage that re-runs takes the post-stages after it along:

        >>> (log_dir / "enrich_codes.done").touch()
        >>> (log_dir / "finalize_MEDS_metadata.done").unlink()
        >>> reconcile_done_markers(log_dir, stages, post_stages={"enrich_codes"})
        >>> markers()
        ['convert_to_parquet']

        After a cleaned-up run, a post-stage still runs alone, but a canonical stage that
        re-runs takes the cleaned stages before it along, since their output is gone:

        >>> for s in [*stages, "_all_stages"]:
        ...     (log_dir / f"{s}.done").touch()
        >>> _ = (log_dir / CLEANED_STAGES_FILE).write_text(json.dumps(stages))
        >>> (log_dir / "enrich_codes.done").unlink()
        >>> reconcile_done_markers(log_dir, stages, post_stages={"enrich_codes"})
        >>> markers()
        ['convert_to_parquet', 'finalize_MEDS_metadata']
        >>> (log_dir / "finalize_MEDS_metadata.done").unlink()
        >>> reconcile_done_markers(log_dir, stages, post_stages={"enrich_codes"})
        >>> markers()
        []

        A fresh output dir has no log dir yet, which is fine:

        >>> reconcile_done_markers(log_dir / "fresh", stages, post_stages={"enrich_codes"})
    """
    pending = [s for s in stages if not (log_dir / f"{s}.done").exists()]
    if not pending:
        return
    if rerun := next((s for s in pending if s not in post_stages), None):
        cleaned = read_cleaned_stages(log_dir)
        for stage in stages[: stages.index(rerun)]:
            if stage in cleaned and (marker := log_dir / f"{stage}.done").exists():
                logger.info(f"Re-running {stage}: cleanup removed its output, which {rerun} needs.")
                marker.unlink()
    first = next(s for s in stages if not (log_dir / f"{s}.done").exists())
    for stage in stages[stages.index(first) + 1 :]:
        if stage in post_stages and (marker := log_dir / f"{stage}.done").exists():
            logger.info(f"Re-running post-stage {stage}: an earlier stage runs again first.")
            marker.unlink()
    if (all_done := log_dir / "_all_stages.done").exists():
        logger.info(f"Resuming a finished pipeline to run {pending}.")
        all_done.unlink()


@hydra_registered_dataclass(group=None, name="run_defaults")
class RunConfig:
    """Typed config for ``meds-extract-run``.

    Fields:
        spec: What to run — a name registered in the ``MEDS_extract.pipelines``
            entry-point group, a ``pkg://`` reference, or a path to a MESSY file.
        output_dir: Where the FINAL MEDS cohort lives (``data/``, ``metadata/``).
            Run-internal artifacts (the synthesized pipeline config, the download
            child's Hydra run dir) live under the derived :attr:`work_dir`
            (``<output_dir>/.meds_extract_run``), alongside the pipeline's own
            ``.logs``/intermediate stage outputs.
        do_download: Whether to run the download stage at all. ``False`` spawns
            no download child (``input_dir`` is then required).
        dataset_key: Which ``sources:`` bucket is in play (``dataset`` /
            ``demo`` / ...). It selects what the download child fetches
            (``common`` is always appended) AND which entry of a mapping-form
            ``sources.dataset_version`` is stamped into the output's
            ``etl_metadata.dataset_version`` — always, whether or not the
            download runs, so a pre-staged demo run
            (``do_download=false dataset_key=demo input_dir=...``) stamps the
            demo version.
        download_dest_dir: Where to download raw data (only with
            ``do_download=true``); it is then also the pipeline's effective
            input. Defaults under :attr:`work_dir` — point it somewhere durable
            to keep raw data across output trees.
        input_dir: Where pre-staged raw input data lives (only with
            ``do_download=false``).
        dataset_version: Explicit override for ``etl_metadata.dataset_version``
            (default: computed — see ``MessyConfig.dataset_version_for``).
        skip_stages: Extension stages (the spec's ``etl.pre_stages`` /
            ``etl.post_stages``) to leave out of this run, e.g.
            ``skip_stages=[enrich_codes]``. Naming a canonical stage, or a name the spec
            does not declare, is a config error. A skipped stage need not be installed.

    Passthroughs to the two children. The runner is a shuttle, so these add no
    semantics of their own — each is forwarded verbatim and documented by the child
    that consumes it:

    Fields:
        stage_runner_fp: Path to a MEDS-transforms stage-runner file, forwarded as
            ``--stage_runner_fp``. This is the parallelism knob: the runner reads a
            top-level ``parallelize`` block out of that file as every stage's default
            (and it may override ``parallelize`` or ``script`` per stage), so
            ``parallelize: {n_workers: 8, launcher: joblib}`` in a two-line file makes
            a whole pipeline parallel. Deliberately a *runner* argument rather than an
            ``etl:`` option: worker counts are a property of the machine, not of the
            dataset, and a registered spec ships inside a wheel.
        do_profile: Forwarded as ``--do_profile``. The pipeline runner consumes this
            itself and appends the profiler callback to each stage command, so it is
            reachable no other way (it is neither a pipeline-config key nor a Hydra
            override of that entry point).
        overrides: Extra pipeline-config overrides, forwarded as ``--overrides``.
            The escape valve for every pipeline key the synthesized config does not
            template — ``seed``, pipeline-level ``do_overwrite``, and anything a future
            MEDS-transforms release adds — so new keys need no change here. Quote each
            element on the command line, since the values themselves contain ``=``::

                meds-extract-run ... "overrides=['seed=2','do_overwrite=True']"

            An override also wins over a pipeline key the spec sets in its ``etl:``
            block: ``"overrides=['cleanup_intermediates=false']"`` keeps the stage dirs
            of a spec with ``etl.cleanup_intermediates: true``.
        download_do_overwrite: If ``True``, the download child re-fetches every file,
            even when the local copy verifies against the manifest (forwarded as
            ``do_overwrite=``). Distinct from the pipeline-level ``do_overwrite``,
            which is an ``overrides=`` key.
        download_concurrency: Max parallel transport streams for the download child.
        download_continue_on_error: If ``True``, per-file download failures don't sink
            the run; every source is still attempted and the child exits non-zero at
            the end if anything failed.

    ``__post_init__`` runs when the CLI materializes the Hydra config via
    ``OmegaConf.to_object`` (Hydra itself always hands the task function a
    ``DictConfig``; ``to_object`` is the idiomatic bridge back to the registered
    dataclass). It validates the plausible combinations — exactly one of "download
    into ``download_dest_dir``" (``do_download=true``) or "read pre-staged
    ``input_dir``" (``do_download=false``) describes where the pipeline's raw
    input comes from (:attr:`effective_input_dir`) — rejects non-default
    ``download_*`` knobs on a download-free run (``do_download=false`` spawns no
    download child, so they could only be silently ignored) — and normalizes the
    directory fields to absolute local paths (URL-shaped values are rejected;
    relative paths resolve against the invoking CWD, which
    ``hydra.job.chdir=false`` leaves untouched), so no path handling is left to
    the CLI body.
    """

    spec: str = MISSING
    output_dir: str = MISSING
    do_download: bool = True
    dataset_key: str = "dataset"
    download_dest_dir: str | None = None
    input_dir: str | None = None
    dataset_version: str | None = None
    skip_stages: list[str] = field(default_factory=list)
    stage_runner_fp: str | None = None
    do_profile: bool = False
    overrides: list[str] = field(default_factory=list)
    download_do_overwrite: bool = False
    download_concurrency: int = 4
    download_continue_on_error: bool = False

    def __post_init__(self):
        # The CLI runs with ``hydra.job.chdir=false`` (see ``_cli.yaml``), so relative
        # paths mean what the shell suggests; they are absolutized here only so the
        # children (which may manage their own CWDs) see unambiguous paths. URL-shaped
        # values are rejected outright — these fields are local-only.
        for f in ("output_dir", "download_dest_dir", "input_dir", "stage_runner_fp"):
            if getattr(self, f) not in (None, MISSING):
                setattr(self, f, str(user_local_path(getattr(self, f), field=f)))
        if not self.do_download and self.input_dir is None:
            raise ValueError("do_download=false requires input_dir= pointing at pre-staged raw data.")
        if self.do_download and self.input_dir is not None:
            raise ValueError(
                "input_dir= is for download-free runs (do_download=false). With do_download=true, "
                "data is downloaded into download_dest_dir=, which is also the pipeline's input "
                "— set one or the other."
            )
        if not self.do_download and self.download_dest_dir is not None:
            raise ValueError("download_dest_dir= has no effect with do_download=false; use input_dir=.")
        if not self.do_download:
            # A download-free run spawns no download child, so a download-only knob
            # could only be silently ignored — reject it instead.
            defaults = {f.name: f.default for f in fields(self)}
            knobs = ("download_do_overwrite", "download_concurrency", "download_continue_on_error")
            set_knobs = [k for k in knobs if getattr(self, k) != defaults[k]]
            if set_knobs:
                raise ValueError(
                    f"{', '.join(f'{k}=' for k in set_knobs)} has no effect with do_download=false "
                    "(no download child runs); remove it or set do_download=true."
                )

    @property
    def work_dir(self) -> Path:
        """Run-internal artifact dir: ``<output_dir>/.meds_extract_run``."""
        return Path(self.output_dir) / ".meds_extract_run"

    @property
    def effective_input_dir(self) -> Path:
        """The pipeline's raw-data input: ``input_dir`` or the download destination."""
        if self.input_dir is not None:
            return Path(self.input_dir)
        return Path(self.download_dest_dir) if self.download_dest_dir else self.work_dir / "raw_input"

    def download_argv(self, spec_ref: str) -> list[str]:
        """Build the ``meds-extract-download`` child command line.

        The path-valued overrides (``spec``, ``output_dir``, ``hydra.run.dir``) are
        wrapped via :func:`_hydra_quote` so paths containing override-grammar
        metacharacters survive the child's parse. Quoting stays confined to those
        three: the child is always spawned single-run, so no legitimate value ever
        carries sweep/range syntax that quoting would suppress, while the remaining
        overrides render from typed dataclass fields (``bool``/``int``, plus a
        ``key`` constrained to sources-bucket names) whose bare forms are what the
        grammar types natively.

        Examples:
            >>> run = RunConfig(spec="Example", output_dir="/data/out", dataset_key="demo")
            >>> run.download_argv("pkg://ex.messy.yaml")
            ['MEDS_extract.download.cli', "spec='pkg://ex.messy.yaml'",
             "output_dir='/data/out/.meds_extract_run/raw_input'", 'key=demo', 'do_overwrite=False',
             'concurrency=4', 'continue_on_error=False',
             "hydra.run.dir='/data/out/.meds_extract_run/hydra_download'"]

            Quoting keeps paths with override-grammar metacharacters (a bare ``,``
            starts a choice sweep; a bare ``=`` splits the override) intact:

            >>> run = RunConfig(spec="Example", output_dir="/data/comma,dir/eq=dir/out")
            >>> [a for a in run.download_argv("/data/comma,dir/s.yaml") if a.startswith("spec")]
            ["spec='/data/comma,dir/s.yaml'"]
            >>> [a for a in run.download_argv("s") if a.startswith("output_dir")]
            ["output_dir='/data/comma,dir/eq=dir/out/.meds_extract_run/raw_input'"]

            The download knobs are forwarded verbatim (``download_do_overwrite``
            as the child's ``do_overwrite=``):

            >>> run = RunConfig(
            ...     spec="Example", output_dir="/data/out", download_do_overwrite=True,
            ...     download_concurrency=8, download_continue_on_error=True,
            ... )
            >>> [a for a in run.download_argv("s") if a.startswith(("do_over", "concurrency", "continue"))]
            ['do_overwrite=True', 'concurrency=8', 'continue_on_error=True']
        """
        return [
            "MEDS_extract.download.cli",
            f"spec={_hydra_quote(spec_ref)}",
            f"output_dir={_hydra_quote(str(self.effective_input_dir))}",
            f"key={self.dataset_key}",
            f"do_overwrite={self.download_do_overwrite}",
            f"concurrency={self.download_concurrency}",
            f"continue_on_error={self.download_continue_on_error}",
            # Keep the child's Hydra run dir out of the user's CWD.
            f"hydra.run.dir={_hydra_quote(str(self.work_dir / 'hydra_download'))}",
        ]

    def pipeline_argv(self, pipeline_fp: Path) -> list[str]:
        """Build the pipeline-runner child command line (``python -m MEDS_transforms.runner``).

        The pipeline runner's CLI is argparse, not Hydra, so these are real flags rather
        than dotlist overrides. ``--overrides`` is ``nargs="*"`` and therefore always goes
        last — any flag after it would be swallowed as another override.

        Examples:
            Nothing optional set — just the config path:

            >>> run = RunConfig(spec="Example", output_dir="/data/out")
            >>> run.pipeline_argv(Path("/data/out/.meds_extract_run/pipeline.yaml"))
            ['MEDS_transforms.runner', '/data/out/.meds_extract_run/pipeline.yaml']

            Each knob appends its flag; ``--overrides`` stays last:

            >>> run = RunConfig(
            ...     spec="Example", output_dir="/data/out",
            ...     stage_runner_fp="/cfg/runner.yaml", do_profile=True,
            ...     overrides=["seed=2", "do_overwrite=True"],
            ... )
            >>> run.pipeline_argv(Path("/p.yaml"))
            ['MEDS_transforms.runner', '/p.yaml', '--stage_runner_fp', '/cfg/runner.yaml',
             '--do_profile', '--overrides', 'seed=2', 'do_overwrite=True']
        """
        argv = ["MEDS_transforms.runner", str(pipeline_fp)]
        if self.stage_runner_fp is not None:
            argv += ["--stage_runner_fp", self.stage_runner_fp]
        if self.do_profile:
            argv.append("--do_profile")
        if self.overrides:
            argv += ["--overrides", *self.overrides]
        return argv


@hydra.main(version_base=None, config_path=".", config_name="_cli")
def _hydra_main(cfg: DictConfig) -> None:
    """Hydra task function for ``meds-extract-run``; see :func:`main`.

    Required args (Hydra dotlist syntax): ``spec=...`` and ``output_dir=...``; see
    :class:`RunConfig` for the optional knobs.
    """

    try:
        # Materialize the registered dataclass: __post_init__ validates the flag
        # combinations and absolutizes the directory fields.
        run: RunConfig = OmegaConf.to_object(cfg)
        # One load of the one config object; everything else comes off it.
        messy = MessyConfig.load(run.spec)
        _ = messy.event_tables  # the pipeline needs event tables: fail before any work
        # Version stamping always follows the selected sources bucket (dataset_key),
        # whether or not the download runs — only relevant for mapping-form
        # sources.dataset_version, where buckets carry distinct versions.
        pipeline_cfg = messy.pipeline_config(
            input_dir=run.effective_input_dir,
            output_dir=Path(run.output_dir),
            key=run.dataset_key,
            dataset_version=run.dataset_version,
            skip_stages=run.skip_stages,
        )
    except (ValueError, FileNotFoundError) as e:
        logger.error(str(e))
        sys.exit(1)
    logger.info(f"Resolved spec={run.spec!r} to {messy.spec_ref}")

    if run.do_download:
        rc = run_command(run.download_argv(messy.spec_ref))
        if rc != 0:
            logger.error(f"meds-extract-download failed with exit code {rc}.")
            sys.exit(rc)
    else:
        logger.info("do_download=false: skipping the download stage.")

    pipeline_fp = run.work_dir / "pipeline.yaml"
    pipeline_fp.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(pipeline_cfg), pipeline_fp)
    logger.info(f"Wrote synthesized pipeline config to {pipeline_fp}")

    stages = [stage_name(entry) for entry in pipeline_cfg["stages"]]
    post_stages = {s.name for s in messy.etl.post_stages}
    log_dir = Path(run.output_dir) / ".logs"
    reconcile_done_markers(log_dir, stages, post_stages)

    rc = run_command(run.pipeline_argv(pipeline_fp))
    if rc == 0:
        restore_cleaned_markers(log_dir, stages)
    sys.exit(rc)


def main() -> None:
    """Console-script entry point for ``meds-extract-run``.

    Validates the required dotlist args before Hydra owns the process: the Hydra run
    dir is anchored at ``${output_dir}/...``, so without this check a bare invocation
    would die inside interpolation resolution instead of printing usage — and a failed
    invocation must create no directories anywhere.
    """
    for arg in sys.argv[1:]:
        if arg.lstrip("+").startswith("do_cleanup_intermediates="):
            value = arg.partition("=")[2].strip().lower() or "true"
            sys.exit(
                f"do_cleanup_intermediates was removed; pass \"overrides=['cleanup_intermediates={value}']\" "
                f"(a spec sets its default with etl.cleanup_intermediates)."
            )
    require_dotlist_args(
        "meds-extract-run",
        {"spec": "<name|pkg://...|/path|./path>", "output_dir": "<dir>"},
        local_only=("output_dir", "input_dir", "download_dest_dir"),
    )
    _hydra_main()
