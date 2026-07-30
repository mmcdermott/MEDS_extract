"""Spec resolution for ``meds-extract-run``: registered name → ``pkg://`` → path.

Dataset packages register under the ``MEDS_extract.pipelines`` entry-point group
(precedent one level down: ``MEDS_transforms.stages``), mapping a public dataset name
to the module whose ``importlib.resources`` contain the package's MESSY file::

    [project.entry-points."MEDS_extract.pipelines"]
    MIMIC-IV = "MIMIC_IV_MEDS.configs"

The entry point is never ``load()``-ed — only its module name is used for resource
lookup — so registration cannot execute dataset-package code. Because an entry point
knows its providing distribution, resolution by registered name also yields the
distribution's version, which the runner stamps into
``etl_metadata.dataset_version`` as ``{raw_dataset_version}:{distribution_version}``
for zero-code ETL provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

from MEDS_transforms.utils import PKG_PFX, resolve_pkg_path

if TYPE_CHECKING:
    from collections.abc import Callable

PIPELINES_ENTRY_POINT_GROUP = "MEDS_extract.pipelines"

# The canonical bundled-MESSY filename, used to disambiguate when the registered
# module carries more than one YAML resource. Matches what existing dataset packages
# (ETL_MEDS_Template, MIMIC_IV_MEDS) already call the file.
CANONICAL_MESSY_FILENAME = "event_configs.yaml"

_YAML_SUFFIXES = (".yaml", ".yml")


@dataclass(frozen=True)
class ResolvedSpec:
    """A resolved ``spec=`` argument: the MESSY file plus its provenance.

    ``dist_name`` / ``dist_version`` are populated only for registry resolutions —
    they come from the entry point's providing distribution and drive the automatic
    ``dataset_version`` stamp. For ``pkg`` / ``path`` resolutions they are ``None``
    and the runner falls back to the ``etl:`` block's ``raw_dataset_version`` alone
    (or an explicit ``dataset_version=`` override).
    """

    spec_fp: Path
    origin: str  # "registry" | "pkg" | "path"
    dist_name: str | None = None
    dist_version: str | None = None


def messy_file_for_module(module: str) -> Path:
    """Locate the bundled MESSY file among ``module``'s package resources.

    The convention: if the module carries exactly one ``*.yaml``/``*.yml`` resource
    (the pure-config end state — one YAML describes the whole ETL), that file is it;
    with several YAMLs, the one named :data:`CANONICAL_MESSY_FILENAME`
    (``event_configs.yaml``, the name existing dataset packages already use) wins.
    Anything else is a config error naming the candidates.

    Examples:
        ``MEDS_extract.configs`` happens to carry exactly one YAML resource, so it
        demonstrates the single-YAML rule with no test scaffolding:

        >>> messy_file_for_module("MEDS_extract.configs").name
        '_extract.yaml'

        A module with no YAML at all is rejected:

        >>> messy_file_for_module("MEDS_extract.run")
        Traceback (most recent call last):
            ...
        ValueError: Module 'MEDS_extract.run' contains no *.yaml/*.yml resource to use as the
        MESSY spec.

        (The several-YAMLs disambiguation branch is exercised in
        ``tests/test_run.py`` with synthetic packages.)
    """
    root = files(module)
    candidates = sorted(
        entry.name for entry in root.iterdir() if entry.is_file() and entry.name.endswith(_YAML_SUFFIXES)
    )
    if len(candidates) == 1:
        return Path(str(root / candidates[0]))
    if not candidates:
        raise ValueError(f"Module {module!r} contains no *.yaml/*.yml resource to use as the MESSY spec.")
    if CANONICAL_MESSY_FILENAME in candidates:
        return Path(str(root / CANONICAL_MESSY_FILENAME))
    raise ValueError(
        f"Module {module!r} contains several YAML resources {candidates} and none is named "
        f"{CANONICAL_MESSY_FILENAME!r}. Bundle a single YAML, or name the MESSY file "
        f"{CANONICAL_MESSY_FILENAME!r}."
    )


def resolve_spec(spec: str, *, path_resolver: Callable[[str], Path] = Path) -> ResolvedSpec:
    """Resolve a ``spec=`` argument down the registered-name → ``pkg://`` → path ladder.

    The same ladder ``MEDS_transform-pipeline`` has for pipeline configs, extended
    one rung up with the ``MEDS_extract.pipelines`` registry. Rungs:

    1. **Registered name** — exact match against the ``MEDS_extract.pipelines``
       entry-point group; the MESSY file is located inside the registered module via
       :func:`messy_file_for_module`, and the providing distribution's name/version
       ride along for provenance stamping.
    2. **pkg://** — MEDS-transforms' ``resolve_pkg_path`` syntax
       (``pkg://<pkg_name>.<dotted.relative.path>.<ext>``), reused verbatim so the
       CLIs share one syntax.
    3. **Filesystem path** — anything else, passed through ``path_resolver`` first
       (the CLI injects Hydra's original-CWD resolution here; the default is a plain
       ``Path``).

    Args:
        spec: The raw ``spec=`` string.
        path_resolver: Maps a non-registry, non-``pkg://`` spec string to a
            candidate :class:`~pathlib.Path` before the existence check.

    Raises:
        FileNotFoundError: If no rung matches — the message lists the registered
            pipeline names so a typo'd name is diagnosable.
        ValueError: From rung internals (e.g. an ambiguous bundled-YAML layout, or a
            ``pkg://`` package that isn't installed).

    Examples:
        Path mode (no registration involved):

        >>> with yaml_disk("spec.yaml: 'patients: {dob: {code: BIRTH, time: null}}'") as d:
        ...     rs = resolve_spec(str(Path(d) / "spec.yaml"))
        ...     (rs.origin, rs.spec_fp.name, rs.dist_version)
        ('path', 'spec.yaml', None)

        pkg:// mode (resolving a YAML bundled in an installed package):

        >>> rs = resolve_spec("pkg://MEDS_extract.configs._extract.yaml")
        >>> (rs.origin, rs.spec_fp.name, rs.dist_version)
        ('pkg', '_extract.yaml', None)

        A spec matching nothing fails naming every rung (and the registered names,
        here none):

        >>> resolve_spec("Not-A-Registered-Name")
        Traceback (most recent call last):
            ...
        FileNotFoundError: spec='Not-A-Registered-Name' is not a registered pipeline name, a
        pkg:// reference, or an existing file. Registered pipelines: (none).

        Registry mode is exercised in ``tests/test_run.py`` via synthetic entry
        points — this environment has no real ``MEDS_extract.pipelines``
        registrations to demonstrate with.
    """
    by_name = {ep.name: ep for ep in entry_points(group=PIPELINES_ENTRY_POINT_GROUP)}
    if spec in by_name:
        ep = by_name[spec]
        dist = ep.dist
        return ResolvedSpec(
            spec_fp=messy_file_for_module(ep.module),
            origin="registry",
            dist_name=dist.name if dist is not None else None,
            dist_version=dist.version if dist is not None else None,
        )

    if spec.startswith(PKG_PFX):
        fp = Path(str(resolve_pkg_path(spec)))
        if not fp.is_file():
            raise FileNotFoundError(f"spec={spec!r} resolved to {fp}, which does not exist.")
        return ResolvedSpec(spec_fp=fp, origin="pkg")

    fp = path_resolver(spec)
    if not fp.is_file():
        registered = ", ".join(sorted(by_name)) or "(none)"
        raise FileNotFoundError(
            f"spec={spec!r} is not a registered pipeline name, a pkg:// reference, or an "
            f"existing file. Registered pipelines: {registered}."
        )
    return ResolvedSpec(spec_fp=fp.resolve(), origin="path")
