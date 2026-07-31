"""Spec resolution for ``meds-extract-run``: registered name → ``pkg://`` → path.

Dataset packages register under the ``MEDS_extract.pipelines`` entry-point group
(precedent one level down: ``MEDS_transforms.stages``), pointing **directly at the
bundled MESSY file** in ``<package.module>:<filename.yaml>`` form::

    [project.entry-points."MEDS_extract.pipelines"]
    MIMIC-IV = "MIMIC_IV_MEDS.configs:event_configs.yaml"

The entry point's value string is parsed here and the file is resolved as
``importlib.resources.files(module) / filename`` — the entry point is never
``load()``-ed, so registration cannot execute dataset-package code. Because an
entry point knows its providing distribution, resolution by registered name also
yields the distribution's version, which the runner stamps into
``etl_metadata.dataset_version`` for zero-code ETL provenance.
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


@dataclass(frozen=True)
class ResolvedSpec:
    """A resolved ``spec=`` argument: the MESSY file plus its provenance.

    ``dist_version`` is populated only for registry resolutions — it comes from the
    entry point's providing distribution and drives the automatic
    ``dataset_version`` stamp suffix. For ``pkg`` / ``path`` resolutions it is
    ``None`` and the runner stamps the raw data version alone (or an explicit
    ``dataset_version=`` override).
    """

    spec_fp: Path
    origin: str  # "registry" | "pkg" | "path"
    dist_version: str | None = None


def _spec_fp_from_entry_point(name: str, value: str) -> Path:
    """Resolve a ``MEDS_extract.pipelines`` entry-point value to its MESSY file.

    The registration points directly at the file: ``<package.module>:<filename.yaml>``.
    A bare module reference (no ``:filename``) is rejected — the registration must
    name the file, so there is no bundled-layout convention to learn or get wrong.
    """
    module, _, filename = value.partition(":")
    if not module or not filename:
        raise ValueError(
            f"Entry point {name!r} in group {PIPELINES_ENTRY_POINT_GROUP!r} has value {value!r}, "
            f"which does not name the bundled MESSY file. Register it as "
            f"'<package.module>:<filename.yaml>', e.g. 'MIMIC_IV_MEDS.configs:event_configs.yaml'."
        )
    fp = Path(str(files(module) / filename))
    if not fp.is_file():
        raise ValueError(
            f"Entry point {name!r} points at {value!r}, but module {module!r} has no resource "
            f"named {filename!r} (resolved to {fp})."
        )
    return fp


def resolve_spec(spec: str, *, path_resolver: Callable[[str], Path] = Path) -> ResolvedSpec:
    """Resolve a ``spec=`` argument down the registered-name → ``pkg://`` → path ladder.

    The same ladder ``MEDS_transform-pipeline`` has for pipeline configs, extended
    one rung up with the ``MEDS_extract.pipelines`` registry. Rungs, in order:

    1. **Registered name** — exact match against the ``MEDS_extract.pipelines``
       entry-point group; the entry-point value names the MESSY file directly
       (:func:`_spec_fp_from_entry_point`), and the providing distribution's
       version rides along for provenance stamping.
    2. **pkg://** — MEDS-transforms' ``resolve_pkg_path`` syntax
       (``pkg://<pkg_name>.<dotted.relative.path>.<ext>``), reused verbatim so the
       CLIs share one syntax. Unambiguous by its literal prefix.
    3. **Filesystem path** — anything else, passed through ``path_resolver`` first
       (the CLI injects Hydra's original-CWD resolution here; the default is a plain
       ``Path``).

    Registered names win over paths by design: they are dataset display names
    ("MIMIC-IV") that in practice never collide with an existing file path — a
    collision would require a file literally named after a registered dataset
    sitting in the working directory, and resolving to the registration is the
    least surprising reading there. ``pkg://`` cannot collide with either
    neighbor (the prefix is literal, and entry-point names containing ``://``
    are not a thing).

    Args:
        spec: The raw ``spec=`` string.
        path_resolver: Maps a non-registry, non-``pkg://`` spec string to a
            candidate :class:`~pathlib.Path` before the existence check.

    Raises:
        FileNotFoundError: If no rung matches — the message lists the registered
            pipeline names so a typo'd name is diagnosable.
        ValueError: From rung internals (a malformed registration value, a
            registration naming a missing resource, or a ``pkg://`` package that
            isn't installed).

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
            spec_fp=_spec_fp_from_entry_point(ep.name, ep.value),
            origin="registry",
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
