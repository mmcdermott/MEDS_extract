"""Shared download layer for MEDS_extract-based ETLs.

The public surface is:

- :class:`Source` ABC — every backend implements this. Public entry point is
  :meth:`Source.download_all`.
- :class:`RemoteFile` — the validated manifest row every ``_list_files``
  implementation (in-repo backend or downstream :class:`Source` subclass) constructs.
- :class:`ChecksumError` — raised by ``download_all`` (directly or wrapped in an
  ``ExceptionGroup``) when a fetched file's SHA-256 doesn't match the manifest.
- :func:`validate_unique_destinations` — cross-source collision check for callers
  staging several sources into one shared directory (the CLI runs it automatically).
- Concrete backends: :class:`HTTPSource`, :class:`FsspecSource`, :class:`PhysioNetSource`
- :func:`sources_from_spec` / :func:`source_from_config` — build :class:`Source`
  instances from a MESSY ``sources:`` block

See https://github.com/mmcdermott/MEDS_extract/issues/81 for the design rationale.

The heavy HTTP deps (:mod:`httpx`, :mod:`tenacity`) are declared under the ``download``
extra in ``pyproject.toml`` and imported lazily — only accessing :class:`HTTPSource` /
:class:`PhysioNetSource` (directly or via a ``type: http`` / ``type: physionet`` spec
entry) requires them. ``FsspecSource``, the ``Source`` ABC, and the spec helpers work
from the base install. Install the extra with ``pip install 'MEDS_extract[download]'``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .source import ChecksumError, RemoteFile, Source, validate_unique_destinations
from .spec import source_from_config, sources_from_spec

if TYPE_CHECKING:
    from .backends import FsspecSource, HTTPSource, PhysioNetSource

_LAZY_BACKENDS = frozenset({"FsspecSource", "HTTPSource", "PhysioNetSource"})

__all__ = [
    "ChecksumError",
    "FsspecSource",
    "HTTPSource",
    "PhysioNetSource",
    "RemoteFile",
    "Source",
    "source_from_config",
    "sources_from_spec",
    "validate_unique_destinations",
]


def __getattr__(name: str):
    if name in _LAZY_BACKENDS:
        backends = importlib.import_module(".backends", __package__)
        return getattr(backends, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
