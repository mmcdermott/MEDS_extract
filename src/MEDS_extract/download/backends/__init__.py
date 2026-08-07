"""Concrete :class:`~MEDS_extract.download.source.Source` backends.

Each submodule implements one transport type:

- :mod:`.http` — explicit URL list (no crawling), optional per-URL SHA-256.
- :mod:`.fsspec` — any protocol :mod:`fsspec` supports via :mod:`universal_pathlib`
  (local filesystem, ``file://``, ``s3://``, ``gs://``, …). Used for re-runs against a
  pre-downloaded local / cloud mirror.
- :mod:`.physionet` — ``SHA256SUMS.txt``-manifest-driven. PhysioNet specifically; the
  manifest file is published by every dataset release and is the authoritative file
  list + hash index.

All backends return :class:`~MEDS_extract.download.source.RemoteFile` instances and
follow the :class:`~MEDS_extract.download.source.Source` protocol invariants.

Backend classes are resolved lazily (PEP 562) so that importing this package — or the
always-usable :class:`.fsspec.FsspecSource` — does not require the HTTP stack. The
``download`` extra (``httpx``/``tenacity``) is only needed the moment
:class:`.http.HTTPSource` or :class:`.physionet.PhysioNetSource` is actually accessed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .fsspec import FsspecSource
    from .http import HTTPSource
    from .physionet import PhysioNetSource

_EXPORTS: dict[str, str] = {
    "FsspecSource": ".fsspec",
    "HTTPSource": ".http",
    "PhysioNetSource": ".physionet",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        module = importlib.import_module(_EXPORTS[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
