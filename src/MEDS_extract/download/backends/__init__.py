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
"""

from .fsspec import FsspecSource
from .http import HTTPSource
from .physionet import PhysioNetSource

__all__ = ["FsspecSource", "HTTPSource", "PhysioNetSource"]
