"""Shared download layer for MEDS_extract-based ETLs.

The public surface is:

- :class:`Fetcher` — bounded-concurrency orchestrator
- :class:`Source` protocol and :class:`RemoteFile` / :class:`FetchResult` / :class:`FetchReport`
- Concrete backends: :class:`HTTPSource`, :class:`FsspecSource`, :class:`PhysioNetSource`
- :func:`sources_from_spec` / :func:`source_from_config` — build :class:`Source` instances
  from a MESSY ``sources:`` block

See https://github.com/mmcdermott/MEDS_extract/issues/81 for the design rationale.

The heavy HTTP deps (:mod:`httpx`, :mod:`tenacity`) are declared under the ``download``
extra in ``pyproject.toml``. Install with ``pip install 'MEDS_extract[download]'``.
"""

from .backends import FsspecSource, HTTPSource, PhysioNetSource
from .dispatch import source_from_config, sources_from_spec
from .fetcher import Fetcher
from .source import FetchReport, FetchResult, RemoteFile, Source

__all__ = [
    "FetchReport",
    "FetchResult",
    "Fetcher",
    "FsspecSource",
    "HTTPSource",
    "PhysioNetSource",
    "RemoteFile",
    "Source",
    "source_from_config",
    "sources_from_spec",
]
