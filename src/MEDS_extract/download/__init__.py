"""Shared download layer for MEDS_extract-based ETLs.

The public surface is intentionally small:

- :class:`Fetcher` — bounded-concurrency policy holder (max workers, retry, overwrite,
  unarchive). One instance is reused across :class:`Source` constructions.
- :class:`Source` ABC and the concrete backends (:class:`HTTPSource`,
  :class:`FsspecSource`, :class:`PhysioNetSource`). Each Source takes a
  :class:`Fetcher` at construction and exposes
  :meth:`Source.download_all <Source.download_all>` as the only fetch entry point.
- :class:`FetchReport` / :class:`FetchResult` — the bundle-level outcome from
  :meth:`Source.download_all`.
- :func:`sources_from_spec` / :func:`source_from_config` — build :class:`Source`
  instances (with a shared :class:`Fetcher`) from a MESSY ``sources:`` block.

There is intentionally no per-file fetch API; every consumer we have downloads the
whole bundle, and per-file granularity invites the "wrong source x wrong manifest"
mismatched-pair footgun. See https://github.com/mmcdermott/MEDS_extract/pull/96 for
the design discussion that landed on this shape.

The heavy HTTP deps (:mod:`httpx`, :mod:`tenacity`) are declared under the ``download``
extra in ``pyproject.toml``. Install with ``pip install 'MEDS_extract[download]'``.
"""

from .backends import FsspecSource, HTTPSource, PhysioNetSource
from .dispatch import source_from_config, sources_from_spec
from .fetcher import Fetcher, FetchReport, FetchResult
from .source import Source

__all__ = [
    "FetchReport",
    "FetchResult",
    "Fetcher",
    "FsspecSource",
    "HTTPSource",
    "PhysioNetSource",
    "Source",
    "source_from_config",
    "sources_from_spec",
]
