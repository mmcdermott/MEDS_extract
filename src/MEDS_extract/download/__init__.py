"""Shared download layer for MEDS_extract-based ETLs.

The public surface is intentionally small:

- :class:`Source` ABC and the concrete backends (:class:`HTTPSource`,
  :class:`FsspecSource`, :class:`PhysioNetSource`). Each backend implements two
  private hooks (``_list_files``, ``_fetch``) and inherits :meth:`Source.download_all`
  from the base class.
- :class:`DownloadPolicy` — frozen config bag for ``continue_on_error`` /
  ``do_overwrite``. Concurrency is the pool's, not policy's.
- :class:`FetchReport` / :class:`FetchResult` — the bundle-level outcome from
  :meth:`Source.download_all`.
- :func:`sources_from_spec` / :func:`source_from_config` — build :class:`Source`
  instances from a MESSY ``sources:`` block.

Typical CLI usage builds one :class:`~concurrent.futures.ThreadPoolExecutor` + one
:class:`DownloadPolicy`, then calls ``src.download_all(dest, pool=pool, policy=policy)``
per source. The simple-case caller writes ``src.download_all(dest)`` and gets a
private 4-worker pool with sensible defaults.

The heavy HTTP deps (:mod:`httpx`, :mod:`tenacity`) are declared under the ``download``
extra in ``pyproject.toml``. Install with ``pip install 'MEDS_extract[download]'``.
"""

from .backends import FsspecSource, HTTPSource, PhysioNetSource
from .dispatch import source_from_config, sources_from_spec
from .source import DownloadPolicy, FetchReport, FetchResult, Source

__all__ = [
    "DownloadPolicy",
    "FetchReport",
    "FetchResult",
    "FsspecSource",
    "HTTPSource",
    "PhysioNetSource",
    "Source",
    "source_from_config",
    "sources_from_spec",
]
