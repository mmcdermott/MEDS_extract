"""Shared download layer for MEDS_extract-based ETLs.

The public surface is:

- :class:`Source` ABC — every backend implements this. Public entry point is
  :meth:`Source.download_all`.
- Concrete backends: :class:`HTTPSource`, :class:`FsspecSource`, :class:`PhysioNetSource`
- :func:`sources_from_spec` / :func:`source_from_config` — build :class:`Source`
  instances from a MESSY ``sources:`` block

See https://github.com/mmcdermott/MEDS_extract/issues/81 for the design rationale.

The heavy HTTP deps (:mod:`httpx`, :mod:`tenacity`) are declared under the ``download``
extra in ``pyproject.toml``. Install with ``pip install 'MEDS_extract[download]'``.
"""

from .backends import FsspecSource, HTTPSource, PhysioNetSource
from .dispatch import source_from_config, sources_from_spec
from .source import Source

__all__ = [
    "FsspecSource",
    "HTTPSource",
    "PhysioNetSource",
    "Source",
    "source_from_config",
    "sources_from_spec",
]
