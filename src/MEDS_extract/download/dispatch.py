"""Read a MESSY ``sources:`` block and construct :class:`Source` instances.

The MESSY spec gained a top-level ``sources:`` block under which each ETL declares the
backends it pulls raw data from — see
https://github.com/mmcdermott/MEDS_extract/issues/81 for the design. This module is the
dispatcher from raw YAML entries to concrete :class:`Source` subclasses.

Currently supported ``type:`` values: ``physionet``, ``http``, ``fsspec``. New backends
added under :mod:`~MEDS_extract.download.backends` register in the match statement of
:func:`source_from_config`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .backends import FsspecSource, HTTPSource, PhysioNetSource

if TYPE_CHECKING:
    from .source import Source


def source_from_config(cfg: dict) -> Source:
    """Construct one :class:`Source` from a single ``sources:`` list entry.

    Each entry is a dict carrying at minimum a ``type:`` field. Remaining keys are
    forwarded to the backend's constructor; the dispatcher validates the type is known
    and strips the ``type:`` key before forwarding.

    Examples:
        >>> source_from_config({"type": "http", "urls": ["https://example.com/x.csv"]})
        ... # doctest: +ELLIPSIS
        <MEDS_extract.download.backends.http.HTTPSource object at 0x...>
        >>> source_from_config({"type": "fsspec", "root": "/tmp"})
        ... # doctest: +ELLIPSIS
        <MEDS_extract.download.backends.fsspec.FsspecSource object at 0x...>
        >>> source_from_config(
        ...     {"type": "physionet", "base_url": "https://physionet.org/files/mimic-iv-demo/2.2"}
        ... )  # doctest: +ELLIPSIS
        <MEDS_extract.download.backends.physionet.PhysioNetSource object at 0x...>

        Unknown types surface a clear error:

        >>> source_from_config({"type": "s3"})
        Traceback (most recent call last):
            ...
        ValueError: Unknown source type 's3'. Supported: ['fsspec', 'http', 'physionet'].

        Missing ``type:`` is flagged the same way:

        >>> source_from_config({"urls": ["https://example.com/x.csv"]})
        Traceback (most recent call last):
            ...
        ValueError: Source config is missing a 'type:' key. Got: {'urls': ['https://example.com/x.csv']}
    """
    cfg = dict(cfg)
    source_type = cfg.pop("type", None)
    if source_type is None:
        raise ValueError(f"Source config is missing a 'type:' key. Got: {cfg}")

    match source_type:
        case "physionet":
            return PhysioNetSource(**cfg)
        case "http":
            return HTTPSource(**cfg)
        case "fsspec":
            return FsspecSource(**cfg)
        case _:
            raise ValueError(
                f"Unknown source type {source_type!r}. Supported: ['fsspec', 'http', 'physionet']."
            )


def sources_from_spec(spec: dict, key: str = "dataset") -> list[Source]:
    """Read a full MESSY ``sources:`` block and return the configured + common sources.

    The ``key`` argument selects which bucket of sources to pull — ``"dataset"`` (the
    default), ``"demo"`` (for demo downloads), or any other top-level key under
    ``sources:``. The ``"common"`` bucket is always appended and carries shared
    metadata files (e.g. MIMIC's ``concept_map`` CSVs from GitHub) that all ETL runs
    need regardless of which primary source is selected.

    Examples:
        >>> spec = {
        ...     "sources": {
        ...         "dataset": [
        ...             {"type": "http", "urls": ["https://example.com/data.csv"]},
        ...         ],
        ...         "demo": [
        ...             {"type": "http", "urls": ["https://example.com/demo.csv"]},
        ...         ],
        ...         "common": [
        ...             {"type": "http", "urls": ["https://example.com/shared.csv"]},
        ...         ],
        ...     },
        ... }
        >>> [type(s).__name__ for s in sources_from_spec(spec, key="dataset")]
        ['HTTPSource', 'HTTPSource']
        >>> [type(s).__name__ for s in sources_from_spec(spec, key="demo")]
        ['HTTPSource', 'HTTPSource']

        Missing keys quietly resolve to an empty list (not an error — a MESSY file that
        doesn't declare ``demo`` is legal):

        >>> sources_from_spec({"sources": {"dataset": []}}, key="demo")
        []

        Missing top-level ``sources:`` is also legal (returns empty):

        >>> sources_from_spec({}, key="dataset")
        []
    """
    sources_block = spec.get("sources", {}) or {}
    configured = list(sources_block.get(key, []) or [])
    common = list(sources_block.get("common", []) or [])
    return [source_from_config(c) for c in configured + common]
