"""Read a MESSY ``sources:`` block and construct :class:`Source` instances.

The MESSY spec gained a top-level ``sources:`` block under which each ETL declares the
backends it pulls raw data from — see
https://github.com/mmcdermott/MEDS_extract/issues/81 for the design. This module owns
everything spec-shaped: the ``type:`` → backend registry (:data:`_SOURCE_TYPES`), the
per-entry constructor (:func:`source_from_config`), and the whole-block reader
(:func:`sources_from_spec`).

Backend modules are imported lazily, per selected ``type:`` — so a spec that only uses
``fsspec`` sources never imports the HTTP stack, and the ``download`` extra
(``httpx``/``tenacity``) is only required when an ``http`` / ``physionet`` source is
actually constructed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .source import Source

# The one place the ``type:`` → backend map lives. Adding a backend = adding a
# ``backends/`` module and one row here; the error message below derives its
# supported-types list from this dict, so the two can't drift.
_SOURCE_TYPES: dict[str, tuple[str, str]] = {
    "fsspec": (".backends.fsspec", "FsspecSource"),
    "http": (".backends.http", "HTTPSource"),
    "physionet": (".backends.physionet", "PhysioNetSource"),
}

# Reserved non-bucket keys inside ``sources:``. ``dataset_version`` declares the raw
# data release the buckets point at (scalar string, or {bucket: version} mapping) —
# it is a property of the source data, baked into download URLs, and is
# interpolatable from bucket entries via ``${sources.dataset_version}`` /
# ``${sources.dataset_version.<bucket>}``. It is consumed by ``meds-extract-run``
# for the ``etl_metadata.dataset_version`` stamp (see
# ``MessyConfig._sources_dataset_version`` in ``MEDS_extract.config``, which owns
# its shape validation, and the public accessors ``sources_version`` /
# ``raw_version_for``) and must never be treated as a bucket here.
SOURCES_RESERVED_KEYS: frozenset[str] = frozenset({"dataset_version"})


def source_from_config(cfg: dict) -> Source:
    """Construct one :class:`Source` from a single ``sources:`` list entry.

    Each entry is a dict carrying at minimum a ``type:`` field. Remaining keys are
    forwarded to the backend's constructor; the ``type:`` key is validated against
    :data:`_SOURCE_TYPES` and stripped before forwarding.

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

        Missing ``type:`` is flagged the same way. Only the key names are echoed —
        by the time an entry reaches this function its ``${oc.env:...}`` interpolations
        are resolved, so echoing values could put live credentials in logs:

        >>> source_from_config({"urls": ["https://example.com/x.csv"], "password": "hunter2"})
        Traceback (most recent call last):
            ...
        ValueError: Source config is missing a 'type:' key. Got keys: ['password', 'urls']

        Backend-specific kwargs pass through verbatim — e.g. ``HTTPSource``'s ``headers:``
        for API-key auth (DataVerse ``X-Dataverse-key``, bearer tokens, ``Accept:``):

        >>> src = source_from_config({
        ...     "type": "http",
        ...     "urls": ["https://example.com/x.csv"],
        ...     "headers": {"X-Dataverse-key": "secret-token"},
        ... })
        >>> src._client.headers["X-Dataverse-key"]
        'secret-token'
        >>> src.close()

        That includes the generic ``include:`` / ``exclude:`` manifest filters every
        backend accepts:

        >>> src = source_from_config({"type": "fsspec", "root": "/tmp", "include": ["hosp/*"]})
        >>> src._include
        ['hosp/*']
    """
    cfg = dict(cfg)
    source_type = cfg.pop("type", None)
    if source_type is None:
        # Echo key names only: values may hold resolved credentials, and this message
        # lands on stderr and in the persisted Hydra log.
        raise ValueError(f"Source config is missing a 'type:' key. Got keys: {sorted(cfg)}")
    if source_type not in _SOURCE_TYPES:
        raise ValueError(f"Unknown source type {source_type!r}. Supported: {sorted(_SOURCE_TYPES)}.")

    module_name, class_name = _SOURCE_TYPES[source_type]
    module = importlib.import_module(module_name, package=__package__)
    return getattr(module, class_name)(**cfg)


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
        doesn't declare ``demo`` is legal; the CLI layers its own stricter
        key-must-exist validation on top):

        >>> sources_from_spec({"sources": {"dataset": []}}, key="demo")
        []

        Missing top-level ``sources:`` is also legal (returns empty):

        >>> sources_from_spec({}, key="dataset")
        []

        ``key="common"`` doesn't double-count — the common bucket is already
        the selected one, so it isn't appended a second time:

        >>> [type(s).__name__ for s in sources_from_spec(spec, key="common")]
        ['HTTPSource']

        The reserved ``dataset_version`` key is metadata, not a bucket — selecting
        it is an error (its string/mapping value would otherwise be iterated as if
        it were a source list):

        >>> sources_from_spec({"sources": {"dataset_version": "3.1"}}, key="dataset_version")
        Traceback (most recent call last):
            ...
        ValueError: key='dataset_version' is a reserved sources: key (raw-data version metadata),
        not a bucket.
    """
    if key in SOURCES_RESERVED_KEYS:
        raise ValueError(f"key={key!r} is a reserved sources: key (raw-data version metadata), not a bucket.")
    sources_block = spec.get("sources", {}) or {}
    configured = list(sources_block.get(key, []) or [])
    # ``common`` is always appended UNLESS it's already the selected bucket —
    # otherwise ``sources_from_spec(spec, key="common")`` would build every
    # common backend twice and race two writers on the same dest
    # mid-orchestration.
    common = list(sources_block.get("common", []) or []) if key != "common" else []
    return [source_from_config(c) for c in configured + common]
