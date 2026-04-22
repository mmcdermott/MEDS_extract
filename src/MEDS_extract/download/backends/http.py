"""Explicit-URL-list HTTP backend."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from .._http import _make_httpx_client, _resumable_download
from ..source import RemoteFile

if TYPE_CHECKING:
    from collections.abc import Iterable

    import httpx


class HTTPSource:
    """A :class:`Source` backed by an explicit list of HTTP URLs.

    Use this for shared metadata downloads where the file list is known up-front — e.g.
    MIMIC's ``common:`` block of concept-map CSVs from ``raw.githubusercontent.com``. No
    crawling, no manifest parsing.

    Each URL entry can be either a plain string or a dict with optional ``sha256``,
    ``rel_path``, and ``size`` fields. ``rel_path`` defaults to the URL's basename.

    Examples:
        Plain string URLs resolve to basename-based relative paths:

        >>> src = HTTPSource(urls=["https://example.com/foo.csv", "https://example.com/bar.csv"])
        >>> [r.rel_path for r in src.list_files()]
        ['foo.csv', 'bar.csv']

        Dict entries can override ``rel_path`` and provide a checksum:

        >>> src = HTTPSource(
        ...     urls=[
        ...         {"url": "https://example.com/foo.csv", "rel_path": "lookups/foo.csv"},
        ...         {"url": "https://example.com/bar.csv", "sha256": "abc123"},
        ...     ]
        ... )
        >>> fs = list(src.list_files())
        >>> fs[0].rel_path, fs[0].sha256
        ('lookups/foo.csv', None)
        >>> fs[1].rel_path, fs[1].sha256
        ('bar.csv', 'abc123')

        URLs without a path component fall back to ``"index.html"``:

        >>> src = HTTPSource(urls=["https://example.com/"])
        >>> [r.rel_path for r in src.list_files()]
        ['index.html']
    """

    def __init__(self, urls: list[str | dict], client: httpx.Client | None = None):
        self._entries = [_normalize(u) for u in urls]
        self._client = client or _make_httpx_client()

    def list_files(self) -> Iterable[RemoteFile]:
        for e in self._entries:
            yield RemoteFile(
                rel_path=e["rel_path"],
                size=e.get("size"),
                sha256=e.get("sha256"),
                extra={"url": e["url"]},
            )

    def fetch(self, remote: RemoteFile, dest: Path) -> None:
        _resumable_download(self._client, remote.extra["url"], dest, remote.sha256)


def _normalize(entry: str | dict) -> dict:
    """Normalize a URL entry to ``{"url", "rel_path", "sha256"?, "size"?}``.

    Examples:
        >>> _normalize("https://example.com/foo.csv")
        {'url': 'https://example.com/foo.csv', 'rel_path': 'foo.csv'}

        >>> _normalize({"url": "https://example.com/foo.csv", "sha256": "abc"})
        {'url': 'https://example.com/foo.csv', 'sha256': 'abc', 'rel_path': 'foo.csv'}

        Explicit ``rel_path`` wins over the URL-derived default:

        >>> _normalize({"url": "https://example.com/foo.csv", "rel_path": "lookups/foo.csv"})
        {'url': 'https://example.com/foo.csv', 'rel_path': 'lookups/foo.csv'}

        Raises on missing ``url`` or bad type:

        >>> _normalize({"sha256": "abc"})
        Traceback (most recent call last):
            ...
        ValueError: HTTPSource url entry is missing 'url': {'sha256': 'abc'}
        >>> _normalize(42)
        Traceback (most recent call last):
            ...
        TypeError: HTTPSource url entry must be a str or dict, got int: 42
    """
    if isinstance(entry, str):
        return {"url": entry, "rel_path": _filename_from_url(entry)}
    if isinstance(entry, dict):
        if "url" not in entry:
            raise ValueError(f"HTTPSource url entry is missing 'url': {entry}")
        out = dict(entry)
        out.setdefault("rel_path", _filename_from_url(entry["url"]))
        return out
    raise TypeError(f"HTTPSource url entry must be a str or dict, got {type(entry).__name__}: {entry}")


def _filename_from_url(url: str) -> str:
    """Derive a filesystem-friendly rel_path from ``url``.

    Examples:
        >>> _filename_from_url("https://example.com/foo.csv")
        'foo.csv'
        >>> _filename_from_url("https://example.com/path/to/bar.csv.gz")
        'bar.csv.gz'
        >>> _filename_from_url("https://example.com/")
        'index.html'
        >>> _filename_from_url("https://example.com")
        'index.html'
    """
    return Path(urlparse(url).path).name or "index.html"
