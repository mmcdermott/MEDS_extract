"""HTTP-backed :class:`Source`.

This module absorbs the helpers that used to live in a separate ``_http`` submodule.
Everything HTTP-specific — client construction with tenacity retry, ``.part``-file
Range-resume download, ``Content-Range`` validation, URL-entry normalization — is now
attached to :class:`HTTPSource` as static methods (or plain module-level helpers where
the logic is generic). :class:`~MEDS_extract.download.backends.physionet.PhysioNetSource`
inherits from :class:`HTTPSource` and only overrides :meth:`list_files`.

``HTTPSource.get`` (the tenacity-wrapped method installed on the client in
:meth:`HTTPSource._make_client`) retries on connection errors, read timeouts, and 5xx
responses with exponential backoff. 4xx errors surface immediately — retrying a bad URL
or bad auth makes things worse, not better. Streaming downloads
(:meth:`HTTPSource._resumable_download`) are not auto-retried inside the call; mid-stream
errors surface to the caller, and the staged ``.part`` file lets the caller re-invoke the
download and pick up via ``Range: bytes=N-``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from ..source import ChecksumError, RemoteFile, sha256_of

try:
    import httpx
except ImportError as e:  # pragma: no cover — covered by the core CI job
    raise ImportError(
        "MEDS_extract.download requires the 'download' extra. "
        "Install with: pip install 'MEDS_extract[download]'"
    ) from e

try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
except ImportError as e:  # pragma: no cover — covered by the core CI job
    raise ImportError(
        "MEDS_extract.download requires the 'download' extra. "
        "Install with: pip install 'MEDS_extract[download]'"
    ) from e

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


class HTTPSource:
    """A :class:`Source` backed by an explicit list of HTTP URLs.

    Use this for shared metadata downloads where the file list is known up-front — e.g.
    MIMIC's ``common:`` block of concept-map CSVs from ``raw.githubusercontent.com``. No
    crawling, no manifest parsing.

    Each URL entry can be either a plain string or a dict with optional ``sha256``,
    ``rel_path``, and ``size`` fields. ``rel_path`` defaults to the URL's basename.

    Args:
        urls: List of URL entries — plain strings or dicts. Subclasses that discover URLs
            at :meth:`list_files` time (e.g. :class:`PhysioNetSource`) may pass ``None``.
        client: Optional pre-built :class:`httpx.Client`. When omitted, one is built via
            :meth:`_make_client` with the remaining kwargs.
        auth, timeout, max_retries, transport: Forwarded to :meth:`_make_client` when
            ``client`` is not provided.

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

    # Retried via tenacity. 4xx errors are NOT retried (including 429) — those usually
    # mean the URL is wrong or auth failed, and retrying makes things worse. If
    # per-endpoint 429-with-Retry-After handling is needed later, wire it in as a
    # separate retry predicate rather than expanding this list.
    _RETRY_EXC: tuple[type[BaseException], ...] = (
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
        httpx.ConnectError,
        httpx.RemoteProtocolError,
    )

    def __init__(
        self,
        urls: list[str | dict] | None = None,
        client: httpx.Client | None = None,
        auth: tuple[str, str] | None = None,
        timeout: tuple[float, float] = (10.0, 60.0),
        max_retries: int = 5,
        transport: httpx.BaseTransport | None = None,
    ):
        self._entries = [self._normalize(u) for u in (urls or [])]
        self._client = client or self._make_client(
            auth=auth, timeout=timeout, max_retries=max_retries, transport=transport
        )

    def list_files(self) -> Iterable[RemoteFile]:
        for e in self._entries:
            yield RemoteFile(
                rel_path=e["rel_path"],
                size=e.get("size"),
                sha256=e.get("sha256"),
                extra={"url": e["url"]},
            )

    def fetch(self, remote: RemoteFile, dest: Path, do_overwrite: bool = False) -> None:
        self._resumable_download(
            self._client,
            remote.extra["url"],
            dest,
            expected_sha256=remote.sha256,
            do_overwrite=do_overwrite,
        )

    @classmethod
    def _make_client(
        cls,
        auth: tuple[str, str] | None = None,
        timeout: tuple[float, float] = (10.0, 60.0),
        max_retries: int = 5,
        transport: httpx.BaseTransport | None = None,
    ) -> httpx.Client:
        """Build an :class:`httpx.Client` with a tenacity-wrapped ``get``.

        Args:
            auth: Optional ``(username, password)`` for Basic auth — e.g. PhysioNet credentials.
            timeout: ``(connect_timeout, read_timeout)`` in seconds.
            max_retries: How many times to retry on transient failures before giving up.
            transport: Optional :class:`httpx.BaseTransport` override. Defaults to the
                standard HTTP transport; pass an :class:`httpx.MockTransport` to stub out
                the wire for tests without reaching into the returned client's private
                attributes.

        The returned client's ``get`` transparently retries on connection / read-timeout /
        5xx errors with exponential backoff. 4xx errors (wrong URL, bad auth) surface
        immediately. Streaming downloads (``client.stream``) are **not** wrapped here —
        they go through :meth:`_resumable_download`, which surfaces mid-stream errors and
        relies on the ``.part`` file + ``Range: bytes=N-`` for retry-across-the-whole-file
        via re-invocation.

        Examples:
            >>> client = HTTPSource._make_client()
            >>> isinstance(client, httpx.Client)
            True
            >>> client.close()

            Basic auth is threaded through unchanged:

            >>> client = HTTPSource._make_client(auth=("user", "pass"))
            >>> client.auth
            <httpx.BasicAuth object at 0x...>
            >>> client.close()

            5xx responses are retried; 4xx fails fast. The test below uses the public
            ``transport=`` param rather than reaching into the client's private attributes:

            >>> import httpx as _httpx
            >>> attempts = []
            >>> def flaky_then_ok(request):
            ...     attempts.append(None)
            ...     return _httpx.Response(503 if len(attempts) < 3 else 200, text="ok")
            >>> client = HTTPSource._make_client(
            ...     max_retries=5, transport=_httpx.MockTransport(flaky_then_ok)
            ... )
            >>> r = client.get("https://example.com/x")
            >>> r.status_code
            200
            >>> len(attempts)  # 2 retries before the 200
            3
            >>> client.close()
        """
        connect_timeout, read_timeout = timeout
        client_kwargs: dict = {
            "auth": httpx.BasicAuth(*auth) if auth else None,
            "timeout": httpx.Timeout(
                connect=connect_timeout, read=read_timeout, write=read_timeout, pool=60.0
            ),
            "follow_redirects": True,
        }
        if transport is not None:
            client_kwargs["transport"] = transport
        client = httpx.Client(**client_kwargs)

        retry_decorator = retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type((*cls._RETRY_EXC, httpx.HTTPStatusError)),
            reraise=True,
        )
        original_get = client.get

        def _get_with_5xx_retry(*args, **kwargs):
            response = original_get(*args, **kwargs)
            # Raise only on 5xx so tenacity retries. 4xx passes through unwrapped and the
            # caller decides (typically via ``raise_for_status()`` in the calling method).
            if 500 <= response.status_code < 600:
                response.raise_for_status()
            return response

        client.get = retry_decorator(_get_with_5xx_retry)  # type: ignore[method-assign]
        return client

    @staticmethod
    def _resumable_download(
        client: httpx.Client,
        url: str,
        dest: Path,
        expected_sha256: str | None = None,
        chunk_size: int = 1024 * 1024,
        do_overwrite: bool = False,
    ) -> None:
        """HTTP GET with ``.part`` staging, Range resume, SHA-256 verify, atomic rename.

        Invariant on successful return: ``dest`` exists with correct contents, no ``.part``
        file remains. On failure: ``dest`` does not exist; the ``.part`` file may persist
        and will be picked up by a subsequent call via a ``Range: bytes=<offset>-`` resume
        header.

        Args:
            client: A configured :class:`httpx.Client` (ideally from :meth:`_make_client`).
            url: Absolute URL to fetch.
            dest: Final destination path. Parent dirs must already exist.
            expected_sha256: Optional SHA-256 digest (lowercase hex). If set, verified
                after write; mismatch raises :class:`ChecksumError` and deletes the
                ``.part``.
            chunk_size: Bytes per streamed chunk.
            do_overwrite: If ``True``, delete any pre-existing ``dest`` and ``.part`` files
                before fetching — forces a fresh download regardless of cached state. If
                ``False`` (default), existing ``.part`` files trigger a Range-resume and
                existing ``dest`` files with matching ``expected_sha256`` are treated as
                no-ops.

        Raises:
            ChecksumError: If ``expected_sha256`` is set and doesn't match.
            httpx.HTTPStatusError: If the server returns 4xx/5xx.
        """
        part = dest.with_name(dest.name + ".part")

        if do_overwrite:
            if dest.exists():
                dest.unlink()
            if part.exists():
                part.unlink()
        else:
            # Already downloaded + verified → no-op.
            if dest.exists():
                if expected_sha256 is None or sha256_of(dest) == expected_sha256:
                    return
                logger.warning(f"Re-downloading {dest}: existing file failed SHA-256 check.")
                dest.unlink()

        resume_from = part.stat().st_size if part.exists() else 0

        # Range-resume retry loop: if the server rejects the Range or returns a mismatched
        # 206 (or the source file changed between runs, producing 416), we restart from
        # byte 0 after clearing the `.part`. Without this, a mismatched 206 silently
        # appends the wrong bytes to the existing `.part` — undetectable except by a
        # SHA-256 mismatch after the fact.
        while True:
            headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
            with client.stream("GET", url, headers=headers) as r:
                # 416 "Range Not Satisfiable" — remote file shrank or changed; restart.
                if resume_from and r.status_code == 416:
                    logger.warning(f"Server rejected resume for {url} with 416; restarting from byte 0.")
                    if part.exists():
                        part.unlink()
                    resume_from = 0
                    continue
                r.raise_for_status()
                if resume_from:
                    # Server ignored Range (200 instead of 206) → restart.
                    if r.status_code == 200:
                        logger.info(f"Server ignored Range for {url}; restarting from byte 0.")
                        if part.exists():
                            part.unlink()
                        resume_from = 0
                        continue
                    # Validate Content-Range starts at our requested offset. Without this,
                    # a server returning 206 with a shifted range silently corrupts `.part`.
                    if not HTTPSource._content_range_starts_at(r.headers.get("Content-Range"), resume_from):
                        logger.warning(
                            f"Server returned mismatched Content-Range for {url} "
                            f"(got {r.headers.get('Content-Range')!r} for "
                            f"resume_from={resume_from}); restarting from byte 0."
                        )
                        if part.exists():
                            part.unlink()
                        resume_from = 0
                        continue
                mode = "ab" if resume_from else "wb"
                with part.open(mode) as f:
                    for chunk in r.iter_bytes(chunk_size=chunk_size):
                        f.write(chunk)
            break

        if expected_sha256 is not None:
            actual = sha256_of(part)
            if actual != expected_sha256:
                part.unlink()
                raise ChecksumError(url, expected_sha256, actual)

        part.replace(dest)

    @staticmethod
    def _content_range_starts_at(header: str | None, expected_start: int) -> bool:
        """Parse an HTTP ``Content-Range`` header and verify it begins at ``expected_start``.

        ``Content-Range: bytes <start>-<end>/<total>`` (per RFC 7233). Only valid when the
        server sends a 206 Partial Content response. Returns ``False`` for a missing,
        malformed, or mismatched header — the caller is expected to restart the download
        on ``False``.

        Examples:
            >>> HTTPSource._content_range_starts_at("bytes 100-999/10000", 100)
            True
            >>> HTTPSource._content_range_starts_at("bytes 100-999/10000", 200)  # start mismatch
            False
            >>> HTTPSource._content_range_starts_at(None, 100)  # missing header
            False
            >>> HTTPSource._content_range_starts_at("garbage", 100)  # malformed
            False
            >>> HTTPSource._content_range_starts_at("bytes */10000", 100)  # unsatisfied-range
            False
        """
        if not header or not header.startswith("bytes "):
            return False
        range_spec = header[6:].split("/", 1)[0]
        start_str, sep, _end = range_spec.partition("-")
        return bool(sep) and start_str.isdigit() and int(start_str) == expected_start

    @staticmethod
    def _normalize(entry: str | dict) -> dict:
        """Normalize a URL entry to ``{"url", "rel_path", "sha256"?, "size"?}``.

        Examples:
            >>> HTTPSource._normalize("https://example.com/foo.csv")
            {'url': 'https://example.com/foo.csv', 'rel_path': 'foo.csv'}

            >>> HTTPSource._normalize({"url": "https://example.com/foo.csv", "sha256": "abc"})
            {'url': 'https://example.com/foo.csv', 'sha256': 'abc', 'rel_path': 'foo.csv'}

            Explicit ``rel_path`` wins over the URL-derived default:

            >>> HTTPSource._normalize(
            ...     {"url": "https://example.com/foo.csv", "rel_path": "lookups/foo.csv"}
            ... )
            {'url': 'https://example.com/foo.csv', 'rel_path': 'lookups/foo.csv'}

            Raises on missing ``url`` or bad type:

            >>> HTTPSource._normalize({"sha256": "abc"})
            Traceback (most recent call last):
                ...
            ValueError: HTTPSource url entry is missing 'url': {'sha256': 'abc'}
            >>> HTTPSource._normalize(42)
            Traceback (most recent call last):
                ...
            TypeError: HTTPSource url entry must be a str or dict, got int: 42
        """
        if isinstance(entry, str):
            return {"url": entry, "rel_path": HTTPSource._filename_from_url(entry)}
        if isinstance(entry, dict):
            if "url" not in entry:
                raise ValueError(f"HTTPSource url entry is missing 'url': {entry}")
            out = dict(entry)
            out.setdefault("rel_path", HTTPSource._filename_from_url(entry["url"]))
            return out
        raise TypeError(f"HTTPSource url entry must be a str or dict, got {type(entry).__name__}: {entry}")

    @staticmethod
    def _filename_from_url(url: str) -> str:
        """Derive a filesystem-friendly rel_path from ``url``.

        Examples:
            >>> HTTPSource._filename_from_url("https://example.com/foo.csv")
            'foo.csv'
            >>> HTTPSource._filename_from_url("https://example.com/path/to/bar.csv.gz")
            'bar.csv.gz'
            >>> HTTPSource._filename_from_url("https://example.com/")
            'index.html'
            >>> HTTPSource._filename_from_url("https://example.com")
            'index.html'
        """
        return Path(urlparse(url).path).name or "index.html"
