"""HTTP-backed :class:`Source`.

This module absorbs the helpers that used to live in a separate ``_http`` submodule.
Everything HTTP-specific — client construction with tenacity retry, ``.part``-file
Range-resume download, ``Content-Range`` validation, URL-entry normalization — is now
attached to :class:`HTTPSource` as static methods (or plain module-level helpers where
the logic is generic). :class:`~MEDS_extract.download.backends.physionet.PhysioNetSource`
inherits from :class:`HTTPSource` and only overrides :meth:`_list_files`.

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

from .._types import ChecksumError, RemoteFile, sha256_of
from ..source import Source

try:
    import httpx
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
except ImportError as e:
    raise ImportError(
        "MEDS_extract.download requires the 'download' extra. "
        "Install with: pip install 'MEDS_extract[download]'"
    ) from e

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..fetcher import Fetcher

logger = logging.getLogger(__name__)

# ``_resumable_download`` retries its Range-resume loop at most this many times before
# raising a :class:`RuntimeError`. By construction the loop terminates in at most 2
# iterations (a single restart zeros ``resume_from``, which guards every restart
# branch), so the cap is defense-in-depth against a future refactor.
_MAX_RESUME_ATTEMPTS = 3


class HTTPSource(Source):
    """A :class:`Source` backed by an explicit list of HTTP URLs.

    Use this for shared metadata downloads where the file list is known up-front — e.g.
    MIMIC's ``common:`` block of concept-map CSVs from ``raw.githubusercontent.com``. No
    crawling, no manifest parsing.

    Each URL entry can be either a plain string or a dict with optional ``sha256``,
    ``rel_path``, and ``size`` fields. ``rel_path`` defaults to the URL's basename.

    Args:
        urls: List of URL entries — plain strings or dicts. Subclasses that discover URLs
            at ``_list_files`` time (e.g. :class:`PhysioNetSource`) may pass ``None``.
        fetcher: :class:`~MEDS_extract.download.fetcher.Fetcher` policy injected at
            construction. ``None`` (default) builds a default fetcher.
        client: Optional pre-built :class:`httpx.Client`. When omitted, one is built via
            :meth:`_make_client` with the remaining kwargs.
        auth, headers, timeout, max_attempts, transport: Forwarded to :meth:`_make_client`
            when ``client`` is not provided. ``headers`` is a ``{name: value}`` mapping
            applied as default headers on every request — used for API-key auth
            (``X-Dataverse-key``, bearer tokens) and content negotiation (``Accept:``).

    Examples:
        Plain string URLs resolve to basename-based relative paths:

        >>> src = HTTPSource(urls=["https://example.com/foo.csv", "https://example.com/bar.csv"])
        >>> [r.rel_path for r in src._list_files()]
        ['foo.csv', 'bar.csv']

        Dict entries can override ``rel_path`` and provide a checksum:

        >>> src = HTTPSource(
        ...     urls=[
        ...         {"url": "https://example.com/foo.csv", "rel_path": "lookups/foo.csv"},
        ...         {"url": "https://example.com/bar.csv", "sha256": "abc123"},
        ...     ]
        ... )
        >>> fs = list(src._list_files())
        >>> fs[0].rel_path, fs[0].sha256
        ('lookups/foo.csv', None)
        >>> fs[1].rel_path, fs[1].sha256
        ('bar.csv', 'abc123')

        URLs without a path component fall back to ``"index.html"``:

        >>> src = HTTPSource(urls=["https://example.com/"])
        >>> [r.rel_path for r in src._list_files()]
        ['index.html']

        Per-entry ``unarchive`` / ``cleanup_archive`` propagate through the manifest
        to the :class:`~MEDS_extract.download.fetcher.Fetcher`'s post-fetch hook —
        the motivating case is AUMCdb on DANS Data Stations, which ships its entire
        dataset as a single DataVerse zip that we want unpacked into ``dest_dir``
        and then discarded:

        >>> src = HTTPSource(urls=[{
        ...     "url": "https://example.com/AUMCdb.zip",
        ...     "unarchive": "zip",
        ...     "cleanup_archive": True,
        ... }])
        >>> [(r.rel_path, r.unarchive, r.cleanup_archive) for r in src._list_files()]
        [('AUMCdb.zip', 'zip', True)]
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
        *,
        fetcher: Fetcher | None = None,
        client: httpx.Client | None = None,
        auth: tuple[str, str] | None = None,
        headers: dict[str, str] | None = None,
        timeout: tuple[float, float] = (10.0, 60.0),
        max_attempts: int = 5,
        transport: httpx.BaseTransport | None = None,
    ):
        super().__init__(fetcher=fetcher)
        self._entries = [self._normalize(u) for u in (urls or [])]
        # Track client ownership: only close clients we built ourselves. An injected
        # ``client`` is the caller's to manage — typically tests with a shared
        # ``MockTransport``, which is reused across calls.
        self._owns_client = client is None
        self._client = (
            client
            if client is not None
            else self._make_client(
                auth=auth,
                headers=headers,
                timeout=timeout,
                max_attempts=max_attempts,
                transport=transport,
            )
        )

    def _list_files(self) -> Iterable[RemoteFile]:
        for e in self._entries:
            yield RemoteFile(
                rel_path=e["rel_path"],
                size=e.get("size"),
                sha256=e.get("sha256"),
                unarchive=e.get("unarchive"),
                cleanup_archive=e.get("cleanup_archive"),  # tri-state: None defers to unarchive mode
                extra={"url": e["url"]},
            )

    def _fetch(self, remote: RemoteFile, dest: Path) -> None:
        self._resumable_download(
            self._client,
            remote.extra["url"],
            dest,
            expected_sha256=remote.sha256,
        )

    def close(self) -> None:
        """Close the owned httpx client; no-op if the client was injected."""
        if self._owns_client:
            self._client.close()

    @classmethod
    def _make_client(
        cls,
        auth: tuple[str, str] | None = None,
        headers: dict[str, str] | None = None,
        timeout: tuple[float, float] = (10.0, 60.0),
        max_attempts: int = 5,
        transport: httpx.BaseTransport | None = None,
    ) -> httpx.Client:
        """Build an :class:`httpx.Client` with a tenacity-wrapped ``get``.

        Args:
            auth: Optional ``(username, password)`` for Basic auth — e.g. PhysioNet credentials.
            headers: Optional ``{name: value}`` mapping applied as default headers on every
                request the client issues (both ``list_files`` GETs and streaming ``.part``
                downloads). Intended for API-key auth (DataVerse's ``X-Dataverse-key``,
                generic bearer tokens) and content negotiation (``Accept:``). ``None``
                behaves like absent.
            timeout: ``(connect_timeout, read_timeout)`` in seconds.
            max_attempts: Total number of attempts (including the first) before giving up.
                ``max_attempts=5`` = 1 initial try + up to 4 retries.
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
            ...     max_attempts=5, transport=_httpx.MockTransport(flaky_then_ok)
            ... )
            >>> r = client.get("https://example.com/x")
            >>> r.status_code
            200
            >>> len(attempts)  # 2 retries before the 200
            3
            >>> client.close()

            Custom ``headers`` reach the transport on every request — the motivating case
            is DataVerse's ``X-Dataverse-key`` API-key auth, but the same kwarg covers
            bearer tokens and ``Accept:`` content negotiation:

            >>> seen_headers = []
            >>> def capture(request):
            ...     seen_headers.append(dict(request.headers))
            ...     return _httpx.Response(200, text="ok")
            >>> client = HTTPSource._make_client(
            ...     headers={"X-Dataverse-key": "secret-token", "Accept": "application/json"},
            ...     transport=_httpx.MockTransport(capture),
            ... )
            >>> _ = client.get("https://example.com/x")
            >>> seen_headers[0]["x-dataverse-key"]
            'secret-token'
            >>> seen_headers[0]["accept"]
            'application/json'
            >>> client.close()
        """
        connect_timeout, read_timeout = timeout
        client_kwargs: dict = {
            "auth": httpx.BasicAuth(*auth) if auth else None,
            "headers": headers,
            "timeout": httpx.Timeout(
                connect=connect_timeout, read=read_timeout, write=read_timeout, pool=60.0
            ),
            "follow_redirects": True,
        }
        if transport is not None:
            client_kwargs["transport"] = transport
        client = httpx.Client(**client_kwargs)

        retry_decorator = retry(
            stop=stop_after_attempt(max_attempts),
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
    ) -> None:
        """HTTP GET with ``.part`` staging, Range resume, SHA-256 verify, atomic rename.

        Precondition: ``dest`` does not exist on entry. A pre-existing ``.part`` file IS
        allowed and will be picked up via ``Range: bytes=N-`` for resume. Overwrite /
        cleanup semantics live on :meth:`Source.fetch`; this primitive trusts its caller.

        Invariant on successful return: ``dest`` exists with correct contents, no ``.part``
        file remains.

        Args:
            client: A configured :class:`httpx.Client` (ideally from :meth:`_make_client`).
            url: Absolute URL to fetch.
            dest: Final destination path. Must not exist; parent dirs must exist.
            expected_sha256: Optional SHA-256 digest (lowercase hex). If set, verified
                after write; mismatch raises :class:`ChecksumError` and deletes the
                ``.part``.
            chunk_size: Bytes per streamed chunk.

        Raises:
            ChecksumError: If ``expected_sha256`` is set and doesn't match.
            httpx.HTTPStatusError: If the server returns 4xx/5xx.
        """
        part = dest.with_name(dest.name + ".part")

        # Resume-without-checksum is unsafe: if the remote file changed between runs, the
        # server may happily serve a 206 starting at the stored ``resume_from`` and we'd
        # silently append fresh bytes to a stale prefix. Without an ``expected_sha256`` to
        # catch the corruption after, the only safe move is to discard the ``.part`` and
        # restart. Callers who want resume support should include ``sha256`` in the manifest.
        if expected_sha256 is None and part.exists():
            logger.warning(f"Discarding .part for {url}: resume requires an expected_sha256 to be safe.")
            part.unlink()

        resume_from = part.stat().st_size if part.exists() else 0

        # Range-resume retry loop: if the server rejects the Range or returns a mismatched
        # 206 (or the source file changed between runs, producing 416), we restart from
        # byte 0 after clearing the `.part`. Without this, a mismatched 206 silently
        # appends the wrong bytes to the existing `.part` — undetectable except by a
        # SHA-256 mismatch after the fact.
        #
        # Iteration cap: by construction, a single restart zeroes ``resume_from`` and the
        # next iteration's ``if resume_from and ...`` guards short-circuit all three
        # restart branches. So the loop terminates in at most 2 iterations. The
        # ``range(_MAX_RESUME_ATTEMPTS)`` cap is defense-in-depth against a future
        # refactor breaking that invariant (e.g. someone dropping the ``resume_from = 0``
        # assignment) — better an explicit RuntimeError than a silent infinite loop.
        for _ in range(_MAX_RESUME_ATTEMPTS):
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
        else:
            # Exhausted the iteration cap without a successful write+break — a bug
            # elsewhere broke the "restart zeros resume_from" invariant that makes the
            # loop terminate. Surface it loudly rather than looping forever.
            raise RuntimeError(
                f"_resumable_download exhausted {_MAX_RESUME_ATTEMPTS} restart attempts "
                f"for {url}; range-resume loop failed to converge. This indicates a bug "
                "in the restart logic — the expected invariant is that each restart "
                "resets resume_from to 0, which prevents any subsequent restart."
            )

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
