"""HTTP-backed :class:`Source`.

Everything HTTP-specific — client construction, Range-resume streaming,
``Content-Range`` validation, URL-entry normalization — is attached to
:class:`HTTPSource` as static methods (or module-level helpers where the logic is
generic). :class:`~MEDS_extract.download.backends.physionet.PhysioNetSource` inherits
from :class:`HTTPSource` and only overrides :meth:`_list_files` (plus its
constructor).

Both request paths share one retry policy (:meth:`HTTPSource._retrying` — transient
transport errors per ``_RETRY_EXC``: connect failures, timeouts, mid-body TCP resets
(``ReadError`` / ``WriteError``), protocol errors — plus 5xx responses), with
exponential backoff capped at the same ``max_attempts``. The policy lives on the
*source*, not the client, so it applies whether the client was built by
:meth:`HTTPSource._make_client` or injected via ``client=``:

- Manifest GETs (:meth:`HTTPSource._get`, used by ``_list_files``) retry each
  whole request.
- Streaming downloads (:meth:`HTTPSource._pull`) retry the same classes around
  each whole :meth:`HTTPSource._resumable_stream` attempt — a mid-body failure
  leaves the partial ``target`` in place, so the retried attempt resumes via
  ``Range: bytes=N-`` rather than starting over.

Each backoff sleep is logged at WARNING (tenacity ``before_sleep_log``), so a
flaky-network run is distinguishable from a hang. 4xx errors surface immediately
on both paths — retrying a bad URL or bad auth makes things worse, not better.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from ..source import RemoteFile, Source

try:
    import httpx
    from tenacity import (
        Retrying,
        before_sleep_log,
        retry_if_exception,
        stop_after_attempt,
        wait_exponential,
    )
except ImportError as e:
    raise ImportError(
        "The 'http' and 'physionet' download backends require the 'download' extra "
        "(httpx, tenacity). Install with: pip install 'MEDS_extract[download]'. "
        "Other download sources (fsspec) and the CLI work without it."
    ) from e

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tenacity.wait import wait_base

logger = logging.getLogger(__name__)

# ``_resumable_stream`` retries its Range-resume loop at most this many times before
# raising a :class:`RuntimeError`. By construction the loop terminates in at most 2
# iterations (a single restart zeros ``resume_from``, which guards every restart
# branch), so the cap is defense-in-depth against a future refactor.
_MAX_RESUME_ATTEMPTS = 3

# Default backoff shared by the manifest-GET retry and the streaming retry. Tests and
# doctests pass an explicit ``retry_wait`` (e.g. ``tenacity.wait_fixed(0)``) so
# exercising the retry paths costs no real sleep time.
_RETRY_WAIT = wait_exponential(multiplier=1, min=1, max=30)


class HTTPSource(Source):
    """A :class:`Source` backed by an explicit list of HTTP URLs.

    Use this for shared metadata downloads where the file list is known up-front — e.g.
    MIMIC's ``common:`` block of concept-map CSVs from ``raw.githubusercontent.com``. No
    crawling, no manifest parsing.

    Each URL entry can be either a plain string or a dict with optional ``sha256``
    and ``rel_path`` fields. ``rel_path`` defaults to the URL's basename.

    Args:
        urls: List of URL entries — plain strings or dicts. Subclasses that discover URLs
            at :meth:`_list_files` time (e.g. :class:`PhysioNetSource`) may pass ``None``.
        client: Optional pre-built :class:`httpx.Client`. When omitted, one is built via
            :meth:`_make_client` with the remaining kwargs.
        auth, headers, timeout, transport: Forwarded to :meth:`_make_client` when
            ``client`` is not provided. ``headers`` is a ``{name: value}``
            mapping applied as default headers on every request — used for API-key
            auth (``X-Dataverse-key``, bearer tokens) and content negotiation
            (``Accept:``).
        max_attempts, retry_wait: Govern the shared retry policy
            (:meth:`_retrying`) applied to both manifest GETs (:meth:`_get`) and
            streaming downloads (:meth:`_pull`) — regardless of whether ``client``
            was injected.
        include, exclude: Optional :mod:`fnmatch` globs applied to the manifest —
            see :class:`~MEDS_extract.download.source.Source`.

    Examples:
        Plain string URLs resolve to basename-based relative paths:

        >>> src = HTTPSource(urls=["https://example.com/foo.csv", "https://example.com/bar.csv"])
        >>> [r.rel_path for r in src._list_files()]
        ['foo.csv', 'bar.csv']
        >>> src.close()

        Dict entries can override ``rel_path`` and provide a checksum:

        >>> src = HTTPSource(
        ...     urls=[
        ...         {"url": "https://example.com/foo.csv", "rel_path": "lookups/foo.csv"},
        ...         {"url": "https://example.com/bar.csv", "sha256": "ab" * 32},
        ...     ]
        ... )
        >>> fs = list(src._list_files())
        >>> fs[0].rel_path, fs[0].sha256
        ('lookups/foo.csv', None)
        >>> fs[1].rel_path, fs[1].sha256 == "ab" * 32
        ('bar.csv', True)
        >>> src.close()

        URLs without a path component fall back to ``"index.html"``:

        >>> with HTTPSource(urls=["https://example.com/"]) as src:
        ...     [r.rel_path for r in src._list_files()]
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
        # A mid-body TCP reset surfaces as ReadError (WriteError for uploads),
        # not RemoteProtocolError — both are as transient as the timeouts above.
        httpx.ReadError,
        httpx.WriteError,
        httpx.RemoteProtocolError,
    )

    def __init__(
        self,
        urls: list[str | dict] | None = None,
        client: httpx.Client | None = None,
        auth: tuple[str, str] | None = None,
        headers: dict[str, str] | None = None,
        timeout: tuple[float, float] = (10.0, 60.0),
        max_attempts: int = 5,
        transport: httpx.BaseTransport | None = None,
        retry_wait: wait_base | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ):
        super().__init__(include=include, exclude=exclude)
        self._entries = [self._normalize(u) for u in (urls or [])]
        self._max_attempts = max_attempts
        self._retry_wait = retry_wait if retry_wait is not None else _RETRY_WAIT
        # Track client ownership: only close clients we built ourselves. An injected
        # ``client`` is the caller's to manage — typically tests with a shared
        # ``MockTransport``, which is reused across calls.
        self._owns_client = client is None
        self._client = (
            client
            if client is not None
            else self._make_client(auth=auth, headers=headers, timeout=timeout, transport=transport)
        )

    def _list_files(self) -> Iterable[RemoteFile]:
        yield from self._entries

    def _retrying(self) -> Retrying:
        """The shared retry policy for both request paths (``_get`` and ``_pull``).

        Built from ``self._max_attempts`` / ``self._retry_wait``, so it applies
        identically whether the httpx client was built by :meth:`_make_client` or
        injected via ``client=``. Each backoff sleep logs a WARNING naming the
        exception and wait time, so retries are distinguishable from a hang.
        """
        return Retrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=self._retry_wait,
            retry=retry_if_exception(self._should_retry_stream),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

    def _get(self, url: str) -> httpx.Response:
        """Manifest-style GET with the source's retry policy applied.

        Raises inside the retry loop only on 5xx (so tenacity retries alongside
        the transient transport errors); 4xx responses are returned unwrapped and
        the caller decides — typically via ``raise_for_status()`` — so a bad URL
        or bad auth fails fast rather than being retried.

        Examples:
            5xx responses are retried; the third attempt succeeds. This works
            identically for an injected ``client=`` — the retry policy lives on
            the source, not the client:

            >>> import httpx as _httpx
            >>> from tenacity import wait_fixed
            >>> attempts = []
            >>> def flaky_then_ok(request):
            ...     attempts.append(None)
            ...     return _httpx.Response(503 if len(attempts) < 3 else 200, text="ok")
            >>> client = _httpx.Client(transport=_httpx.MockTransport(flaky_then_ok))
            >>> src = HTTPSource(urls=[], client=client, max_attempts=5, retry_wait=wait_fixed(0))
            >>> src._get("https://example.com/x").status_code
            200
            >>> len(attempts)  # 2 retries before the 200
            3
            >>> client.close()

            4xx is not retried — the response comes back unwrapped after one
            attempt:

            >>> attempts.clear()
            >>> def always_404(request):
            ...     attempts.append(None)
            ...     return _httpx.Response(404)
            >>> client = _httpx.Client(transport=_httpx.MockTransport(always_404))
            >>> src = HTTPSource(urls=[], client=client)
            >>> src._get("https://example.com/x").status_code
            404
            >>> len(attempts)
            1
            >>> client.close()
        """

        def _once() -> httpx.Response:
            response = self._client.get(url)
            if 500 <= response.status_code < 600:
                response.raise_for_status()
            return response

        return self._retrying()(_once)

    def _pull(self, source_path: str, target: Path) -> None:
        # Retry the whole resumable-stream attempt on transient failures. A
        # request-phase failure (connect error, 5xx before any bytes arrive) simply
        # re-issues the request; a mid-body failure leaves the enlarged ``target``
        # partial in place, so the next attempt resumes via ``Range: bytes=N-``.
        self._retrying()(self._resumable_stream, self._client, source_path, target)

    @classmethod
    def _should_retry_stream(cls, exc: BaseException) -> bool:
        """Retry transient transport errors and 5xx responses; never 4xx.

        Shared by both request paths: ``_get`` raises only on 5xx inside its
        retry loop (4xx returns unwrapped), and ``_resumable_stream`` calls
        ``raise_for_status`` on everything — so gating ``HTTPStatusError`` on
        ``status_code >= 500`` here is what keeps 404s failing fast on both.
        """
        if isinstance(exc, cls._RETRY_EXC):
            return True
        return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500

    def close(self) -> None:
        """Close the owned httpx client; no-op if the client was injected.

        Examples:
            When no ``client=`` is injected, the source builds and owns one, and
            ``close()`` closes it. A second ``close()`` is a no-op (httpx clients
            are re-close-safe):

            >>> src = HTTPSource(urls=["https://example.com/a.csv"])
            >>> src._owns_client, src._client.is_closed
            (True, False)
            >>> src.close()
            >>> src._client.is_closed
            True
            >>> src.close()  # idempotent

            An injected client belongs to the caller — ``close()`` leaves it open:

            >>> client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
            >>> src = HTTPSource(urls=["https://example.com/a.csv"], client=client)
            >>> src._owns_client
            False
            >>> src.close()
            >>> client.is_closed
            False
            >>> client.close()  # caller cleans up

            The context-manager form closes the owned client on exit:

            >>> with HTTPSource(urls=["https://example.com/a.csv"]) as src:
            ...     inner = src._client
            ...     inner.is_closed
            False
            >>> inner.is_closed
            True
        """
        if self._owns_client:
            self._client.close()

    @classmethod
    def _make_client(
        cls,
        auth: tuple[str, str] | None = None,
        headers: dict[str, str] | None = None,
        timeout: tuple[float, float] = (10.0, 60.0),
        transport: httpx.BaseTransport | None = None,
    ) -> httpx.Client:
        """Build a plain :class:`httpx.Client` — pure client construction, no retry.

        Retry lives on the *source* (:meth:`_retrying`, applied by :meth:`_get`
        and :meth:`_pull`), not on the client — that way an injected ``client=``
        gets exactly the same retry behavior as a client built here.

        Args:
            auth: Optional ``(username, password)`` for Basic auth — e.g. PhysioNet credentials.
            headers: Optional ``{name: value}`` mapping applied as default headers on every
                request the client issues (both ``_list_files`` manifest GETs and streaming
                ``.part`` downloads). Intended for API-key auth (DataVerse's
                ``X-Dataverse-key``, generic bearer tokens) and content negotiation
                (``Accept:``). ``None`` behaves like absent.
            timeout: ``(connect_timeout, read_timeout)`` in seconds.
            transport: Optional :class:`httpx.BaseTransport` override. Defaults to the
                standard HTTP transport; pass an :class:`httpx.MockTransport` to stub out
                the wire for tests without reaching into the returned client's private
                attributes.

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

            Custom ``headers`` reach the transport on every request — the motivating case
            is DataVerse's ``X-Dataverse-key`` API-key auth, but the same kwarg covers
            bearer tokens and ``Accept:`` content negotiation:

            >>> import httpx as _httpx
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
        return httpx.Client(**client_kwargs)

    @staticmethod
    def _resumable_stream(
        client: httpx.Client,
        url: str,
        target: Path,
        chunk_size: int = 1024 * 1024,
    ) -> None:
        """HTTP GET that streams bytes into ``target``, with ``Range``-resume.

        If ``target`` exists, an HTTP ``Range`` request resumes from its end;
        otherwise the download starts from byte 0. On a 416, a mismatched
        ``Content-Range``, or a server that ignores ``Range`` and returns 200,
        the resume is abandoned and the download restarts from byte 0.

        Every request sends ``Accept-Encoding: identity``. Transparent
        content-coding (httpx's default ``gzip, deflate``) would make ``target``
        hold *decoded* bytes while ``Range`` offsets and ``Content-Range``
        validation operate on the *encoded* representation — a resume against a
        compressing server would then pass the offset check yet feed the
        decompressor a mid-stream fragment. Requesting the identity coding keeps
        on-wire bytes, ``target.stat().st_size``, and the manifest's SHA-256 all
        describing the same byte stream.

        Args:
            client: A configured :class:`httpx.Client` (from :meth:`_make_client`).
            url: Absolute URL to fetch.
            target: Path to write into. May already contain partial bytes from a
                prior failed attempt — those are appended to via ``Range``.
            chunk_size: Bytes per streamed chunk.

        Raises:
            httpx.HTTPStatusError: If the server returns 4xx/5xx.
            RuntimeError: If the Range-resume restart loop fails to converge —
                defense-in-depth against a future refactor breaking the loop's
                termination invariant.

        Examples:
            The basic contract: bytes from ``url`` land in ``target``, and every
            request advertises ``Accept-Encoding: identity`` (see above for why):

            >>> def echo_handler(request):
            ...     print(f"Accept-Encoding: {request.headers.get('Accept-Encoding')}")
            ...     return httpx.Response(200, content=b"hello world")
            >>> client = httpx.Client(transport=httpx.MockTransport(echo_handler))
            >>> with tempfile.TemporaryDirectory() as d:
            ...     target = Path(d) / "x.csv.part"
            ...     HTTPSource._resumable_stream(client, "https://example.com/x.csv", target)
            ...     target.read_bytes()
            Accept-Encoding: identity
            b'hello world'
            >>> client.close()

            The Range-resume / 416 / ``Content-Range``-mismatch restart behavior
            is wire-protocol machinery exercised in ``tests/test_download.py``
            (``test_resumable_stream_*``), where multi-request handler state
            machines are more readable than doctests.
        """
        resume_from = target.stat().st_size if target.exists() else 0

        # Range-resume retry loop: if the server rejects the Range or returns a mismatched
        # 206 (or the source file changed between runs, producing 416), we restart from
        # byte 0 after clearing ``target``. Without this, a mismatched 206 silently
        # appends the wrong bytes to the existing file — undetectable except by a
        # SHA-256 mismatch on the wrapper's verify step.
        #
        # Iteration cap: by construction, a single restart zeroes ``resume_from`` and the
        # next iteration's ``if resume_from and ...`` guards short-circuit all three
        # restart branches. So the loop terminates in at most 2 iterations. The
        # ``range(_MAX_RESUME_ATTEMPTS)`` cap is defense-in-depth against a future
        # refactor breaking that invariant (e.g. someone dropping the ``resume_from = 0``
        # assignment) — better an explicit RuntimeError than a silent infinite loop.
        for _ in range(_MAX_RESUME_ATTEMPTS):
            headers = {"Accept-Encoding": "identity"}
            if resume_from:
                headers["Range"] = f"bytes={resume_from}-"
            with client.stream("GET", url, headers=headers) as r:
                # 416 "Range Not Satisfiable" — remote file shrank or changed; restart.
                if resume_from and r.status_code == 416:
                    logger.warning(f"Server rejected resume for {url} with 416; restarting from byte 0.")
                    if target.exists():
                        target.unlink()
                    resume_from = 0
                    continue
                r.raise_for_status()
                if resume_from:
                    # Server ignored Range (200 instead of 206) → restart.
                    if r.status_code == 200:
                        # WARNING for consistency with the 416 and Content-Range
                        # siblings — all three discard the accumulated partial and
                        # re-transfer from byte 0.
                        logger.warning(f"Server ignored Range for {url}; restarting from byte 0.")
                        if target.exists():
                            target.unlink()
                        resume_from = 0
                        continue
                    # Validate Content-Range starts at our requested offset. Without this,
                    # a server returning 206 with a shifted range silently corrupts ``target``.
                    if not HTTPSource._content_range_starts_at(r.headers.get("Content-Range"), resume_from):
                        logger.warning(
                            f"Server returned mismatched Content-Range for {url} "
                            f"(got {r.headers.get('Content-Range')!r} for "
                            f"resume_from={resume_from}); restarting from byte 0."
                        )
                        if target.exists():
                            target.unlink()
                        resume_from = 0
                        continue
                mode = "ab" if resume_from else "wb"
                with target.open(mode) as f:
                    for chunk in r.iter_bytes(chunk_size=chunk_size):
                        f.write(chunk)
            return
        # Exhausted the iteration cap without a successful write+return — a bug
        # elsewhere broke the "restart zeros resume_from" invariant that makes the
        # loop terminate. Surface it loudly rather than looping forever.
        raise RuntimeError(
            f"_resumable_stream exhausted {_MAX_RESUME_ATTEMPTS} restart attempts "
            f"for {url}; range-resume loop failed to converge. This indicates a bug "
            "in the restart logic — the expected invariant is that each restart "
            "resets resume_from to 0, which prevents any subsequent restart."
        )

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
    def _normalize(entry: str | dict) -> RemoteFile:
        """Normalize a URL entry to a validated :class:`RemoteFile`.

        Unknown dict keys are rejected rather than silently dropped — a typo like
        ``sha_256:`` would otherwise leave the download unverified while the user
        believes they pinned a checksum. Digest format/case validation happens in
        :class:`RemoteFile` itself.

        Examples:
            >>> HTTPSource._normalize("https://example.com/foo.csv")
            RemoteFile(rel_path='foo.csv', source_path='https://example.com/foo.csv', sha256=None)

            >>> HTTPSource._normalize({"url": "https://example.com/foo.csv", "sha256": "ab" * 32})
            RemoteFile(rel_path='foo.csv', source_path='https://example.com/foo.csv', sha256='abab...

            Explicit ``rel_path`` wins over the URL-derived default:

            >>> HTTPSource._normalize(
            ...     {"url": "https://example.com/foo.csv", "rel_path": "lookups/foo.csv"}
            ... )
            RemoteFile(rel_path='lookups/foo.csv', source_path='https://example.com/foo.csv', sha256=None)

            Raises on missing ``url``, unknown keys, malformed digests, or bad type:

            >>> HTTPSource._normalize({"sha256": "ab" * 32})
            Traceback (most recent call last):
                ...
            ValueError: HTTPSource url entry is missing 'url': {'sha256': ...
            >>> HTTPSource._normalize({"url": "https://example.com/foo.csv", "sha_256": "ab" * 32})
            Traceback (most recent call last):
                ...
            ValueError: HTTPSource url entry has unknown keys ['sha_256'] ...
            >>> HTTPSource._normalize({"url": "https://example.com/foo.csv", "sha256": "abc"})
            Traceback (most recent call last):
                ...
            ValueError: sha256 must be 64 hex chars, got 'abc'
            >>> HTTPSource._normalize(42)
            Traceback (most recent call last):
                ...
            TypeError: HTTPSource url entry must be a str or dict, got int: 42
        """
        if isinstance(entry, str):
            return RemoteFile(rel_path=HTTPSource._filename_from_url(entry), source_path=entry)
        if isinstance(entry, dict):
            if "url" not in entry:
                raise ValueError(f"HTTPSource url entry is missing 'url': {entry}")
            unknown = sorted(set(entry) - {"url", "rel_path", "sha256"})
            if unknown:
                raise ValueError(
                    f"HTTPSource url entry has unknown keys {unknown} "
                    f"(supported: url, rel_path, sha256): {entry}"
                )
            return RemoteFile(
                rel_path=entry.get("rel_path") or HTTPSource._filename_from_url(entry["url"]),
                source_path=entry["url"],
                sha256=entry.get("sha256"),
            )
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
