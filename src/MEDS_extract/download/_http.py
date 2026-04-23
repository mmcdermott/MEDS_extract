"""Shared HTTP primitives for :class:`HTTPSource` and :class:`PhysioNetSource`.

``_resumable_download`` is the core download primitive: ``.part``-file staging, Range-based
resume, SHA-256 verify, atomic rename into place. Every HTTP-backed :class:`Source` in
:mod:`MEDS_extract.download.backends` calls through it. Note that the streaming read itself
(``client.stream("GET", ...)`` inside ``_resumable_download``) is **not** auto-retried —
a transient mid-stream failure surfaces to the caller; the caller can simply re-invoke
``_resumable_download`` and the ``.part`` file will be picked up via ``Range: bytes=N-``.
The ``Fetcher`` orchestrator does not itself retry; if retry-across-the-whole-file is
desired, wrap the ``Fetcher.fetch_all`` call in a tenacity decorator at the call site.

``_make_httpx_client`` wraps :class:`httpx.Client.get` with a :mod:`tenacity`-based retry
policy for transient errors (``5xx`` + connection / read timeouts). Non-streaming requests
— e.g. the ``SHA256SUMS.txt`` manifest fetch in :class:`PhysioNetSource` — retry
automatically; streaming requests go through the primitive above.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

try:
    import httpx
except ImportError as e:  # pragma: no cover
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
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "MEDS_extract.download requires the 'download' extra. "
        "Install with: pip install 'MEDS_extract[download]'"
    ) from e

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


# Transports consider these retriable. 4xx (non-429) errors are NOT retried — those mean the
# URL is wrong or auth failed, and retrying makes things worse.
_RETRY_EXC = (
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
)


class ChecksumError(ValueError):
    """Raised when a downloaded file's SHA-256 doesn't match the expected digest."""

    def __init__(self, url: str, expected: str, actual: str):
        self.url = url
        self.expected = expected
        self.actual = actual
        super().__init__(f"SHA-256 mismatch for {url}: expected {expected}, got {actual}")


def _sha256(fp: Path) -> str:
    """Compute the SHA-256 of ``fp``, streaming 1 MiB at a time.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.NamedTemporaryFile(delete=False) as tmp:
        ...     _ = tmp.write(b"hello world")
        ...     fp = Path(tmp.name)
        >>> _sha256(fp)
        'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
        >>> fp.unlink()
    """
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_httpx_client(
    auth: tuple[str, str] | None = None,
    timeout: tuple[float, float] = (10.0, 60.0),
    max_retries: int = 5,
) -> httpx.Client:
    """Build an :class:`httpx.Client` with a tenacity-wrapped ``get``.

    Args:
        auth: Optional ``(username, password)`` for Basic auth — e.g. PhysioNet credentials.
        timeout: ``(connect_timeout, read_timeout)`` in seconds.
        max_retries: How many times to retry on transient failures before giving up.

    The returned client has the same public interface as a plain :class:`httpx.Client`, but
    its ``get`` method transparently retries on connection / read-timeout / 5xx errors with
    exponential backoff. 4xx errors (wrong URL, bad auth) surface immediately. Streaming
    downloads (``client.stream``) are **not** wrapped here — they go through
    :func:`_resumable_download`, which surfaces mid-stream errors to the caller and relies
    on the ``.part`` file + ``Range: bytes=N-`` for retry-across-the-whole-file via
    re-invocation.

    Examples:
        >>> client = _make_httpx_client()
        >>> isinstance(client, httpx.Client)
        True
        >>> client.close()

        Basic auth is threaded through unchanged:

        >>> client = _make_httpx_client(auth=("user", "pass"))
        >>> client.auth
        <httpx.BasicAuth object at 0x...>
        >>> client.close()

        5xx responses are retried; 4xx fails fast (bad URL / bad auth shouldn't spam
        retries):

        >>> import httpx as _httpx
        >>> attempts = []
        >>> def flaky_then_ok(request):
        ...     attempts.append(None)
        ...     return _httpx.Response(503 if len(attempts) < 3 else 200, text="ok")
        >>> client = _make_httpx_client(max_retries=5)
        >>> client._transport = _httpx.MockTransport(flaky_then_ok)
        >>> r = client.get("https://example.com/x")
        >>> r.status_code
        200
        >>> len(attempts)  # 2 retries before the 200
        3
        >>> client.close()
    """
    connect_timeout, read_timeout = timeout
    client = httpx.Client(
        auth=httpx.BasicAuth(*auth) if auth else None,
        timeout=httpx.Timeout(connect=connect_timeout, read=read_timeout, write=read_timeout, pool=60.0),
        follow_redirects=True,
    )

    # Retry on transient transport failures AND on 5xx responses. 5xx is modeled as
    # `httpx.HTTPStatusError` by calling `raise_for_status()` inside the wrapped function
    # so tenacity can observe and retry it; we then re-fetch the response on retry.
    retry_decorator = retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((*_RETRY_EXC, httpx.HTTPStatusError)),
        reraise=True,
    )
    original_get = client.get

    def _get_with_5xx_retry(*args, **kwargs):
        response = original_get(*args, **kwargs)
        # Raise HTTPStatusError so tenacity retries; let 4xx pass through unwrapped since
        # the retry predicate doesn't distinguish, but fail fast if still 4xx after the
        # final attempt by raising below the retry layer.
        if 500 <= response.status_code < 600:
            response.raise_for_status()
        return response

    client.get = retry_decorator(_get_with_5xx_retry)  # type: ignore[method-assign]
    # Can't wrap `stream` the same way — it returns a context manager, not a response. See
    # the module docstring for how streaming-side retry is handled (caller re-invokes
    # `_resumable_download`, which picks up the `.part` via Range).
    return client


def _resumable_download(
    client: httpx.Client,
    url: str,
    dest: Path,
    expected_sha256: str | None = None,
    chunk_size: int = 1024 * 1024,
) -> None:
    """HTTP GET with ``.part`` staging, Range resume, SHA-256 verify, atomic rename.

    Invariant on successful return: ``dest`` exists with correct contents, no ``.part``
    file remains. On failure: ``dest`` does not exist; the ``.part`` file may persist and
    will be picked up by a subsequent call via a ``Range: bytes=<offset>-`` resume header.

    Args:
        client: A configured :class:`httpx.Client` (ideally from :func:`_make_httpx_client`).
        url: Absolute URL to fetch.
        dest: Final destination path. Parent dirs must already exist.
        expected_sha256: Optional SHA-256 digest (lowercase hex). If set, verified after
            write; mismatch raises :class:`ChecksumError` and deletes the ``.part``.
        chunk_size: Bytes per streamed chunk.

    Raises:
        ChecksumError: If ``expected_sha256`` is set and doesn't match.
        httpx.HTTPStatusError: If the server returns 4xx/5xx.
    """
    part = dest.with_name(dest.name + ".part")

    # Already downloaded + verified → no-op.
    if dest.exists():
        if expected_sha256 is None or _sha256(dest) == expected_sha256:
            return
        logger.warning(f"Re-downloading {dest}: existing file failed SHA-256 check.")
        dest.unlink()

    resume_from = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

    with client.stream("GET", url, headers=headers) as r:
        r.raise_for_status()
        # Server ignored the Range header (returned 200 instead of 206) → restart.
        if resume_from and r.status_code == 200:
            logger.info(f"Server ignored Range for {url}; restarting from byte 0.")
            resume_from = 0
        mode = "ab" if resume_from else "wb"
        with part.open(mode) as f:
            for chunk in r.iter_bytes(chunk_size=chunk_size):
                f.write(chunk)

    if expected_sha256 is not None:
        actual = _sha256(part)
        if actual != expected_sha256:
            part.unlink()
            raise ChecksumError(url, expected_sha256, actual)

    part.replace(dest)
