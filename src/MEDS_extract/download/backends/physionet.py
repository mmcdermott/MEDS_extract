"""PhysioNet :class:`Source` — ``SHA256SUMS.txt``-driven, no HTML crawl."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..source import RemoteFile
from .http import HTTPSource

if TYPE_CHECKING:
    from collections.abc import Iterable

    import httpx


class PhysioNetSource(HTTPSource):
    """A :class:`Source` for any PhysioNet dataset release.

    Inherits all HTTP machinery (client, retry, Range-resume download, checksum verify)
    from :class:`HTTPSource` — only :meth:`list_files` differs. Uses the
    ``SHA256SUMS.txt`` manifest that every PhysioNet release publishes as the
    authoritative file list: each line is ``<sha256>  <rel_path>``, and each entry's URL
    is just ``{base_url}/{rel_path}``. This eliminates the HTML-crawl (BeautifulSoup)
    pattern that every ETL's bespoke ``download.py`` used to need.

    Credential plumbing for restricted datasets (MIMIC-IV, eICU, etc.) is HTTP Basic auth
    via the ``username`` / ``password`` kwargs; open datasets (MIMIC-IV demo) need
    neither.

    Args:
        base_url: The PhysioNet release URL, with or without trailing slash — e.g.
            ``"https://physionet.org/files/mimiciv/3.1/"``.
        username: PhysioNet username (Basic auth). Omit for open-access datasets.
        password: PhysioNet password. Omit for open-access datasets.
        client: Optional injected :class:`httpx.Client` (used by tests). When omitted,
            one is built via :meth:`HTTPSource._make_client` with the supplied auth.
        unarchive: Blanket unpack mode applied to every :class:`RemoteFile` this source
            lists. Typically ``"auto"`` — members whose ``rel_path`` ends in ``.zip`` /
            ``.tar.gz`` / ``.tgz`` / ``.tar`` get unpacked after fetch; everything else
            (``.csv.gz``, ``.txt``, ...) is a no-op. The motivating case is HIRID, which
            ships its raw data as ``raw_stage/*.tar.gz`` members inside the PhysioNet
            release. ``None`` (default) preserves the "write archive as-is" behavior used
            by MIMIC-IV / eICU / MIMIC-IV demo.
        cleanup_archive: Tri-state controlling per-file archive cleanup after a
            successful extraction. ``None`` (default) defers to the per-member ``unarchive``
            mode — see :class:`~MEDS_extract.download.source.RemoteFile`. Set ``True`` /
            ``False`` to force the choice for every listed member.
        headers, timeout, max_attempts, transport: Forwarded to :meth:`HTTPSource._make_client`
            when ``client`` is not provided. ``headers`` is rarely needed for PhysioNet —
            Basic auth covers the credentialed releases — but it's passed through for
            symmetry with :class:`HTTPSource`.

    Examples:
        Public releases (e.g. MIMIC-IV demo) need no auth — construction is eager but does
        no network I/O until :meth:`list_files` is called:

        >>> src = PhysioNetSource(base_url="https://physionet.org/files/mimic-iv-demo/2.2")
        >>> src._base_url
        'https://physionet.org/files/mimic-iv-demo/2.2/'

        Credentialed releases (MIMIC-IV, eICU, etc.) take ``username`` / ``password``:

        >>> src = PhysioNetSource(
        ...     base_url="https://physionet.org/files/mimiciv/3.1",
        ...     username="demo_user", password="demo_pw",
        ... )

        Half-credentials are rejected eagerly (better to fail at construction than on
        first Basic-auth request):

        >>> PhysioNetSource(base_url="https://physionet.org/x/1.0", username="u")
        Traceback (most recent call last):
            ...
        ValueError: PhysioNetSource: username and password must be supplied together ...

        ``unarchive`` / ``cleanup_archive`` propagate to every
        :class:`~MEDS_extract.download.source.RemoteFile` listed. ``"auto"`` is the
        expected value for HIRID-shaped releases where only some members are archives:

        >>> src = PhysioNetSource(
        ...     base_url="https://physionet.org/files/hirid/1.1.1",
        ...     unarchive="auto",
        ... )
        >>> src._unarchive, src._cleanup_archive
        ('auto', None)
    """

    def __init__(
        self,
        base_url: str,
        username: str | None = None,
        password: str | None = None,
        client: httpx.Client | None = None,
        unarchive: str | None = None,
        cleanup_archive: bool | None = None,
        headers: dict[str, str] | None = None,
        timeout: tuple[float, float] = (10.0, 60.0),
        max_attempts: int = 5,
        transport: httpx.BaseTransport | None = None,
    ):
        if (username is None) != (password is None):
            raise ValueError(
                f"{type(self).__name__}: username and password must be supplied together "
                f"(got username={username!r}, password={'<set>' if password else None!r}). "
                f"Omit both for open-access datasets (e.g. MIMIC-IV demo)."
            )
        self._base_url = base_url if base_url.endswith("/") else base_url + "/"
        self._unarchive = unarchive
        self._cleanup_archive = cleanup_archive
        auth = (username, password) if username is not None else None
        super().__init__(
            urls=None,
            client=client,
            auth=auth,
            headers=headers,
            timeout=timeout,
            max_attempts=max_attempts,
            transport=transport,
        )

    def list_files(self) -> Iterable[RemoteFile]:
        sums_url = self._base_url + "SHA256SUMS.txt"
        r = self._client.get(sums_url)
        r.raise_for_status()
        for entry in self._parse_sha256sums(r.text):
            yield RemoteFile(
                rel_path=entry["rel_path"],
                sha256=entry["sha256"],
                unarchive=self._unarchive,
                cleanup_archive=self._cleanup_archive,
                extra={"url": self._base_url + entry["rel_path"]},
            )

    @staticmethod
    def _parse_sha256sums(text: str) -> list[dict]:
        """Parse PhysioNet's ``SHA256SUMS.txt`` format.

        Lines look like:

        .. code-block:: text

            9c3a...f2  subdir/file.csv.gz
            abc1...23  README.md

        The separator is arbitrary whitespace (two spaces by convention on PhysioNet, but
        some manifests use tabs). Blank lines and comment lines (leading ``#``) are
        skipped.

        Examples:
            >>> text = (
            ...     "abc123  foo.csv\\n"
            ...     "def456  sub/bar.csv.gz\\n"
            ... )
            >>> PhysioNetSource._parse_sha256sums(text)
            [{'sha256': 'abc123', 'rel_path': 'foo.csv'}, {'sha256': 'def456', 'rel_path': 'sub/bar.csv.gz'}]

            Blank / comment lines are tolerated:

            >>> PhysioNetSource._parse_sha256sums("# header\\n\\nabc  x.csv\\n")
            [{'sha256': 'abc', 'rel_path': 'x.csv'}]

            Paths with spaces (rare on PhysioNet but legal) are preserved:

            >>> PhysioNetSource._parse_sha256sums("abc  folder with spaces/file.txt\\n")
            [{'sha256': 'abc', 'rel_path': 'folder with spaces/file.txt'}]

            Malformed lines raise:

            >>> PhysioNetSource._parse_sha256sums("no_separator_on_this_line\\n")
            Traceback (most recent call last):
                ...
            ValueError: Malformed SHA256SUMS line: 'no_separator_on_this_line'
        """
        out = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                raise ValueError(f"Malformed SHA256SUMS line: {raw!r}")
            sha, rel = parts
            out.append({"sha256": sha, "rel_path": rel})
        return out
