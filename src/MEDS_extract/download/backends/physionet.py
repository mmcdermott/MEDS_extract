"""PhysioNet :class:`Source` — ``SHA256SUMS.txt``-driven, no HTML crawl."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import quote

from ..source import RemoteFile
from .http import HTTPSource

if TYPE_CHECKING:
    from collections.abc import Iterable

    import httpx
    from tenacity.wait import wait_base


class PhysioNetSource(HTTPSource):
    """A :class:`Source` for any PhysioNet dataset release.

    Inherits all HTTP machinery (client, retry, Range-resume download, checksum verify)
    from :class:`HTTPSource` — it overrides :meth:`_list_files` (plus its constructor,
    which takes a release URL and credentials instead of an explicit URL list). Uses
    the ``SHA256SUMS.txt`` manifest that every PhysioNet release publishes as the
    authoritative file list: each line is ``<sha256>  <rel_path>``, and each entry's URL
    is just ``{base_url}/{rel_path}``.

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
        headers, timeout, max_attempts, transport, retry_wait: Forwarded to
            :meth:`HTTPSource._make_client` when ``client`` is not provided.
            ``headers`` is rarely needed for PhysioNet — Basic auth covers the
            credentialed releases — but it's passed through for symmetry with
            :class:`HTTPSource`.
        include, exclude: Optional :mod:`fnmatch` globs applied to the manifest —
            e.g. ``include=["hosp/*.csv.gz"]`` stages only the hospital tables from
            a release that also bundles data the ETL never reads. See
            :class:`~MEDS_extract.download.source.Source`.

    Examples:
        Public releases (e.g. MIMIC-IV demo) need no auth — construction is eager but does
        no network I/O until :meth:`Source.download_all` is called:

        >>> src = PhysioNetSource(base_url="https://physionet.org/files/mimic-iv-demo/2.2")
        >>> src._base_url
        'https://physionet.org/files/mimic-iv-demo/2.2/'
        >>> src.close()

        Credentialed releases (MIMIC-IV, eICU, etc.) take ``username`` / ``password``:

        >>> src = PhysioNetSource(
        ...     base_url="https://physionet.org/files/mimiciv/3.1",
        ...     username="demo_user", password="demo_pw",
        ... )
        >>> src.close()

        Half-credentials are rejected eagerly (better to fail at construction than on
        first Basic-auth request):

        >>> PhysioNetSource(base_url="https://physionet.org/x/1.0", username="u")
        Traceback (most recent call last):
            ...
        ValueError: PhysioNetSource: username and password must be supplied together ...
    """

    def __init__(
        self,
        base_url: str,
        username: str | None = None,
        password: str | None = None,
        client: httpx.Client | None = None,
        headers: dict[str, str] | None = None,
        timeout: tuple[float, float] = (10.0, 60.0),
        max_attempts: int = 5,
        transport: httpx.BaseTransport | None = None,
        retry_wait: wait_base | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ):
        if (username is None) != (password is None):
            raise ValueError(
                f"{type(self).__name__}: username and password must be supplied together "
                f"(got username={username!r}, password={'<set>' if password else None!r}). "
                f"Omit both for open-access datasets (e.g. MIMIC-IV demo)."
            )
        self._base_url = base_url if base_url.endswith("/") else base_url + "/"
        auth = (username, password) if username is not None else None
        super().__init__(
            urls=None,
            client=client,
            auth=auth,
            headers=headers,
            timeout=timeout,
            max_attempts=max_attempts,
            transport=transport,
            retry_wait=retry_wait,
            include=include,
            exclude=exclude,
        )

    def _list_files(self) -> Iterable[RemoteFile]:
        sums_url = self._base_url + "SHA256SUMS.txt"
        r = self._client.get(sums_url)
        r.raise_for_status()
        for entry in self._parse_sha256sums(r.text):
            yield RemoteFile(
                rel_path=entry["rel_path"],
                sha256=entry["sha256"],
                # Percent-encode the path segment: a rel_path containing ``#``,
                # ``?``, or ``%`` would otherwise be parsed as fragment / query /
                # existing-escape and silently request the wrong resource.
                source_path=self._base_url + quote(entry["rel_path"], safe="/"),
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
