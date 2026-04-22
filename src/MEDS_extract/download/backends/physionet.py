"""PhysioNet :class:`Source` — ``SHA256SUMS.txt``-driven, no HTML crawl."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._http import _make_httpx_client, _resumable_download
from ..source import RemoteFile

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    import httpx


class PhysioNetSource:
    """A :class:`Source` for any PhysioNet dataset release.

    Uses the ``SHA256SUMS.txt`` manifest that every PhysioNet release publishes as the
    authoritative file list. Each line in that file is ``<sha256>  <rel_path>``, and each
    entry's URL is just ``{base_url}/{rel_path}``. This eliminates the HTML-crawl
    (BeautifulSoup) pattern that every ETL's bespoke ``download.py`` used to need.

    Credential plumbing for restricted datasets (MIMIC-IV, eICU, etc.) is HTTP Basic auth
    via the ``username`` / ``password`` kwargs; open datasets (MIMIC-IV demo) need
    neither.

    Args:
        base_url: The PhysioNet release URL, with or without trailing slash — e.g.
            ``"https://physionet.org/files/mimiciv/3.1/"``.
        username: PhysioNet username (Basic auth). Omit for open-access datasets.
        password: PhysioNet password. Omit for open-access datasets.
        client: Optional injected :class:`httpx.Client` (used by tests). When omitted,
            one is built internally via :func:`MEDS_extract.download._http._make_httpx_client`
            with the supplied auth.

    Examples:
        Construction is eager — we normalize the base URL and stash the auth, but don't
        hit the network until :meth:`list_files` is called:

        >>> src = PhysioNetSource(
        ...     base_url="https://physionet.org/files/mimic-iv-demo/2.2",
        ...     username="demo_user", password="demo_pw",
        ... )
        >>> src._base_url
        'https://physionet.org/files/mimic-iv-demo/2.2/'
    """

    def __init__(
        self,
        base_url: str,
        username: str | None = None,
        password: str | None = None,
        client: httpx.Client | None = None,
    ):
        self._base_url = base_url if base_url.endswith("/") else base_url + "/"
        auth = (username, password) if username else None
        self._client = client or _make_httpx_client(auth=auth)

    def list_files(self) -> Iterable[RemoteFile]:
        sums_url = self._base_url + "SHA256SUMS.txt"
        r = self._client.get(sums_url)
        r.raise_for_status()
        for entry in _parse_sha256sums(r.text):
            yield RemoteFile(
                rel_path=entry["rel_path"],
                sha256=entry["sha256"],
                extra={"url": self._base_url + entry["rel_path"]},
            )

    def fetch(self, remote: RemoteFile, dest: Path) -> None:
        _resumable_download(self._client, remote.extra["url"], dest, remote.sha256)


def _parse_sha256sums(text: str) -> list[dict]:
    """Parse PhysioNet's ``SHA256SUMS.txt`` format.

    Lines look like:

    .. code-block:: text

        9c3a...f2  subdir/file.csv.gz
        abc1...23  README.md

    The separator is arbitrary whitespace (two spaces by convention on PhysioNet, but
    some manifests use tabs). Blank lines and comment lines (leading ``#``) are skipped.

    Examples:
        >>> text = (
        ...     "abc123  foo.csv\\n"
        ...     "def456  sub/bar.csv.gz\\n"
        ... )
        >>> _parse_sha256sums(text)
        [{'sha256': 'abc123', 'rel_path': 'foo.csv'}, {'sha256': 'def456', 'rel_path': 'sub/bar.csv.gz'}]

        Blank / comment lines are tolerated:

        >>> _parse_sha256sums("# header\\n\\nabc  x.csv\\n")
        [{'sha256': 'abc', 'rel_path': 'x.csv'}]

        Paths with spaces (rare on PhysioNet but legal) are preserved:

        >>> _parse_sha256sums("abc  folder with spaces/file.txt\\n")
        [{'sha256': 'abc', 'rel_path': 'folder with spaces/file.txt'}]

        Malformed lines raise:

        >>> _parse_sha256sums("no_separator_on_this_line\\n")
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
