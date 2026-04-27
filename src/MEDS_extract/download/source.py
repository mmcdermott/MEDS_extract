"""The :class:`Source` ABC and its companions :class:`RemoteFile` / :class:`ChecksumError`.

A :class:`Source` is anywhere raw data comes from — a PhysioNet dataset release, an
explicit list of HTTP URLs, an S3 / GCS / local-filesystem tree. Concrete sources
inherit from this ABC and implement two private hooks:

- :meth:`Source._list_files` — enumerate what files the source offers.
- :meth:`Source._fetch` — move one file's bytes from the source to a local path.

The single public fetch entry point is :meth:`Source.download_all`, which delegates
to the source's injected :class:`~MEDS_extract.download.fetcher.Fetcher` for
per-file dispatch, skip-checks, sha verification, and post-fetch unarchive.

There is intentionally no per-file public fetch method. Every consumer we have
downloads the whole bundle, and per-file granularity invites the "wrong source x
wrong manifest" mismatched-pair footgun. See
https://github.com/mmcdermott/MEDS_extract/pull/96 for the design discussion.

:class:`RemoteFile` is the manifest-row type that ``_list_files`` returns. It is
*not* a user-facing input — users never construct one and never pass one back to
the source. It exists so :class:`Fetcher` and the per-backend ``_fetch`` agree on
a manifest entry shape (``rel_path`` + ``size`` + ``sha256`` + optional
``unarchive`` / ``cleanup_archive`` + transport-specific ``extra``). Stub sources
in tests construct it; that's the only legitimate construction site outside the
backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

# Re-exported (legacy import path: ``from MEDS_extract.download.source import RemoteFile``).
# These are runtime imports on purpose so users that ``from .source import RemoteFile``
# get the real class, not a forward-ref string. Hence the noqa for ruff's TC* / F401:
from ._types import ChecksumError, RemoteFile, sha256_of  # noqa: F401, TC001
from .fetcher import Fetcher, FetchReport

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class Source(ABC):
    """A place raw data comes from.

    Subclasses implement two private hooks:

    - :meth:`_list_files` — enumerate what files the source offers, as
      :class:`RemoteFile` rows.
    - :meth:`_fetch` — move one file's bytes from the source to a local path.

    Users never call those directly; the only public fetch entry point is
    :meth:`download_all`, which delegates to the injected
    :class:`~MEDS_extract.download.fetcher.Fetcher` for orchestration. One
    :class:`Fetcher` instance can (and typically does) drive multiple
    :class:`Source` instances — that's why it lives off the source as a
    constructor arg rather than as a method on the source itself.

    Invariants implementations must uphold:

    - :meth:`_list_files` is idempotent across calls. Re-enumerating must produce
      the same set of :class:`RemoteFile` rows (in the same order when possible).
    - :meth:`_fetch` writes to ``dest.with_name(dest.name + ".part")`` then
      atomic-renames into place, so a partial write never leaves a corrupt or
      truncated final ``dest`` on disk. The staged ``.part`` file MAY remain after
      a transport failure — that is intentional, since it enables range-resume on
      a subsequent attempt. The :class:`Fetcher` clears stale ``.part`` / ``dest``
      before invoking ``_fetch`` when a refetch is needed.
    - :meth:`_fetch` honors ``remote.sha256`` when set: verify after write, raise
      on mismatch, delete the ``.part``.
    - :meth:`_fetch` raises on transport errors rather than completing the rename
      into ``dest``.
    - When :meth:`_fetch` is called by the base ``download_all``, ``dest`` does
      not exist — the fetcher's overwrite/skip logic has already run.

    Examples:
        End-to-end ``download_all`` on a stub source. The stub overrides
        :meth:`_list_files` and :meth:`_fetch`; everything else (concurrency,
        skip-check, dest resolution, optional unarchive) is handled by the
        :class:`Fetcher` injected at construction.

        >>> from MEDS_extract.download import Fetcher
        >>> class StubSource(Source):
        ...     def _list_files(self):
        ...         return [RemoteFile("a.txt"), RemoteFile("sub/b.txt")]
        ...     def _fetch(self, remote, dest):
        ...         dest.write_text(f"contents of {remote.rel_path}")
        >>>
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     src = StubSource(fetcher=Fetcher(max_concurrency=1))
        ...     report = src.download_all(d)
        ...     files = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
        >>> report.n_downloaded, report.n_skipped, report.n_failed
        (2, 0, 0)
        >>> files
        ['a.txt', 'sub/b.txt']

        A :class:`Fetcher` is reusable across sources. Build one with the policy
        the run wants and hand it to every source:

        >>> shared = Fetcher(max_concurrency=4, continue_on_error=True)
        >>> src_a = StubSource(fetcher=shared)
        >>> src_b = StubSource(fetcher=shared)
        >>> src_a._fetcher is src_b._fetcher
        True

        If no fetcher is supplied, a default one is constructed:

        >>> StubSource()._fetcher.max_concurrency
        4
    """

    def __init__(self, *, fetcher: Fetcher | None = None):
        self._fetcher = fetcher or Fetcher()

    def download_all(self, dest_dir: Path) -> FetchReport:
        """Download every file this source lists into ``dest_dir``.

        Concurrency, skip-on-existing, SHA-256 verification, and post-fetch
        unarchive are all governed by the :class:`Fetcher` passed at construction.
        Returns a :class:`~MEDS_extract.download.fetcher.FetchReport` with
        per-file outcomes.
        """
        return self._fetcher._drive(list(self._list_files()), dest_dir, self._fetch)

    @abstractmethod
    def _list_files(self) -> Iterable[RemoteFile]:
        """Enumerate the files this source offers.

        Implementations MAY stream via a generator for large manifests. Callers
        should not assume the result is re-iterable — :meth:`download_all`
        materializes into a ``list`` once.
        """

    @abstractmethod
    def _fetch(self, remote: RemoteFile, dest: Path) -> None:
        """Transport-specific fetch implementation.

        See :class:`Source` invariants. Called by the injected :class:`Fetcher`,
        not directly by users.
        """

    def close(self) -> None:  # noqa: B027 — intentional no-op default; subclasses override when needed
        """Release transport resources held by this source.

        Default is a no-op. Subclasses that own network clients / file handles / connection pools override
        this. Safe to call multiple times; safe to call on sources that own nothing.
        """

    def __enter__(self) -> Source:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
