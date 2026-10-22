"""Redivis-backed :class:`Source` — pull raw files from a Redivis dataset's file index.

`Redivis <https://redivis.com>`_ hosts datasets as an ``organization``/``user`` →
``dataset`` (optionally version) → ``table`` hierarchy. Raw (unstructured) files live in
a *file index* table; each row is one file addressed by a system ``file_id``. This
backend enumerates that index (:meth:`_list_files`) and streams each file to disk
(:meth:`_pull`) via the official ``redivis`` client.

The motivating dataset is `EHRSHOT <https://redivis.com/datasets/53gc-8rhx41kgt>`_
(Stanford Shah Lab), distributed on Redivis as zip bundles (``EHRSHOT_MEDS.zip`` etc.).
Access is credentialed (a signed data-use agreement), so an end-to-end pull requires a
``REDIVIS_API_TOKEN`` for an account that has been granted access — it cannot run in CI.
The design discussion and open questions live in issue #128.

Scaffold status: the enumeration + streaming *logic and wiring* are exercised
hermetically via an injected fake client (see the doctests). The exact live-API calls
that could not be verified against a DUA-gated account — notably dataset **version**
pinning — are marked with ``TODO(#128)`` and must be confirmed by a maintainer with
access before this backend is relied on against real Redivis data.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from ..source import RemoteFile, Source

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

logger = logging.getLogger(__name__)


class RedivisSource(Source):
    """A :class:`Source` for raw files hosted in a Redivis dataset's file index.

    Args:
        dataset: The Redivis dataset name or id (e.g. ``"53gc-8rhx41kgt"`` for EHRSHOT).
        organization: Owning organization slug. Exactly one of ``organization`` /
            ``user`` must be given.
        user: Owning user name. Exactly one of ``organization`` / ``user`` must be given.
        version: Optional dataset version to pin (``None`` = the current release).
        table: Name of the file-index table to enumerate. Required — Redivis surfaces
            raw files through a named index table.
        file_names: Optional allow-list of file names; when given, only files whose
            ``name`` is in this set are listed. Handy to pull just one bundle (e.g.
            ``["EHRSHOT_MEDS.zip"]``) out of a larger dataset.
        api_token: Optional Redivis API token. When omitted, the ``redivis`` client reads
            ``REDIVIS_API_TOKEN`` from the environment (the documented mechanism for
            unattended use). When given, it seeds that env var if unset.
        client: Optional injected client exposing the ``redivis`` module surface this
            backend uses (``organization`` / ``user`` / ``file``). Tests and doctests
            pass a fake here; in production it defaults to the real ``redivis`` module
            (imported lazily so the ``redivis`` dependency is only needed when this
            backend is actually used).

    Checksums: Redivis exposes an MD5 (base64) per file, but the orchestrator's
    skip/verify path is SHA-256 only, so every :class:`RemoteFile` here is emitted with
    ``sha256=None``. That means an existing local copy cannot be *verified* for
    skip-on-rerun (it shares the no-manifest-hash behavior tracked in #112). Reconciling
    MD5 with the SHA-256 verifier is design-call 1 of #128.

    Examples:
        Every network call goes through the injected ``client``, so the backend runs
        hermetically against a stand-in covering the surface it uses —
        ``organization``/``user`` → ``dataset`` → ``table`` → ``list_files``, and
        ``file(id).download(path)``:

        >>> class FakeFile:
        ...     def __init__(self, name, file_id, body):
        ...         self.name, self.file_id, self._body = name, file_id, body
        ...     def download(self, path, overwrite=False, progress=False):
        ...         Path(path).write_bytes(self._body)
        ...         return path
        >>> class FakeTable:
        ...     def __init__(self, files):
        ...         self._files = files
        ...     def list_files(self, *a, **k):
        ...         return list(self._files)
        >>> class FakeDataset:
        ...     def __init__(self, table):
        ...         self._table = table
        ...     def table(self, name):
        ...         return self._table
        >>> class FakeRedivis:  # mimics the ``redivis`` module surface we call
        ...     def __init__(self, files):
        ...         self._by_id = {f.file_id: f for f in files}
        ...         self._dataset = FakeDataset(FakeTable(files))
        ...     def organization(self, slug):
        ...         return self
        ...     def user(self, name):
        ...         return self
        ...     def dataset(self, name, version=None):
        ...         return self._dataset
        ...     def file(self, file_id):
        ...         return self._by_id[file_id]

        ``_list_files`` turns the file index into manifest rows (``sha256`` is ``None`` —
        see the checksum note above):

        >>> files = [
        ...     FakeFile("EHRSHOT_MEDS.zip", "fid_meds", b"meds-bytes"),
        ...     FakeFile("splits.csv", "fid_splits", b"split-bytes"),
        ... ]
        >>> src = RedivisSource(
        ...     dataset="53gc-8rhx41kgt", organization="stanford",
        ...     table="release_files", client=FakeRedivis(files),
        ... )
        >>> [(rf.rel_path, rf.source_path, rf.sha256) for rf in src.files]
        [('EHRSHOT_MEDS.zip', 'fid_meds', None), ('splits.csv', 'fid_splits', None)]

        ``download_all`` streams each file to disk — ``_pull`` writes into the ABC's
        ``.part`` target, which is then atomically renamed into place:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     src.download_all(d)
        ...     print_directory(d)
        ├── EHRSHOT_MEDS.zip
        └── splits.csv

        A ``file_names`` allow-list restricts the manifest to a subset — e.g. just the
        MEDS bundle:

        >>> src = RedivisSource(
        ...     dataset="53gc-8rhx41kgt", organization="stanford", table="release_files",
        ...     file_names=["EHRSHOT_MEDS.zip"], client=FakeRedivis(files),
        ... )
        >>> [rf.rel_path for rf in src.files]
        ['EHRSHOT_MEDS.zip']

        Construction validates the owner and target up front — exactly one of
        ``organization`` / ``user``:

        >>> RedivisSource(dataset="d", organization="o", user="u", table="t")
        Traceback (most recent call last):
            ...
        ValueError: RedivisSource: pass exactly one of 'organization' or 'user' (got organization='o', ...

        >>> RedivisSource(dataset="d", table="t")
        Traceback (most recent call last):
            ...
        ValueError: RedivisSource: pass exactly one of 'organization' or 'user' (got organization=None, ...

        A missing ``table`` is rejected too — Redivis lists raw files through a named
        index table:

        >>> RedivisSource(dataset="d", organization="o")
        Traceback (most recent call last):
            ...
        ValueError: RedivisSource: 'table' (the file-index table name) is required.
    """

    def __init__(
        self,
        dataset: str,
        organization: str | None = None,
        user: str | None = None,
        version: str | None = None,
        table: str | None = None,
        file_names: list[str] | None = None,
        api_token: str | None = None,
        client: Any | None = None,
    ):
        if (organization is None) == (user is None):
            raise ValueError(
                "RedivisSource: pass exactly one of 'organization' or 'user' "
                f"(got organization={organization!r}, user={user!r})."
            )
        if not table:
            raise ValueError("RedivisSource: 'table' (the file-index table name) is required.")
        self._dataset = dataset
        self._organization = organization
        self._user = user
        self._version = version
        self._table = table
        self._file_names = set(file_names) if file_names else None
        self._api_token = api_token
        self._client = client

    def _resolve_client(self) -> Any:
        """Return the injected client, or lazily import the real ``redivis`` module.

        The import is deferred to here (not module load) so the ``redivis`` dependency is
        only required when this backend actually runs — construction, ``files`` with an
        injected client, and the dispatcher stay import-free.
        """
        if self._client is not None:
            return self._client
        try:
            import redivis
        except ImportError as e:  # pragma: no cover - exercised only without the extra installed
            raise ImportError(
                "RedivisSource requires the 'redivis' package. Install it with "
                "`pip install 'MEDS_extract[redivis]'` (or `uv add redivis`)."
            ) from e
        # The redivis client authenticates via REDIVIS_API_TOKEN. Seed it from an
        # explicit api_token without clobbering an already-set environment token.
        if self._api_token is not None:
            os.environ.setdefault("REDIVIS_API_TOKEN", self._api_token)
        return redivis

    def _dataset_handle(self, client: Any) -> Any:
        """Resolve ``org|user → dataset`` (optionally version-pinned)."""
        owner = client.organization(self._organization) if self._organization else client.user(self._user)
        if self._version is not None:
            # TODO(#128): confirm the version-pinning call against the live redivis-python
            # API (``dataset(name, version=...)`` vs a ``.version(...)`` chain) once a
            # DUA-approved account is available; the fake client accepts this form.
            return owner.dataset(self._dataset, version=self._version)
        return owner.dataset(self._dataset)

    def _list_files(self) -> Iterable[RemoteFile]:
        client = self._resolve_client()
        table = self._dataset_handle(client).table(self._table)
        for f in table.list_files():
            if self._file_names is not None and f.name not in self._file_names:
                continue
            # sha256=None: Redivis exposes MD5 (base64), not SHA-256 — see the class
            # docstring's checksum note and #128 design-call 1.
            yield RemoteFile(rel_path=f.name, source_path=str(f.file_id), sha256=None)

    def _pull(self, source_path: str, target: Path) -> None:
        client = self._resolve_client()
        # ``source_path`` is the Redivis file_id captured in ``_list_files``. The base
        # class hands us the ``.part`` target and owns the atomic rename on success.
        client.file(source_path).download(str(target), overwrite=True)
