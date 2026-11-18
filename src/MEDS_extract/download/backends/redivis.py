"""Redivis-backed :class:`Source` — pull raw files *or* tabular tables from a Redivis dataset.

`Redivis <https://redivis.com>`_ hosts datasets as an ``organization``/``user`` →
``dataset`` (optionally version) → ``table`` hierarchy. Tables come in two kinds, and
this backend handles both:

- **File-index tables** (``table:``) hold raw, unstructured files — one row per file,
  addressed by a system file id. Each file is streamed to disk as-is (e.g.
  ``EHRSHOT_MEDS.zip`` for `EHRSHOT <https://redivis.com/datasets/53gc-8rhx41kgt>`_).
- **Tabular tables** (``tables:``) hold rows — e.g. the OMOP CDM tables of
  `INSPECT EHR <https://redivis.com/datasets/dzc6-9jyt6gapt>`_. Each table is
  streamed through the Redivis read API into one ``<table>.parquet`` file, which is the
  bare-file layout MEDS_extract's input resolution (``io.resolve_source_files``) reads
  directly — so a MESSY spec can key its event configs on ``person``, ``measurement``…

Datasets are referenced by **name**, optionally with the reference id
(``"inspect_ehr:dzc6"``) — *not* by the persistent id in a dataset's web URL
(``dzc6-9jyt6gapt``), which the Redivis API rejects.

Auth: access is credentialed (a signed data-use agreement), so the backend requires a
Redivis **API token** for an account that has been granted access — either via
``api_token:`` (typically ``${oc.env:REDIVIS_API_TOKEN,null}`` in the spec; the
``,null`` default lets an unset variable reach this backend's own error instead of an
OmegaConf ``KeyError``) or the ``REDIVIS_API_TOKEN`` environment variable. Without one
it fails fast, while listing and before any fetch, rather than falling into the
``redivis`` client's interactive browser login mid-pipeline. Create a
token under *Settings → API tokens* on redivis.com; it needs data-read access to the
dataset.

Checksums: Redivis exposes an MD5 per raw file, and :meth:`_pull` checks each freshly
downloaded file-index file against it (the ``redivis`` client before 0.22 only compares
it with a file that already exists). Redivis has no SHA-256, so every :class:`RemoteFile`
here has ``sha256=None``.

Re-runs: after each pull, a hidden ``.<name>.redivis.json`` marker next to the file records
what was fetched (a file's id and MD5; or the dataset, version and table of an export)
and the SHA-256 of the bytes written. A re-run skips a file whose marker matches and whose
bytes still hash to that SHA-256 (see :meth:`_verifies`). Tabular exports are only reused
when ``version`` is pinned: a released Redivis version doesn't change, but the current one
can, so an unpinned export is pulled again every run. ``do_overwrite`` re-fetches all.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from ..source import RemoteFile, Source, sha256_of

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

logger = logging.getLogger(__name__)

# A dataset's web-URL persistent id (``dzc6-9jyt6gapt``) — not a valid API reference.
_URL_ID_RE = re.compile(r"^[0-9a-z]{4}-[0-9a-z]{6,}$")

# ``source_path`` prefix that routes a manifest row to the tabular-export pull path.
_TABLE_PREFIX = "table:"


def _marker_path(dest: Path) -> Path:
    """The reuse marker for ``dest``: a hidden ``.<name>.redivis.json`` sibling."""
    return dest.with_name(f".{dest.name}.redivis.json")


def _check_md5(path: Path, expected: bytes | str | None, file_id: str) -> None:
    """Delete ``path`` and raise unless its MD5 is ``expected`` (raw bytes, or base64 text).

    The ``redivis`` client stores a file's listed MD5 as raw bytes (``File.hash``); a file
    listed without one is not checked.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     fp = Path(d) / "a.zip"
        ...     _ = fp.write_bytes(b"data")
        ...     _check_md5(fp, hashlib.md5(b"data").digest(), "fid_1")
        ...     _check_md5(fp, None, "fid_1")
        ...     try:
        ...         _check_md5(fp, base64.b64encode(hashlib.md5(b"other").digest()).decode(), "fid_1")
        ...     except ValueError as e:
        ...         print(e)
        ...     fp.exists()
        RedivisSource: MD5 mismatch for file 'fid_1' (a.zip); the download is corrupt and was removed.
        False
    """
    if not expected:
        return
    if isinstance(expected, str):
        expected = base64.b64decode(expected)
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    if digest.digest() != expected:
        path.unlink()
        raise ValueError(
            f"RedivisSource: MD5 mismatch for file {file_id!r} ({path.name}); the download is corrupt "
            "and was removed."
        )


class RedivisSource(Source):
    """A :class:`Source` for a Redivis dataset's file-index files and/or tabular tables.

    Args:
        dataset: The Redivis dataset name, optionally with its reference id
            (e.g. ``"inspect_ehr:dzc6"``). The persistent id from a dataset's web URL
            (e.g. ``"dzc6-9jyt6gapt"``) is rejected with a pointer to the right form.
        organization: Owning organization slug. Exactly one of ``organization`` /
            ``user`` must be given.
        user: Owning user name. Exactly one of ``organization`` / ``user`` must be given.
        version: Optional dataset version to pin (e.g. ``"v1.2"``; ``None`` = current).
        table: Name of a file-index table whose raw files to download.
        file_names: Optional allow-list of file names within ``table``.
        tables: Tabular tables to export as ``<name>.parquet`` — a list of table names,
            or ``"*"`` for every tabular (non-file-index) table in the dataset. At least
            one of ``table`` / ``tables`` must be given; both may be.
        api_token: Redivis API token. Seeds ``REDIVIS_API_TOKEN`` when that is unset.
            Required unless ``REDIVIS_API_TOKEN`` is already set (or a ``client`` is
            injected).
        unarchive: Post-fetch unpack mode for file-index files (see
            :class:`RemoteFile`); e.g. ``"auto"`` to extract a ``.zip`` bundle.
        cleanup_archive: See :class:`RemoteFile`.
        include: Generic manifest filter globs (see :class:`Source`).
        exclude: Generic manifest filter globs (see :class:`Source`).
        client: Optional injected stand-in for the ``redivis`` module (tests/doctests);
            defaults to the real module, imported lazily.

    Examples:
        The fakes below mirror the attribute names of the real ``redivis`` 0.20+ client:
        ``File.id`` / ``File.path`` and ``Table.properties["isFileIndex"]``.

        >>> import pyarrow as pa, pyarrow.parquet as pq
        >>> class FakeFile:
        ...     def __init__(self, path, file_id, body):
        ...         self.path, self.id, self._body = Path(path), file_id, body
        ...         self.name = self.path.name
        ...     def download(self, path, overwrite=False, progress=None):
        ...         Path(path).write_bytes(self._body)
        ...         return path
        >>> class FakeTable:
        ...     def __init__(self, name, files=None, batches=None):
        ...         self.name, self._files, self._batches = name, files, batches
        ...         self.properties = {"name": name, "isFileIndex": files is not None}
        ...     def list_files(self, *a, **k):
        ...         return list(self._files)
        ...     def to_arrow_batch_iterator(self, *a, **k):
        ...         return iter(self._batches)
        >>> class FakeDataset:
        ...     def __init__(self, tables):
        ...         self._tables = {t.name: t for t in tables}
        ...     def table(self, name):
        ...         return self._tables[name]
        ...     def list_tables(self, *a, **k):
        ...         return list(self._tables.values())
        >>> class FakeRedivis:
        ...     def __init__(self, tables):
        ...         self._dataset = FakeDataset(tables)
        ...     def organization(self, slug):
        ...         return self
        ...     def dataset(self, name, version=None):
        ...         return self._dataset
        >>> person = pa.record_batch({"person_id": [1, 2], "year_of_birth": [1970, 1985]})
        >>> fake = FakeRedivis([
        ...     FakeTable("MEDS", files=[FakeFile("INSPECT_MEDS.zip", "fid_1", b"zip")]),
        ...     FakeTable("person", batches=[person, person]),
        ...     FakeTable("death", batches=[pa.record_batch({"person_id": [2]})]),
        ... ])

        ``tables="*"`` exports every tabular table (file indexes are skipped):

        >>> src = RedivisSource(
        ...     dataset="inspect_ehr:dzc6", organization="shahlab", tables="*", client=fake
        ... )
        >>> [(rf.rel_path, rf.source_path) for rf in src.files]
        [('death.parquet', 'table:death'), ('person.parquet', 'table:person')]
        >>> with tempfile.TemporaryDirectory() as d:
        ...     src.download_all(Path(d))
        ...     print_directory(Path(d))
        ...     pq.read_table(Path(d) / "person.parquet").num_rows
        ├── death.parquet
        └── person.parquet
        4

        ``table=`` downloads a file index's raw files; both modes can be combined:

        >>> src = RedivisSource(
        ...     dataset="inspect_ehr:dzc6", organization="shahlab",
        ...     table="MEDS", tables=["person"], client=fake,
        ... )
        >>> [rf.rel_path for rf in src.files]
        ['INSPECT_MEDS.zip', 'person.parquet']

        Construction validates its arguments up front:

        >>> RedivisSource(dataset="d", organization="o", user="u", table="t")
        Traceback (most recent call last):
            ...
        ValueError: RedivisSource: pass exactly one of 'organization' or 'user' (got organization='o', ...
        >>> RedivisSource(dataset="d", organization="o")
        Traceback (most recent call last):
            ...
        ValueError: RedivisSource: pass 'table' (a file-index table) and/or 'tables' (tabular tables).
        >>> RedivisSource(dataset="dzc6-9jyt6gapt", organization="o", tables="*")
        Traceback (most recent call last):
            ...
        ValueError: RedivisSource: 'dzc6-9jyt6gapt' looks like a dataset's web-URL id; the API needs the
        dataset name, optionally with its reference id (e.g. 'inspect_ehr:dzc6').
    """

    def __init__(
        self,
        dataset: str,
        organization: str | None = None,
        user: str | None = None,
        version: str | None = None,
        table: str | None = None,
        file_names: list[str] | None = None,
        tables: list[str] | str | None = None,
        api_token: str | None = None,
        unarchive: str | None = None,
        cleanup_archive: bool | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        client: Any | None = None,
    ):
        super().__init__(include=include, exclude=exclude)
        if (organization is None) == (user is None):
            raise ValueError(
                "RedivisSource: pass exactly one of 'organization' or 'user' "
                f"(got organization={organization!r}, user={user!r})."
            )
        if not table and not tables:
            raise ValueError(
                "RedivisSource: pass 'table' (a file-index table) and/or 'tables' (tabular tables)."
            )
        if isinstance(tables, str) and tables != "*":
            raise ValueError(f"RedivisSource: 'tables' must be a list of names or '*', got {tables!r}.")
        if _URL_ID_RE.match(dataset):
            raise ValueError(
                f"RedivisSource: {dataset!r} looks like a dataset's web-URL id; the API needs the "
                "dataset name, optionally with its reference id (e.g. 'inspect_ehr:dzc6')."
            )
        self._dataset = dataset
        self._organization = organization
        self._user = user
        self._version = version
        self._table = table
        self._file_names = set(file_names) if file_names else None
        self._tables = tables
        self._api_token = api_token
        self._unarchive = unarchive
        self._cleanup_archive = cleanup_archive
        self._client = client
        # file id -> client File handle, captured while listing. The module-level
        # ``redivis.file(id)`` lookup is removed in redivis-python (it raises
        # DeprecationError), so _pull downloads through the listed handle instead.
        self._file_handles: dict[str, Any] = {}
        # source_path -> (final file name, what was fetched, or None when it can't be pinned).
        self._fetched: dict[str, tuple[str, str | None]] = {}
        self._dataset_cache: Any | None = None

    def _resolve_client(self) -> Any:
        """Return the injected client, or lazily import the real ``redivis`` module.

        Raises:
            ValueError: if no API token is available — the real client would otherwise
                block on an interactive browser login.
        """
        if self._client is not None:
            return self._client
        if self._api_token:
            os.environ.setdefault("REDIVIS_API_TOKEN", self._api_token)
        if not os.environ.get("REDIVIS_API_TOKEN"):
            raise ValueError(
                "RedivisSource needs a Redivis API token: export REDIVIS_API_TOKEN in the shell "
                "running the download (or set `api_token:` on the source). Create one under "
                "Settings → API tokens on redivis.com for an account with access to the dataset."
            )
        try:
            import redivis
        except ImportError as e:  # pragma: no cover - exercised only without the extra installed
            raise ImportError(
                "RedivisSource requires the 'redivis' package. Install it with "
                "`pip install 'MEDS_extract[redivis]'` (or `uv add redivis`)."
            ) from e
        return redivis

    def _dataset_handle(self) -> Any:
        """Resolve ``org|user → dataset`` (optionally version-pinned), cached."""
        if self._dataset_cache is None:
            client = self._resolve_client()
            owner = client.organization(self._organization) if self._organization else client.user(self._user)
            # redivis-python: ``Organization.dataset(name, *, version=None)`` / same on User.
            if self._version is not None:
                self._dataset_cache = owner.dataset(self._dataset, version=self._version)
            else:
                self._dataset_cache = owner.dataset(self._dataset)
        return self._dataset_cache

    def _tabular_table_names(self) -> list[str]:
        if self._tables == "*":
            names = [
                t.properties.get("name", t.name)
                for t in self._dataset_handle().list_tables()
                if not t.properties.get("isFileIndex", False)
            ]
            return sorted(names)
        return list(self._tables or [])

    def _list_files(self) -> Iterable[RemoteFile]:
        # Resolve the client (and so check the token) up front: an explicit ``tables:``
        # list needs no API call to list, so a missing token would otherwise surface
        # only mid-fetch, once per file.
        self._resolve_client()
        if self._table:
            for f in self._dataset_handle().table(self._table).list_files():
                if self._file_names is not None and f.name not in self._file_names:
                    continue
                self._file_handles[str(f.id)] = f
                md5 = getattr(f, "hash", None)
                if isinstance(md5, bytes):
                    md5 = base64.b64encode(md5).decode()
                self._fetched[str(f.id)] = (f.path.name, f"file {f.id} md5={md5}" if md5 else None)
                yield RemoteFile(
                    rel_path=f.path.as_posix(),
                    source_path=str(f.id),
                    sha256=None,
                    unarchive=self._unarchive,
                    cleanup_archive=self._cleanup_archive,
                )
        owner = self._organization or self._user
        for name in self._tabular_table_names():
            pinned = f"table {owner}/{self._dataset}@{self._version}/{name}" if self._version else None
            self._fetched[f"{_TABLE_PREFIX}{name}"] = (f"{name}.parquet", pinned)
            yield RemoteFile(rel_path=f"{name}.parquet", source_path=f"{_TABLE_PREFIX}{name}", sha256=None)

    def _verifies(self, dest: Path, item: RemoteFile) -> bool:
        """True iff ``dest`` can be reused: its marker records this item, and its bytes are unchanged.

        Redivis rows have no ``sha256``, so the marker :meth:`_pull` writes stands in for one:
        it records what was fetched and the SHA-256 of the bytes written.
        """
        if item.sha256 is not None:
            return super()._verifies(dest, item)
        _, fetched = self._fetched.get(item.source_path, (None, None))
        if fetched is None or not dest.exists():
            return False
        try:
            marker = json.loads(_marker_path(dest).read_text())
        except (OSError, ValueError):
            return False
        return marker == {"fetched": fetched, "sha256": sha256_of(dest)}

    def _record_pull(self, source_path: str, target: Path) -> None:
        """Write (or clear) the reuse marker for the file ``target`` is about to become."""
        name, fetched = self._fetched[source_path]
        marker = _marker_path(target.with_name(PurePosixPath(name).name))
        if fetched is None:
            marker.unlink(missing_ok=True)
            return
        marker.write_text(json.dumps({"fetched": fetched, "sha256": sha256_of(target)}))

    def _pull(self, source_path: str, target: Path) -> None:
        if source_path.startswith(_TABLE_PREFIX):
            self._pull_table(source_path[len(_TABLE_PREFIX) :], target)
        else:
            handle = self._file_handles.get(source_path)
            if handle is None:
                raise KeyError(f"RedivisSource: file id {source_path!r} was not in the listed manifest.")
            handle.download(str(target), overwrite=True, progress=False)
            _check_md5(target, getattr(handle, "hash", None), source_path)
        self._record_pull(source_path, target)

    def _pull_table(self, name: str, target: Path) -> None:
        """Stream a tabular table's rows into one parquet file at ``target``."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = self._dataset_handle().table(name)
        writer = None
        n_rows = 0
        try:
            for batch in table.to_arrow_batch_iterator(progress=False):
                if writer is None:
                    schema = batch.schema
                    writer = pq.ParquetWriter(str(target), schema)
                if batch.schema != schema:
                    batch = pa.Table.from_batches([batch]).cast(schema)
                writer.write(batch)
                n_rows += batch.num_rows
            if writer is None:  # empty table: still write a parquet file with its schema
                empty = table.to_arrow_table(max_results=0, progress=False)
                pq.write_table(empty, str(target))
        finally:
            if writer is not None:
                writer.close()
        logger.info(f"Redivis table {name}: wrote {n_rows} rows to {target.name}")
