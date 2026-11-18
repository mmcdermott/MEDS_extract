"""Extras-free tests for post-fetch unarchive: ``archive_subdir`` and extraction completion markers.

Like ``test_download_fsspec.py``, nothing here imports ``httpx`` / ``tenacity``, so the
"core (no extras)" CI job runs it. Sources are stubs that override ``_list_files`` /
``_pull`` and count their pulls, which is what the re-run assertions are about:

- ``archive_subdir`` — keeping one folder of a wrapped bundle (the PhysioNet ``get-zip``
  shape);
- issue #211 — an extraction that dies part-way must be repaired by the next run
  (from the verified local archive, without re-downloading), never skipped;
- issue #212 — with archive cleanup (the ``"auto"`` default) a re-run must not
  re-download, because the completion marker, not the deleted archive, is the
  skip witness.

The URL-entry plumbing (``HTTPSource._normalize`` accepting ``archive_subdir``) and
``RemoteFile`` / ``PhysioNetSource`` validation are covered by doctests.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import zipfile
from typing import TYPE_CHECKING

import pytest

from MEDS_extract.download import RemoteFile, Source
from MEDS_extract.download import source as source_module
from MEDS_extract.download.source import extraction_marker_path
from MEDS_extract.download.unarchive import safe_extract

if TYPE_CHECKING:
    from pathlib import Path

# The MIMIC-IV OMOP demo ``get-zip`` layout, in miniature: tables under a versioned
# wrapper plus sibling folders the ETL never reads.
OMOP_BUNDLE = {
    "mimic-iv-demo-omop-0.9/1_omop_data_csv/person.csv": "person_id\n1\n",
    "mimic-iv-demo-omop-0.9/1_omop_data_csv/visit_occurrence.csv": "visit_occurrence_id\n2\n",
    "mimic-iv-demo-omop-0.9/2_achilles_json/report.json": "{}",
    "mimic-iv-demo-omop-0.9/3_data_quality_dashboard_files/index.html": "<html/>",
    "mimic-iv-demo-omop-0.9/LICENSE.txt": "license",
}


def _zip_bytes(members: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in members.items():
            zf.writestr(name, text)
    return buf.getvalue()


def _tar_gz_bytes(members: list[tarfile.TarInfo | tuple[str, bytes]]) -> bytes:
    """Build a tar.gz from ``(name, data)`` regular files and/or pre-built link ``TarInfo``s."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for m in members:
            if isinstance(m, tarfile.TarInfo):
                tf.addfile(m)
            else:
                name, data = m
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


class BundleSource(Source):
    """One-archive stub source; ``pulls`` counts transfers across instances sharing the list."""

    def __init__(self, body: bytes, pulls: list[str], rel_path: str = "bundle.zip", **remote_kwargs):
        super().__init__()
        self._body = body
        self._pulls = pulls
        self._rel_path = rel_path
        self._remote_kwargs = remote_kwargs

    def _list_files(self):
        return [RemoteFile(self._rel_path, "bundle", **self._remote_kwargs)]

    def _pull(self, source_path, target):
        self._pulls.append(source_path)
        target.write_bytes(self._body)


def _files(root: Path) -> list[str]:
    """Visible (non-dot) files under ``root``, as sorted POSIX rel paths."""
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )


def _leftovers(root: Path) -> list[str]:
    """Scratch dirs / staging files a clean run must not leave behind."""
    return sorted(p.name for p in root.rglob("*") if p.name.endswith((".extract-tmp", ".part")))


def _fail_after_extracting(exc: BaseException):
    """A ``safe_extract`` that does its work and then dies, as on ENOSPC or Ctrl-C."""

    def fake(archive_path, target_dir, fmt):
        safe_extract(archive_path, target_dir, fmt)
        raise exc

    return fake


# --------------------------------------------------------------------------------------
# archive_subdir
# --------------------------------------------------------------------------------------

SUBDIR = "mimic-iv-demo-omop-0.9/1_omop_data_csv"


def test_archive_subdir_physionet_get_zip_layout(tmp_path: Path):
    """Only the selected folder's contents land; sibling folders and the wrapper do not."""
    body = _zip_bytes(OMOP_BUNDLE)
    src = BundleSource(body, [], sha256=_sha(body), unarchive="auto", archive_subdir=SUBDIR)
    src.download_all(tmp_path)

    assert _files(tmp_path) == ["person.csv", "visit_occurrence.csv"]
    assert (tmp_path / "person.csv").read_text() == "person_id\n1\n"
    assert _leftovers(tmp_path) == []
    marker = json.loads(extraction_marker_path(tmp_path / "bundle.zip").read_text())
    assert marker == {"archive_sha256": _sha(body), "format": "zip", "archive_subdir": SUBDIR}


def test_archive_subdir_missing_raises_and_leaves_no_marker(tmp_path: Path):
    body = _zip_bytes(OMOP_BUNDLE)
    src = BundleSource(body, [], sha256=_sha(body), unarchive="auto", archive_subdir="1_omop_data_csv")

    with pytest.raises(ValueError, match="is not a folder of bundle.zip"):
        src.download_all(tmp_path)

    # The verified archive stays, to re-extract from once the config is fixed; no marker.
    assert _files(tmp_path) == ["bundle.zip"]
    assert not extraction_marker_path(tmp_path / "bundle.zip").exists()
    assert _leftovers(tmp_path) == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"archive_subdir": "data"}, "requires unarchive"),
        ({"unarchive": "zip", "archive_subdir": "../data"}, "folder inside the archive"),
        ({"unarchive": "zip", "archive_subdir": "/data"}, "folder inside the archive"),
        ({"unarchive": "zip", "archive_subdir": "./"}, "folder inside the archive"),
        ({"unarchive": "zip", "archive_subdir": "a\\b"}, "folder inside the archive"),
    ],
)
def test_archive_subdir_rejected_at_construction(kwargs, match):
    with pytest.raises(ValueError, match=match):
        RemoteFile("bundle.zip", "", **kwargs)


@pytest.mark.skipif(sys.platform == "win32", reason="symlink semantics differ on Windows")
def test_archive_subdir_moves_symlinks_as_links(tmp_path: Path):
    """A tar symlink within the kept folder is moved as a link and still resolves."""
    link = tarfile.TarInfo("w/data/latest.csv")
    link.type, link.linkname = tarfile.SYMTYPE, "sub/v2.csv"
    body = _tar_gz_bytes([("w/data/sub/v2.csv", b"v2"), ("w/other/s.csv", b"s"), link])
    BundleSource(
        body, [], rel_path="b.tar.gz", sha256=_sha(body), unarchive="tar.gz", archive_subdir="w/data"
    ).download_all(tmp_path)

    assert (tmp_path / "latest.csv").is_symlink()
    assert (tmp_path / "latest.csv").read_bytes() == b"v2"
    assert not (tmp_path / "s.csv").exists()


# --------------------------------------------------------------------------------------
# Completion markers: #211 and #212
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("unarchive", "exc"), [("zip", OSError(28, "No space left on device")), ("auto", KeyboardInterrupt())]
)
def test_interrupted_extraction_is_repaired_on_rerun_issue_211(tmp_path: Path, monkeypatch, unarchive, exc):
    """Run 1 dies mid-extraction; run 2 re-extracts from the local archive (no re-download), never skips.

    Under ``"auto"`` the archive is only removed after a successful extraction.
    """
    body = _zip_bytes({"data/a.csv": "a", "data/b.csv": "b"})
    pulls: list[str] = []

    with monkeypatch.context() as m:
        m.setattr(source_module, "safe_extract", _fail_after_extracting(exc))
        with pytest.raises(type(exc)):
            BundleSource(body, pulls, sha256=_sha(body), unarchive=unarchive).download_all(tmp_path)

    assert _files(tmp_path) == ["bundle.zip"]  # no marker, no partial tree
    assert _leftovers(tmp_path) == []

    BundleSource(body, pulls, sha256=_sha(body), unarchive=unarchive).download_all(tmp_path)

    expected = ["data/a.csv", "data/b.csv"]
    assert _files(tmp_path) == (expected if unarchive == "auto" else ["bundle.zip", *expected])
    assert pulls == ["bundle"]


def test_stale_scratch_dir_from_killed_run_is_removed(tmp_path: Path):
    """A SIGKILL skips every ``finally``; the next extraction clears the orphaned scratch dir."""
    body = _zip_bytes({"a.csv": "a"})
    scratch = tmp_path / ".bundle.zip.extract-tmp"
    (scratch / "junk").mkdir(parents=True)

    BundleSource(body, [], sha256=_sha(body), unarchive="zip").download_all(tmp_path)

    assert not scratch.exists()
    assert _files(tmp_path) == ["a.csv", "bundle.zip"]


def test_cleanup_archive_rerun_skips_without_refetch_issue_212(tmp_path: Path):
    """With ``"auto"`` the archive is deleted, but a fresh run must not re-download or re-extract."""
    body = _zip_bytes(OMOP_BUNDLE)
    pulls: list[str] = []
    BundleSource(body, pulls, sha256=_sha(body), unarchive="auto", archive_subdir=SUBDIR).download_all(
        tmp_path
    )
    (tmp_path / "person.csv").write_text("edited locally")  # proves no silent re-extraction

    BundleSource(body, pulls, sha256=_sha(body), unarchive="auto", archive_subdir=SUBDIR).download_all(
        tmp_path
    )

    assert pulls == ["bundle"]
    assert (tmp_path / "person.csv").read_text() == "edited locally"


def test_cleanup_archive_finishes_interrupted_cleanup(tmp_path: Path):
    """Marker written but the run died before removing the archive: the re-run just removes it."""
    body = _zip_bytes({"a.csv": "a"})
    pulls: list[str] = []
    BundleSource(body, pulls, sha256=_sha(body), unarchive="auto").download_all(tmp_path)
    (tmp_path / "bundle.zip").write_bytes(body)  # as if the unlink never happened

    BundleSource(body, pulls, sha256=_sha(body), unarchive="auto").download_all(tmp_path)

    assert pulls == ["bundle"]
    assert _files(tmp_path) == ["a.csv"]


def test_kept_archive_rerun_skips(tmp_path: Path):
    """Explicit formats keep the archive; a re-run with archive and marker intact is a pure skip."""
    body = _zip_bytes({"a.csv": "a"})
    pulls: list[str] = []
    BundleSource(body, pulls, sha256=_sha(body), unarchive="zip").download_all(tmp_path)
    (tmp_path / "a.csv").write_text("edited locally")

    BundleSource(body, pulls, sha256=_sha(body), unarchive="zip").download_all(tmp_path)

    assert pulls == ["bundle"]
    assert (tmp_path / "a.csv").read_text() == "edited locally"


def test_changed_archive_subdir_triggers_reextract(tmp_path: Path):
    """The marker records ``archive_subdir``; changing it re-extracts from the kept archive."""
    body = _zip_bytes({"w/a/x.csv": "x", "w/b/y.csv": "y"})
    pulls: list[str] = []
    BundleSource(body, pulls, sha256=_sha(body), unarchive="zip", archive_subdir="w/a").download_all(tmp_path)

    BundleSource(body, pulls, sha256=_sha(body), unarchive="zip", archive_subdir="w/b").download_all(tmp_path)

    assert pulls == ["bundle"]
    assert _files(tmp_path) == ["bundle.zip", "x.csv", "y.csv"]  # earlier files are not deleted


def test_do_overwrite_refetches_and_reextracts(tmp_path: Path):
    body = _zip_bytes({"a.csv": "a"})
    pulls: list[str] = []
    BundleSource(body, pulls, sha256=_sha(body), unarchive="auto").download_all(tmp_path)
    (tmp_path / "a.csv").write_text("edited locally")

    BundleSource(body, pulls, sha256=_sha(body), unarchive="auto").download_all(tmp_path, do_overwrite=True)

    assert pulls == ["bundle", "bundle"]
    assert (tmp_path / "a.csv").read_text() == "a"


def test_no_sha_archive_is_refetched_and_reextracted_every_run(tmp_path: Path):
    """Without a manifest sha nothing is provably complete: unchanged behaviour."""
    body = _zip_bytes({"a.csv": "a"})
    pulls: list[str] = []
    BundleSource(body, pulls, unarchive="auto").download_all(tmp_path)
    BundleSource(body, pulls, unarchive="auto").download_all(tmp_path)

    assert pulls == ["bundle", "bundle"]
    assert _files(tmp_path) == ["a.csv"]


@pytest.mark.skipif(sys.platform == "win32", reason="symlink semantics differ on Windows")
def test_move_into_place_refuses_symlink_escape(tmp_path: Path):
    """A pre-existing symlinked directory in dest must not redirect extracted members outside it."""
    outside = tmp_path / "outside"
    outside.mkdir()
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "tables").symlink_to(outside)
    body = _zip_bytes({"tables/a.csv": "a"})

    with pytest.raises(ValueError, match="outside the destination"):
        BundleSource(body, [], sha256=_sha(body), unarchive="zip").download_all(dest)

    assert list(outside.iterdir()) == []
    assert not extraction_marker_path(dest / "bundle.zip").exists()
