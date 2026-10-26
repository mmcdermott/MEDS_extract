"""Extras-free tests for post-fetch unarchive: ``archive_subdir`` and extraction completion markers.

Like ``test_download_fsspec.py``, nothing here imports ``httpx`` / ``tenacity``, so the
"core (no extras)" CI job runs it. Sources are stubs that override ``_list_files`` /
``_pull`` and count their pulls, which is what the re-run assertions are about:

- ``archive_subdir`` — selecting one folder of a wrapped bundle (the PhysioNet
  ``get-zip`` shape), flattened, for zip and tar.gz, with the zip-slip / tar-slip
  guards still applied to the rewritten paths;
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
from MEDS_extract.download.source import extraction_marker_path

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
    return sorted(
        p.name
        for p in root.rglob("*")
        if p.name.endswith((".extract-tmp", ".part")) or "extract-tmp" in p.name
    )


# --------------------------------------------------------------------------------------
# archive_subdir
# --------------------------------------------------------------------------------------


def test_archive_subdir_physionet_get_zip_layout(tmp_path: Path):
    """Only the selected folder lands, flattened; sibling folders and the wrapper do not."""
    body = _zip_bytes(OMOP_BUNDLE)
    pulls: list[str] = []
    src = BundleSource(
        body,
        pulls,
        rel_path="mimic-iv-demo-omop-0.9.zip",
        sha256=_sha(body),
        unarchive="auto",
        archive_subdir="1_omop_data_csv",
    )
    src.download_all(tmp_path)

    assert _files(tmp_path) == ["person.csv", "visit_occurrence.csv"]
    assert (tmp_path / "person.csv").read_text() == "person_id\n1\n"
    assert not (tmp_path / "mimic-iv-demo-omop-0.9").exists()
    assert not (tmp_path / "2_achilles_json").exists()
    assert not (tmp_path / "mimic-iv-demo-omop-0.9.zip").exists()  # auto's cleanup default
    assert _leftovers(tmp_path) == []

    marker = json.loads(extraction_marker_path(tmp_path / "mimic-iv-demo-omop-0.9.zip").read_text())
    assert marker == {
        "marker_version": 1,
        "archive": "mimic-iv-demo-omop-0.9.zip",
        "archive_sha256": _sha(body),
        "format": "zip",
        "archive_subdir": "1_omop_data_csv",
        "members": ["person.csv", "visit_occurrence.csv"],
    }


def test_archive_subdir_lands_next_to_archive_in_nested_rel_path(tmp_path: Path):
    """Members land in the archive's own directory, as without ``archive_subdir``."""
    body = _zip_bytes(OMOP_BUNDLE)
    src = BundleSource(
        body,
        [],
        rel_path="omop/raw.zip",
        sha256=_sha(body),
        unarchive="zip",
        archive_subdir="1_omop_data_csv",
    )
    src.download_all(tmp_path)

    assert _files(tmp_path) == ["omop/person.csv", "omop/raw.zip", "omop/visit_occurrence.csv"]


def test_archive_subdir_multi_component(tmp_path: Path):
    """``a/b`` matches as a run of whole components and keeps deeper structure below it."""
    body = _zip_bytes(
        {
            "wrap/a/b/x.csv": "x",
            "wrap/a/b/deep/y.csv": "y",
            "wrap/a/c/z.csv": "z",  # sibling of b: excluded
            "wrap/b/w.csv": "w",  # ``b`` alone, not under ``a``: excluded
        }
    )
    BundleSource(body, [], sha256=_sha(body), unarchive="zip", archive_subdir="a/b").download_all(tmp_path)

    assert _files(tmp_path) == ["bundle.zip", "deep/y.csv", "x.csv"]


def test_archive_subdir_at_archive_root(tmp_path: Path):
    """No wrapper directory is required — the subdir may sit at the archive root."""
    body = _zip_bytes({"data/x.csv": "x", "docs/readme.txt": "r"})
    BundleSource(body, [], sha256=_sha(body), unarchive="zip", archive_subdir="data").download_all(tmp_path)

    assert _files(tmp_path) == ["bundle.zip", "x.csv"]


def test_archive_subdir_zero_matches_raises_and_leaves_no_marker(tmp_path: Path):
    body = _zip_bytes(OMOP_BUNDLE)
    src = BundleSource(body, [], sha256=_sha(body), unarchive="auto", archive_subdir="1_omop_data_CSV")

    with pytest.raises(ValueError, match="matched no members"):
        src.download_all(tmp_path)

    # The verified archive stays (re-extraction source for a corrected config); nothing
    # else does — in particular no marker, so the failure can't later read as success.
    assert _files(tmp_path) == ["bundle.zip"]
    assert not extraction_marker_path(tmp_path / "bundle.zip").exists()
    assert _leftovers(tmp_path) == []


def test_archive_subdir_ambiguous_raises(tmp_path: Path):
    body = _zip_bytes({"a/data/x.csv": "x", "b/data/y.csv": "y"})
    src = BundleSource(body, [], sha256=_sha(body), unarchive="zip", archive_subdir="data")

    with pytest.raises(ValueError, match="ambiguous"):
        src.download_all(tmp_path)
    assert _files(tmp_path) == ["bundle.zip"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"archive_subdir": "data"}, "requires unarchive"),
        ({"unarchive": "zip", "archive_subdir": "../data"}, r"no '\.\.' components"),
        ({"unarchive": "zip", "archive_subdir": "data/../../x"}, r"no '\.\.' components"),
        ({"unarchive": "zip", "archive_subdir": "/data"}, "must be relative"),
        ({"unarchive": "zip", "archive_subdir": "./"}, "non-empty"),
        ({"unarchive": "zip", "archive_subdir": "a\\b"}, "backslashes"),
    ],
)
def test_archive_subdir_rejected_at_construction(kwargs, match):
    with pytest.raises(ValueError, match=match):
        RemoteFile("bundle.zip", "", **kwargs)


@pytest.mark.parametrize(
    "evil_member",
    [
        # Escapes once rewritten (and as-is): caught on the original name.
        "wrap/data/../../../escaped.csv",
        # Outside the selected folder, but still hostile: rejects the whole archive.
        "../escaped.csv",
        "/abs/escaped.csv",
    ],
)
def test_archive_subdir_keeps_zip_slip_guards(tmp_path: Path, evil_member: str):
    body = _zip_bytes({"wrap/data/ok.csv": "ok", evil_member: "pwned"})
    dest = tmp_path / "dest"
    src = BundleSource(body, [], sha256=_sha(body), unarchive="zip", archive_subdir="data")

    with pytest.raises(ValueError, match="Refusing to extract unsafe zip member"):
        src.download_all(dest)

    assert not (tmp_path / "escaped.csv").exists()
    assert not (dest / "ok.csv").exists()
    assert not extraction_marker_path(dest / "bundle.zip").exists()
    assert _leftovers(dest) == []


def test_archive_subdir_tar_gz(tmp_path: Path):
    """tar.gz: selected members (including an in-tree symlink and hard link) land flattened."""
    sym = tarfile.TarInfo("wrap/data/latest.csv")
    sym.type = tarfile.SYMTYPE
    sym.linkname = "sub/v2.csv"  # relative to the link's own directory: moves with the subtree
    hard = tarfile.TarInfo("wrap/data/copy.csv")
    hard.type = tarfile.LNKTYPE
    hard.linkname = "wrap/data/sub/v2.csv"  # archive-root relative: must be rewritten
    body = _tar_gz_bytes(
        [
            ("wrap/data/sub/v2.csv", b"v2"),
            ("wrap/data/top.csv", b"top"),
            ("wrap/other/skip.csv", b"skip"),
            sym,
            hard,
        ]
    )
    src = BundleSource(
        body, [], rel_path="bundle.tar.gz", sha256=_sha(body), unarchive="auto", archive_subdir="data"
    )
    src.download_all(tmp_path)

    assert _files(tmp_path) == ["copy.csv", "latest.csv", "sub/v2.csv", "top.csv"]
    assert (tmp_path / "copy.csv").read_bytes() == b"v2"
    if sys.platform != "win32":
        assert (tmp_path / "latest.csv").is_symlink()
        assert (tmp_path / "latest.csv").read_bytes() == b"v2"
    assert not (tmp_path / "bundle.tar.gz").exists()
    marker = json.loads(extraction_marker_path(tmp_path / "bundle.tar.gz").read_text())
    assert marker["format"] == "tar.gz"
    assert marker["members"] == ["copy.csv", "latest.csv", "sub/v2.csv", "top.csv"]


def _link(name: str, linkname: str, type_: bytes) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = type_
    info.linkname = linkname
    return info


@pytest.mark.parametrize(
    ("member", "match"),
    [
        # A symlink that escapes once its directory is flattened away.
        (_link("wrap/data/up.csv", "../../../../etc/passwd", tarfile.SYMTYPE), "Refusing to extract"),
        # A symlink that stays inside the archive's own layout but leaves the target
        # once its directory is flattened away (``x.csv -> ../other/s.csv``).
        (_link("wrap/data/x.csv", "../other/s.csv", tarfile.SYMTYPE), "Refusing to extract"),
        # A hard link whose target is outside the selected folder (never extracted).
        (_link("wrap/data/h.csv", "wrap/other/s.csv", tarfile.LNKTYPE), "outside archive_subdir"),
    ],
)
def test_archive_subdir_keeps_tar_link_guards(tmp_path: Path, member: tarfile.TarInfo, match: str):
    body = _tar_gz_bytes([("wrap/data/ok.csv", b"ok"), ("wrap/other/s.csv", b"s"), member])
    dest = tmp_path / "dest"
    src = BundleSource(
        body, [], rel_path="bundle.tar.gz", sha256=_sha(body), unarchive="tar.gz", archive_subdir="data"
    )

    with pytest.raises(ValueError, match=match):
        src.download_all(dest)

    assert not (dest / "ok.csv").exists()
    assert not extraction_marker_path(dest / "bundle.tar.gz").exists()
    assert _leftovers(dest) == []


# --------------------------------------------------------------------------------------
# Completion markers: #211 and #212
# --------------------------------------------------------------------------------------


def test_interrupted_extraction_is_repaired_on_rerun_issue_211(tmp_path: Path, monkeypatch):
    """Run 1 dies after extracting one member; run 2 must re-extract (not skip) and not re-download."""
    body = _zip_bytes({"data/a.csv": "a", "data/b.csv": "b", "data/c.csv": "c"})
    pulls: list[str] = []

    real_extract = zipfile.ZipFile.extract
    calls = {"n": 0}

    def flaky_extract(self, member, path=None, pwd=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(28, "No space left on device")
        return real_extract(self, member, path, pwd)

    with monkeypatch.context() as m:
        m.setattr(zipfile.ZipFile, "extract", flaky_extract)
        with pytest.raises(OSError, match="No space left"):
            BundleSource(body, pulls, sha256=_sha(body), unarchive="zip").download_all(tmp_path)

    # Run 1 left the verified archive but nothing that claims completion, and no
    # partial tree: the one member that did extract stayed in the scratch dir.
    assert _files(tmp_path) == ["bundle.zip"]
    assert not extraction_marker_path(tmp_path / "bundle.zip").exists()
    assert _leftovers(tmp_path) == []

    BundleSource(body, pulls, sha256=_sha(body), unarchive="zip").download_all(tmp_path)

    assert _files(tmp_path) == ["bundle.zip", "data/a.csv", "data/b.csv", "data/c.csv"]
    assert extraction_marker_path(tmp_path / "bundle.zip").exists()
    assert pulls == ["bundle"]  # repaired from the local archive, no re-download
    assert _leftovers(tmp_path) == []


def test_interrupted_extraction_with_cleanup_is_repaired_issue_211(tmp_path: Path, monkeypatch):
    """Same as above under ``"auto"``: the archive is only removed after a successful extraction."""
    body = _zip_bytes({"data/a.csv": "a", "data/b.csv": "b"})
    pulls: list[str] = []
    real_extract = zipfile.ZipFile.extract

    def failing_extract(self, member, path=None, pwd=None):
        raise KeyboardInterrupt  # Ctrl-C mid-unzip: not an ``Exception``, still cleaned up

    with monkeypatch.context() as m:
        m.setattr(zipfile.ZipFile, "extract", failing_extract)
        with pytest.raises(KeyboardInterrupt):
            BundleSource(body, pulls, sha256=_sha(body), unarchive="auto").download_all(tmp_path)
    assert zipfile.ZipFile.extract is real_extract
    assert _files(tmp_path) == ["bundle.zip"]
    assert _leftovers(tmp_path) == []

    BundleSource(body, pulls, sha256=_sha(body), unarchive="auto").download_all(tmp_path)
    assert _files(tmp_path) == ["data/a.csv", "data/b.csv"]
    assert pulls == ["bundle"]


def test_stale_scratch_dir_from_killed_run_is_removed(tmp_path: Path):
    """A SIGKILL skips every ``finally``; the next extraction clears the orphaned scratch dir."""
    body = _zip_bytes({"a.csv": "a"})
    scratch = tmp_path / ".bundle.zip.extract-tmp"
    (scratch / "junk").mkdir(parents=True)
    (scratch / "junk" / "half.csv").write_text("half")

    BundleSource(body, [], sha256=_sha(body), unarchive="zip").download_all(tmp_path)

    assert not scratch.exists()
    assert _files(tmp_path) == ["a.csv", "bundle.zip"]


def test_cleanup_archive_rerun_skips_without_refetch_issue_212(tmp_path: Path):
    """With ``"auto"`` the archive is deleted, but a fresh run must not re-download or re-extract."""
    body = _zip_bytes(OMOP_BUNDLE)
    pulls: list[str] = []

    def make():
        return BundleSource(
            body, pulls, sha256=_sha(body), unarchive="auto", archive_subdir="1_omop_data_csv"
        )

    make().download_all(tmp_path)
    assert pulls == ["bundle"]
    assert not (tmp_path / "bundle.zip").exists()
    (tmp_path / "person.csv").write_text("edited locally")  # proves no silent re-extraction

    make().download_all(tmp_path)

    assert pulls == ["bundle"]  # zero pulls on the second run
    assert _files(tmp_path) == ["person.csv", "visit_occurrence.csv"]
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
    """Explicit formats keep the archive; a re-run with archive + marker intact is a pure skip."""
    body = _zip_bytes({"a.csv": "a"})
    pulls: list[str] = []
    BundleSource(body, pulls, sha256=_sha(body), unarchive="zip").download_all(tmp_path)
    (tmp_path / "a.csv").write_text("edited locally")

    BundleSource(body, pulls, sha256=_sha(body), unarchive="zip").download_all(tmp_path)

    assert pulls == ["bundle"]
    assert (tmp_path / "a.csv").read_text() == "edited locally"


def test_kept_archive_missing_is_refetched_even_with_marker(tmp_path: Path):
    """When the archive is meant to be kept it is part of the result, so its loss forces a re-fetch."""
    body = _zip_bytes({"a.csv": "a"})
    pulls: list[str] = []
    BundleSource(body, pulls, sha256=_sha(body), unarchive="zip").download_all(tmp_path)
    (tmp_path / "bundle.zip").unlink()

    BundleSource(body, pulls, sha256=_sha(body), unarchive="zip").download_all(tmp_path)

    assert pulls == ["bundle", "bundle"]
    assert _files(tmp_path) == ["a.csv", "bundle.zip"]


def test_deleted_member_invalidates_marker(tmp_path: Path):
    """The marker vouches for listed members only while they all still exist."""
    body = _zip_bytes({"a.csv": "a", "b.csv": "b"})
    pulls: list[str] = []
    BundleSource(body, pulls, sha256=_sha(body), unarchive="auto").download_all(tmp_path)
    (tmp_path / "b.csv").unlink()

    BundleSource(body, pulls, sha256=_sha(body), unarchive="auto").download_all(tmp_path)

    assert pulls == ["bundle", "bundle"]  # archive was cleaned up, so re-fetch is the only way
    assert _files(tmp_path) == ["a.csv", "b.csv"]


def test_changed_archive_subdir_triggers_reextract(tmp_path: Path):
    """The marker records ``archive_subdir``; changing it re-extracts from the kept archive."""
    body = _zip_bytes({"w/a/x.csv": "x", "w/b/y.csv": "y"})
    pulls: list[str] = []
    BundleSource(body, pulls, sha256=_sha(body), unarchive="zip", archive_subdir="a").download_all(tmp_path)
    assert _files(tmp_path) == ["bundle.zip", "x.csv"]

    BundleSource(body, pulls, sha256=_sha(body), unarchive="zip", archive_subdir="b").download_all(tmp_path)

    assert pulls == ["bundle"]  # re-extracted from the local archive
    assert "y.csv" in _files(tmp_path)
    marker = json.loads(extraction_marker_path(tmp_path / "bundle.zip").read_text())
    assert marker["archive_subdir"] == "b"
    assert marker["members"] == ["y.csv"]


def test_changed_archive_subdir_with_cleanup_refetches(tmp_path: Path):
    """Same, but the archive was cleaned up: the mismatched marker forces a re-fetch."""
    body = _zip_bytes({"w/a/x.csv": "x", "w/b/y.csv": "y"})
    pulls: list[str] = []
    BundleSource(body, pulls, sha256=_sha(body), unarchive="auto", archive_subdir="a").download_all(tmp_path)
    BundleSource(body, pulls, sha256=_sha(body), unarchive="auto", archive_subdir="b").download_all(tmp_path)

    assert pulls == ["bundle", "bundle"]
    assert "y.csv" in _files(tmp_path)


def test_new_archive_version_triggers_refetch_and_reextract(tmp_path: Path):
    """A marker recording a different archive sha is not a skip witness."""
    v1 = _zip_bytes({"a.csv": "v1"})
    v2 = _zip_bytes({"a.csv": "v2"})
    pulls: list[str] = []
    BundleSource(v1, pulls, sha256=_sha(v1), unarchive="auto").download_all(tmp_path)
    BundleSource(v2, pulls, sha256=_sha(v2), unarchive="auto").download_all(tmp_path)

    assert pulls == ["bundle", "bundle"]
    assert (tmp_path / "a.csv").read_text() == "v2"


def test_do_overwrite_refetches_and_reextracts(tmp_path: Path):
    body = _zip_bytes({"a.csv": "a"})
    pulls: list[str] = []
    BundleSource(body, pulls, sha256=_sha(body), unarchive="auto").download_all(tmp_path)
    (tmp_path / "a.csv").write_text("edited locally")

    BundleSource(body, pulls, sha256=_sha(body), unarchive="auto").download_all(tmp_path, do_overwrite=True)

    assert pulls == ["bundle", "bundle"]
    assert (tmp_path / "a.csv").read_text() == "a"
    assert extraction_marker_path(tmp_path / "bundle.zip").exists()


def test_no_sha_archive_is_refetched_and_reextracted_every_run(tmp_path: Path):
    """Without a manifest sha nothing is provably complete — unchanged behavior, marker or not."""
    body = _zip_bytes({"a.csv": "a"})
    pulls: list[str] = []
    BundleSource(body, pulls, unarchive="auto").download_all(tmp_path)
    marker = json.loads(extraction_marker_path(tmp_path / "bundle.zip").read_text())
    assert marker["archive_sha256"] is None

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
