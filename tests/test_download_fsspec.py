"""Extras-free download tests: the ``meds-extract-download`` CLI and ``FsspecSource``.

Everything here runs without the ``download`` extra (no ``httpx``/``tenacity``
import anywhere in the chain), so the "core (no extras)" CI job exercises the
fsspec-only download path end-to-end — exactly the path a user on a base install
takes when re-running against a pre-downloaded local mirror. HTTP-backed tests
(MockTransport wire behavior, retry, Range-resume) live in ``test_download.py``,
which the no-extras environment skips.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def test_meds_extract_download_cli_end_to_end(tmp_path: Path):
    """End-to-end ``meds-extract-download`` CLI against a local ``fsspec`` source.

    Runs the ``meds-extract-download`` console entry point as a subprocess, mimicking
    how a downstream ETL would shell out to populate ``raw_input_dir`` before handing
    off to the MEDS_extract stage pipeline. Uses a local directory as the source so the
    test needs no network and still exercises the full Hydra → ``sources_from_spec`` →
    ``Source.download_all`` → ``FsspecSource._pull`` path.
    """
    import subprocess

    # 1. Build a local "release" directory that stands in for a PhysioNet/cloud mirror.
    source_dir = tmp_path / "upstream_mirror"
    source_dir.mkdir()
    (source_dir / "patients.csv").write_text("patient_id,dob\n1,2000-01-01\n2,1990-05-05\n")
    (source_dir / "labs").mkdir()
    (source_dir / "labs" / "vitals.csv").write_text("pid,time,hr\n1,2024-01-01 08:00,82\n")

    # 2. Write a MESSY-style spec with a ``sources:`` block (the CLI only reads that
    # block; the rest of a real MESSY file is irrelevant to the download stage).
    spec_fp = tmp_path / "event_configs.yaml"
    spec_fp.write_text(
        f"""
sources:
  dataset:
    - type: fsspec
      root: {source_dir}
"""
    )

    # 3. Invoke the CLI binary as a subprocess, resolving ``spec`` and ``raw_input_dir``
    # through Hydra's dotlist override syntax — exactly how users will run it.
    raw_input_dir = tmp_path / "raw"
    result = subprocess.run(
        [
            "meds-extract-download",
            f"spec={spec_fp}",
            f"raw_input_dir={raw_input_dir}",
            "hydra.run.dir=" + str(tmp_path / ".hydra"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"CLI failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    # 4. Every file from the upstream mirror landed under ``raw_input_dir`` at its
    # expected relative path. No ``.part`` files remain.
    assert (raw_input_dir / "patients.csv").read_text().startswith("patient_id,dob")
    assert (raw_input_dir / "labs" / "vitals.csv").read_text().startswith("pid,time,hr")
    assert not any(raw_input_dir.rglob("*.part"))

    # 5. Re-running with ``do_overwrite=false`` (default) is idempotent — a same-size,
    # same-content file on disk is taken as already-complete and skipped. We verify by
    # re-running the CLI and confirming the file mtime didn't change (skipped, not
    # rewritten).
    patients_fp = raw_input_dir / "patients.csv"
    mtime_before = patients_fp.stat().st_mtime
    subprocess.run(
        [
            "meds-extract-download",
            f"spec={spec_fp}",
            f"raw_input_dir={raw_input_dir}",
            "hydra.run.dir=" + str(tmp_path / ".hydra2"),
        ],
        check=True,
        capture_output=True,
    )
    assert patients_fp.stat().st_mtime == mtime_before, "skipped file should not be rewritten"

    # 6. ``do_overwrite=true`` forces a re-fetch even when the on-disk content matches —
    # mtime changes and any stale local modification is overwritten by the upstream copy.
    patients_fp.write_text("local_edits_that_should_be_blown_away" + "X" * 100)
    subprocess.run(
        [
            "meds-extract-download",
            f"spec={spec_fp}",
            f"raw_input_dir={raw_input_dir}",
            "do_overwrite=true",
            "hydra.run.dir=" + str(tmp_path / ".hydra3"),
        ],
        check=True,
        capture_output=True,
    )
    assert patients_fp.read_text().startswith("patient_id,dob"), "do_overwrite should re-fetch"


def test_cli_only_resolves_sources_subtree(tmp_path: Path):
    """Regression for the symmetric OmegaConf-resolution problem on the CLI side.

    ``MessyConfig.parse`` strips ``sources`` before ``resolve=True`` so the pipeline
    doesn't need download-only env vars set. This is the mirror: the download CLI must
    resolve ONLY the ``sources:`` subtree, so an unrelated ``${oc.env:...}``
    interpolation in the event-conversion section of the combined MESSY file does not
    break ``meds-extract-download``.

    We verify end-to-end with a real console-script subprocess: combined MESSY with
    event-conversion ``${oc.env:UNRELATED_UNSET}`` that is never set anywhere. If the
    CLI were still resolving the whole file, this would fail with an
    ``InterpolationResolutionError``.
    """
    import os
    import subprocess

    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "hello.csv").write_text("a,b\n1,2\n")

    spec_fp = tmp_path / "messy.yaml"
    spec_fp.write_text(
        f"""
sources:
  dataset:
    - type: fsspec
      root: {mirror}

# event-conversion side references an env var that is never set — the CLI must not
# try to resolve this when loading sources.
_defaults:
  subject_id: $patient_id

patients:
  dob:
    code: DOB
    time: ${{oc.env:UNRELATED_UNSET}}
"""
    )

    raw_input_dir = tmp_path / "raw"
    env = {k: v for k, v in os.environ.items() if k != "UNRELATED_UNSET"}
    result = subprocess.run(
        [
            "meds-extract-download",
            f"spec={spec_fp}",
            f"raw_input_dir={raw_input_dir}",
            "hydra.run.dir=" + str(tmp_path / ".hydra"),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        "CLI failed; likely resolved the whole MESSY file instead of only the "
        f"``sources:`` subtree.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert (raw_input_dir / "hello.csv").read_text().startswith("a,b")


def _run_cli(tmp_path: Path, spec_body: str, *args: str, hydra_dir: str = ".hydra"):
    """Run the ``meds-extract-download`` console script against an inline spec."""
    import subprocess

    spec_fp = tmp_path / "spec.yaml"
    spec_fp.write_text(spec_body)
    raw_input_dir = tmp_path / "raw"
    return (
        subprocess.run(
            [
                "meds-extract-download",
                f"spec={spec_fp}",
                f"raw_input_dir={raw_input_dir}",
                "hydra.run.dir=" + str(tmp_path / hydra_dir),
                *args,
            ],
            capture_output=True,
            text=True,
            check=False,
        ),
        raw_input_dir,
    )


def test_cli_failure_exits_nonzero(tmp_path: Path):
    """Regression for the Hydra-discards-return-value bug: a failed download must
    exit non-zero (the fix is an explicit ``sys.exit(1)``), so scripted callers
    (``meds-extract-download && next-step``) stop on failure."""
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "a.csv").write_text("upstream content\n")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.csv").write_text("conflicting local content\n")  # sha mismatch → FileExistsError

    result, _ = _run_cli(tmp_path, f"sources:\n  dataset:\n    - type: fsspec\n      root: {mirror}\n")
    assert result.returncode != 0, f"expected failure exit:\n{result.stdout}\n{result.stderr}"
    # Pin the failure to the intended cause so an unrelated CLI crash can't
    # satisfy this test vacuously.
    assert "Refusing to overwrite" in result.stdout + result.stderr
    assert (raw / "a.csv").read_text() == "conflicting local content\n"  # untouched


def test_cli_unknown_key_exits_nonzero(tmp_path: Path):
    """A typo'd ``key=`` must be an error listing the available buckets — not a silent no-op success
    (``common`` is always appended, so a bad key would otherwise quietly fetch the wrong subset)."""
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "a.csv").write_text("x\n")

    result, raw = _run_cli(
        tmp_path,
        f"sources:\n  dataset:\n    - type: fsspec\n      root: {mirror}\n",
        "key=dtaaset",
    )
    assert result.returncode != 0
    assert "dtaaset" in result.stdout + result.stderr
    assert "dataset" in result.stdout + result.stderr  # available buckets listed
    assert not raw.exists()  # nothing was staged


def test_cli_no_sources_block_warns_and_exits_zero(tmp_path: Path):
    """A spec with no ``sources:`` block at all is a legitimately download-free ETL:

    the CLI must warn and exit 0 (so ``meds-extract-download && next-step`` chains
    keep working), not hard-error like a typo'd ``key=`` does.
    """
    result, raw = _run_cli(tmp_path, "patients:\n  dob:\n    code: DOB\n")
    assert result.returncode == 0, f"expected exit 0:\n{result.stdout}\n{result.stderr}"
    assert "Nothing to do" in result.stdout + result.stderr
    assert not raw.exists()


def test_cli_key_selects_bucket_and_appends_common(tmp_path: Path):
    """``key=demo`` must pull the ``demo`` bucket plus the always-appended ``common`` bucket — and nothing
    from the default ``dataset`` bucket."""
    m_ds = tmp_path / "m_ds"
    m_demo = tmp_path / "m_demo"
    m_common = tmp_path / "m_common"
    for m, fname in [(m_ds, "ds.csv"), (m_demo, "demo.csv"), (m_common, "shared.csv")]:
        m.mkdir()
        (m / fname).write_text(f"from {m.name}\n")

    spec = f"""sources:
  dataset:
    - type: fsspec
      root: {m_ds}
  demo:
    - type: fsspec
      root: {m_demo}
  common:
    - type: fsspec
      root: {m_common}
"""
    result, raw = _run_cli(tmp_path, spec, "key=demo")
    assert result.returncode == 0, f"CLI failed:\n{result.stdout}\n{result.stderr}"
    assert (raw / "demo.csv").exists(), "selected bucket must be fetched"
    assert (raw / "shared.csv").exists(), "common bucket must always be appended"
    assert not (raw / "ds.csv").exists(), "unselected default bucket must not be fetched"


def test_cli_cross_source_collision_exits_before_any_fetch(tmp_path: Path):
    """Two sources listing the same rel_path into one shared raw_input_dir is a config error caught up-front —
    not a mid-download race/FileExistsError."""
    m1 = tmp_path / "m1"
    m2 = tmp_path / "m2"
    for m in (m1, m2):
        m.mkdir()
        (m / "x.csv").write_text(f"from {m.name}\n")

    result, raw = _run_cli(
        tmp_path,
        f"""sources:
  dataset:
    - type: fsspec
      root: {m1}
    - type: fsspec
      root: {m2}
""",
    )
    assert result.returncode != 0
    assert "Duplicate destination across sources" in result.stdout + result.stderr
    assert not raw.exists() or not any(raw.iterdir())  # failed before any fetch


def test_cli_fail_fast_skips_remaining_sources(tmp_path: Path):
    """With the default ``continue_on_error=false``, a failing source stops the whole run — later sources are
    not attempted.

    With ``continue_on_error=true``, they are.
    """
    m1 = tmp_path / "m1"
    m2 = tmp_path / "m2"
    m1.mkdir()
    m2.mkdir()
    (m1 / "a.csv").write_text("upstream a\n")
    (m2 / "b.csv").write_text("upstream b\n")
    spec = f"""sources:
  dataset:
    - type: fsspec
      root: {m1}
    - type: fsspec
      root: {m2}
"""
    # Sabotage source 1: a conflicting pre-existing dest for a.csv.
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.csv").write_text("conflicting\n")

    result, _ = _run_cli(tmp_path, spec)
    assert result.returncode != 0
    assert "Refusing to overwrite" in result.stdout + result.stderr  # the intended failure
    assert not (raw / "b.csv").exists(), "fail-fast must not proceed to source 2"

    result, _ = _run_cli(tmp_path, spec, "continue_on_error=true", hydra_dir=".hydra2")
    assert result.returncode != 0  # source 1 still failed
    assert (raw / "b.csv").read_text() == "upstream b\n"  # but source 2 was attempted


@pytest.mark.skipif(sys.platform == "win32", reason="symlink semantics differ on Windows")
def test_fetch_refuses_symlink_escape(tmp_path: Path):
    """``_resolve_dest`` is the runtime security boundary: a rel_path that is clean at the string level
    (``evil/x.txt``) but whose first component is a symlink pointing outside ``dest_dir`` must be rejected at
    fetch time, with nothing written outside."""
    from MEDS_extract.download import RemoteFile, Source

    outside = tmp_path / "outside"
    outside.mkdir()
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "evil").symlink_to(outside)

    class EvilSource(Source):
        def _list_files(self):
            return [RemoteFile("evil/x.txt", "")]

        def _pull(self, source_path, target):
            target.write_text("escaped!")

    with pytest.raises(ValueError, match="escapes dest_dir"):
        EvilSource().download_all(dest)
    assert list(outside.iterdir()) == [], "nothing may be written outside dest_dir"


def test_fsspec_include_filters_before_hashing(tmp_path: Path, monkeypatch):
    """``include=`` on FsspecSource must filter *before* hashing — the documented cost mitigation for cloud
    mirrors is that excluded bytes are never read.

    (Source.files would re-filter anyway, so only a hash recorder can catch this regressing.)
    """
    import MEDS_extract.download.backends.fsspec as fs_mod

    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "keep.csv").write_text("kept\n")
    (mirror / "drop.csv").write_text("dropped\n")

    hashed: list[str] = []
    real_sha256_of = fs_mod.sha256_of

    def recording_sha256_of(p):
        hashed.append(p.name)
        return real_sha256_of(p)

    monkeypatch.setattr(fs_mod, "sha256_of", recording_sha256_of)

    dst = tmp_path / "dst"
    fs_mod.FsspecSource(root=str(mirror), include=["keep*"]).download_all(dst)

    assert hashed == ["keep.csv"], "excluded files must never be read/hashed"
    assert (dst / "keep.csv").exists()
    assert not (dst / "drop.csv").exists()


def test_pooled_dispatch_completes_manifests_larger_than_window(tmp_path: Path, monkeypatch):
    """The bounded sliding-window submission in ``Source._attempts`` must drain manifests
    larger than the window: every item beyond the initial batch is submitted as prior
    futures complete, and all of them finish."""
    from concurrent.futures import ThreadPoolExecutor

    import MEDS_extract.download.source as source_mod
    from MEDS_extract.download import RemoteFile, Source

    monkeypatch.setattr(source_mod, "_MAX_PENDING_SUBMITS", 4)
    n = 25  # > 6x the patched window

    class ManySource(Source):
        def _list_files(self):
            return [RemoteFile(f"f{i:03d}.txt", "") for i in range(n)]

        def _pull(self, source_path, target):
            target.write_text("ok")

    with ThreadPoolExecutor(max_workers=3) as pool:
        ManySource().download_all(tmp_path, pool=pool)

    assert len(list(tmp_path.glob("f*.txt"))) == n


def test_fsspec_source_memory_protocol(tmp_path: Path):
    """FsspecSource against a non-local protocol (in-process ``memory://``): locks the ``source_path=str(p)``
    → ``UPath(source_path)`` round-trip that every remote root depends on, without needing a network."""
    import uuid

    from upath import UPath

    from MEDS_extract.download import FsspecSource

    # The memory filesystem is process-global — use a unique root and clean it up
    # so nothing leaks into other tests (or later parametrizations) this session.
    root = UPath(f"memory://mirror-{uuid.uuid4().hex}")
    try:
        (root / "sub").mkdir(parents=True, exist_ok=True)
        (root / "patients.csv").write_bytes(b"patient_id\n1\n")
        (root / "sub" / "vitals.csv").write_bytes(b"pid,hr\n1,80\n")

        FsspecSource(root=str(root)).download_all(tmp_path)
        assert (tmp_path / "patients.csv").read_bytes() == b"patient_id\n1\n"
        assert (tmp_path / "sub" / "vitals.csv").read_bytes() == b"pid,hr\n1,80\n"
    finally:
        root.fs.rm(root.path, recursive=True)
