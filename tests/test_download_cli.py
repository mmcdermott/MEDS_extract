"""Generic ``meds-extract-download`` CLI behavior tests (extras-free).

Every test here drives the real console script as a subprocess — exactly how users
invoke it — and asserts CLI-level contracts: exit codes, bucket-``key=`` semantics,
selective interpolation resolution, spec resolution (``pkg://``), reserved
``sources:`` keys, and overwrite/fail-fast policy. Local ``fsspec`` mirrors are used
as the transport vehicle purely because they need no network and no extras;
fsspec-*source*-specific behavior lives in ``test_download_fsspec.py``, and
HTTP-backed wire behavior in ``test_download.py`` (skipped without the ``download``
extra).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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


def _run_cli(
    tmp_path: Path,
    spec_body: str,
    *args: str,
    hydra_dir: str = ".hydra",
    env: dict[str, str] | None = None,
):
    """Run the ``meds-extract-download`` console script against an inline spec.

    ``env``, when given, fully replaces the subprocess environment (pass a copy of
    ``os.environ`` plus/minus the vars under test).
    """
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
            env=env,
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


def test_cli_malformed_source_entry_logs_and_exits_nonzero(tmp_path: Path):
    """A malformed ``sources:`` entry (here an unknown ``type:``) is a user config error:

    the CLI must log it and exit non-zero via the same log-and-``sys.exit(1)`` path as
    manifest-validation failures — naming the spec file and surfacing the underlying
    spec error — rather than letting the raw exception propagate through Hydra.
    """
    result, raw = _run_cli(tmp_path, "sources:\n  dataset:\n    - type: s3\n      root: /x\n")
    assert result.returncode != 0, f"expected failure exit:\n{result.stdout}\n{result.stderr}"
    combined = result.stdout + result.stderr
    # Pin the failure to the intended cause: the CLI's own log line plus the
    # underlying spec error, so an unrelated crash can't satisfy this vacuously.
    assert "Could not construct sources" in combined
    assert "Unknown source type" in combined
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


def _issue_151_spec(tmp_path: Path) -> str:
    """The credentialed-``dataset`` + public-``demo`` (+ interpolated ``common``) spec shape.

    ``dataset`` needs an env var (``FIX151_UNSET_CRED``) the tests deliberately never
    set; ``demo`` is a plain local mirror; ``common`` resolves its root from an env var
    (``FIX151_COMMON_ROOT``) the tests DO set.
    """
    m_demo = tmp_path / "m_demo"
    m_common = tmp_path / "m_common"
    for m, fname in [(m_demo, "demo.csv"), (m_common, "shared.csv")]:
        m.mkdir(exist_ok=True)
        (m / fname).write_text(f"from {m.name}\n")
    return f"""sources:
  dataset:
    - type: fsspec
      root: ${{oc.env:FIX151_UNSET_CRED}}
  demo:
    - type: fsspec
      root: {m_demo}
  common:
    - type: fsspec
      root: ${{oc.env:FIX151_COMMON_ROOT}}
"""


def _issue_151_env(tmp_path: Path) -> dict[str, str]:
    """Subprocess env for the issue-#151-shaped tests: common root set, credential unset."""
    import os

    env = {k: v for k, v in os.environ.items() if k != "FIX151_UNSET_CRED"}
    env["FIX151_COMMON_ROOT"] = str(tmp_path / "m_common")
    return env


def test_cli_unselected_bucket_interpolations_not_resolved(tmp_path: Path):
    """``key=demo`` must succeed without the credentialed ``dataset`` bucket's env vars set.

    Demo users (and credential-free CI) must not need unrelated credentials in the
    environment to pull a bucket that doesn't use them: interpolations are resolved
    per selected bucket, not across all of ``sources:``. The narrowing covers ``key``
    plus the always-appended ``common`` bucket — ``common``'s interpolations ARE
    resolved (and its files staged) whichever key is selected. Both properties are
    asserted against one CLI invocation (formerly two identical-invocation tests).
    """
    result, raw = _run_cli(tmp_path, _issue_151_spec(tmp_path), "key=demo", env=_issue_151_env(tmp_path))
    assert result.returncode == 0, (
        "CLI failed; likely resolved unselected buckets' interpolations too.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert (raw / "demo.csv").read_text() == "from m_demo\n", "selected bucket must be staged"
    assert (raw / "shared.csv").read_text() == "from m_common\n", "common bucket must be staged"


def test_cli_selected_bucket_interpolation_failure_is_clear(tmp_path: Path):
    """Selecting the credentialed bucket WITHOUT its env var set must still fail, and the error must name the
    missing variable — narrowing resolution to the selected bucket may not swallow its own interpolation
    errors."""
    result, raw = _run_cli(tmp_path, _issue_151_spec(tmp_path), "key=dataset", env=_issue_151_env(tmp_path))
    assert result.returncode != 0
    assert "FIX151_UNSET_CRED" in result.stdout + result.stderr, "error must name the missing env var"
    assert not raw.exists(), "nothing may be staged on a failed resolve"


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


def test_cli_pkg_spec_resolution(tmp_path: Path):
    """``spec=pkg://...`` resolves a spec bundled inside an installed package.

    Builds a minimal importable package under ``tmp_path`` (made visible to the CLI
    subprocess via ``PYTHONPATH`` — the same import surface a pip-installed dataset
    package presents) whose only resource is a ``sources:``-bearing YAML, then runs
    the console script against ``pkg://<pkg>.<file>.yaml``. Mirrors
    ``MEDS_transform-pipeline``'s pkg:// syntax via MEDS-transforms'
    ``resolve_pkg_path``, so the two CLIs cannot drift.
    """
    import os
    import subprocess

    from yaml_to_disk import yaml_disk

    mirror = tmp_path / "mirror"
    yaml_disk(
        f"""
mirror:
  patients.csv: "patient_id,dob\\n1,2000-01-01\\n"
pkgs:
  fake_dl_pkg:
    __init__.py: ""
    spec.yaml: |
      sources:
        dataset:
          - type: fsspec
            root: {mirror}
""",
        root_dir=tmp_path,
    )

    raw_input_dir = tmp_path / "raw"
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "pkgs")}
    result = subprocess.run(
        [
            "meds-extract-download",
            "spec=pkg://fake_dl_pkg.spec.yaml",
            f"raw_input_dir={raw_input_dir}",
            "hydra.run.dir=" + str(tmp_path / ".hydra"),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, f"CLI failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert (raw_input_dir / "patients.csv").read_text().startswith("patient_id,dob")


def test_cli_sources_dataset_version_reserved_key(tmp_path: Path):
    """The reserved ``sources.dataset_version`` key is version metadata, not a bucket.

    Three properties pinned here: (1) a spec carrying it downloads normally — the key
    is never treated as a bucket; (2) it is interpolatable into bucket entries via
    document-relative interpolation (``${sources.dataset_version.<bucket>}``; the
    scalar form resolves identically); (3) selecting it (``key=dataset_version``) is
    an error naming the REAL buckets, exactly like any other not-a-bucket key.
    """
    import subprocess

    from yaml_to_disk import yaml_disk

    # Version strings are dot-free ("v31"/"v22") only because a dotted directory name
    # (mirror-3.1) would read as a file to yaml_to_disk; nothing here depends on the
    # version's spelling — interpolation is plain string composition.
    yaml_disk(
        f"""
mirror-v31:
  patients.csv: "patient_id,dob\\n1,2000-01-01\\n"
mirror-v22:
  demo.csv: "patient_id\\n1\\n"
spec.yaml: |
  sources:
    dataset_version:
      dataset: "v31"
      demo: "v22"
    dataset:
      - type: fsspec
        root: {tmp_path}/mirror-${{sources.dataset_version.dataset}}
    demo:
      - type: fsspec
        root: {tmp_path}/mirror-${{sources.dataset_version.demo}}
""",
        root_dir=tmp_path,
    )
    spec_fp = tmp_path / "spec.yaml"

    def _run(*args: str, hydra_dir: str):
        return subprocess.run(
            [
                "meds-extract-download",
                f"spec={spec_fp}",
                *args,
                "hydra.run.dir=" + str(tmp_path / hydra_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    # (1) + (2), mapping form: the selected bucket's root interpolates its own version.
    result = _run(f"raw_input_dir={tmp_path / 'raw'}", hydra_dir=".hydra")
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert (tmp_path / "raw" / "patients.csv").exists()

    result = _run(f"raw_input_dir={tmp_path / 'raw_demo'}", "key=demo", hydra_dir=".hydra2")
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert (tmp_path / "raw_demo" / "demo.csv").exists()

    # (3) key=dataset_version is not a bucket; the error lists only real buckets.
    result = _run(f"raw_input_dir={tmp_path / 'raw_bad'}", "key=dataset_version", hydra_dir=".hydra3")
    assert result.returncode == 1
    combined = result.stdout + result.stderr
    assert "does not name a sources bucket" in combined
    assert "['dataset', 'demo']" in combined
    assert not (tmp_path / "raw_bad").exists()
