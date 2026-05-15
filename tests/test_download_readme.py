"""Doc-consistency check: the download README's file tree matches the on-disk layout.

This test deliberately does NOT import ``MEDS_extract.download`` — it only reads
the README and re-runs ``pretty_print_directory.print_directory`` on the
submodule directory. Living outside the download package means it runs in every
CI job (including ``core (no extras)`` where the download extras aren't
installed), so adding or removing a file in ``src/MEDS_extract/download/``
without updating the README's tree fails the test wherever pytest runs.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from pretty_print_directory import PrintConfig, print_directory

_DOWNLOAD_DIR = Path(__file__).resolve().parents[1] / "src" / "MEDS_extract" / "download"
_README = _DOWNLOAD_DIR / "README.md"


def test_download_readme_file_tree_matches_directory():
    buf = StringIO()
    print_directory(_DOWNLOAD_DIR, config=PrintConfig(file_extension=[".py", ".md"]), file=buf)
    expected_tree = buf.getvalue().rstrip()
    readme = _README.read_text(encoding="utf-8")

    assert expected_tree in readme, (
        f"\n{_README} is out of date — its file tree does not match the on-disk layout.\n\n"
        f"Expected tree (paste into the README's ```text fence):\n{expected_tree}\n"
    )
