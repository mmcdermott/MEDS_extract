"""Registered dataset imports retain actionable context and their root cause."""

from types import SimpleNamespace

import pytest

from MEDS_extract.config import MessyConfig


def register(monkeypatch, module):
    def forbidden_load():
        pytest.fail("The entry-point attribute must not be loaded")

    ep = SimpleNamespace(name="Synthetic-DS", value=f"{module}:events.yaml", dist=None, load=forbidden_load)
    monkeypatch.setattr("MEDS_extract.config.entry_points", lambda group: [ep])


@pytest.mark.parametrize("case", ["missing_package", "missing_submodule", "missing_dependency"])
def test_registry_import_error_names_registration_and_missing_module(tmp_path, monkeypatch, case):
    package_name = f"synthetic_registry_{case}"
    package = tmp_path / package_name
    package.mkdir()
    (package / "__init__.py").write_text(
        "import synthetic_absent_dependency\n" if case == "missing_dependency" else ""
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    module = {
        "missing_package": "synthetic_absent_package",
        "missing_submodule": f"{package_name}.missing",
        "missing_dependency": package_name,
    }[case]
    missing = "synthetic_absent_dependency" if case == "missing_dependency" else module
    register(monkeypatch, module)
    with pytest.raises(ValueError, match="Synthetic-DS") as error:
        MessyConfig.load("Synthetic-DS")
    assert "MEDS_extract.pipelines" in str(error.value)
    assert module in str(error.value)
    assert missing in str(error.value)
    assert "Install" in str(error.value)
    assert isinstance(error.value.__cause__, ModuleNotFoundError)
    assert error.value.__cause__.name == missing


def test_resource_lookup_imports_package_without_loading_entry_point(tmp_path, monkeypatch):
    package = tmp_path / "synthetic_imported_pkg"
    package.mkdir()
    (package / "__init__.py").write_text(
        "from pathlib import Path\nPath(__file__).with_name('imported.txt').write_text('imported')\n"
    )
    (package / "events.yaml").write_text("patients:\n  dob: {code: BIRTH, time: null}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    register(monkeypatch, "synthetic_imported_pkg")
    messy = MessyConfig.load("Synthetic-DS")
    assert messy.registered_name == "Synthetic-DS"
    assert messy.event_tables[0].events[0].name == "dob"
    assert (package / "imported.txt").read_text() == "imported"
