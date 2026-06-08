"""Smoke test: the package installs, imports, and exposes a version string."""
import importlib
import pathlib
import tomllib

import plutobt


def test_package_imports():
    assert importlib.import_module("plutobt") is plutobt


def test_version_is_pep440_string():
    assert isinstance(plutobt.__version__, str)
    assert plutobt.__version__ == "0.1.0"


def test_pyproject_declares_package_and_src_layout():
    root = pathlib.Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as fh:
        cfg = tomllib.load(fh)
    assert cfg["project"]["name"] == "plutobt"
    assert cfg["project"]["version"] == "0.1.0"
    assert cfg["project"]["requires-python"] == ">=3.11"
    # src layout: setuptools must find packages under src/
    assert cfg["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    # pytest must look in tests/ with importmode=importlib (src layout best practice)
    assert "tests" in cfg["tool"]["pytest"]["ini_options"]["testpaths"]
    assert cfg["tool"]["pytest"]["ini_options"]["importmode"] == "importlib"
