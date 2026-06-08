"""Smoke test: the package installs, imports, and exposes a version string."""
import importlib

import plutobt


def test_package_imports():
    assert importlib.import_module("plutobt") is plutobt


def test_version_is_pep440_string():
    assert isinstance(plutobt.__version__, str)
    assert plutobt.__version__ == "0.1.0"
