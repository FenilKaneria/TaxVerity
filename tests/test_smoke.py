from importlib.metadata import version

import taxverity


def test_version_constant_is_exposed():
    assert taxverity.__version__ == "0.1.0"


def test_package_is_installed_and_metadata_matches_source_constant():
    assert version("taxverity") == taxverity.__version__
