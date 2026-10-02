"""Tests for the output-path hardening."""

import pytest

from speedcheck.paths import UnsafePath, safe_output_path


def test_relative_path_stays_inside_workspace(tmp_path) -> None:
    out = safe_output_path("reports/vehicles.csv", cwd=tmp_path)
    assert out == tmp_path / "reports" / "vehicles.csv"


def test_traversal_out_of_workspace_rejected(tmp_path) -> None:
    with pytest.raises(UnsafePath):
        safe_output_path("../escape.csv", cwd=tmp_path)
    with pytest.raises(UnsafePath):
        safe_output_path("a/../../escape.csv", cwd=tmp_path)


def test_absolute_path_is_allowed_as_explicit(tmp_path) -> None:
    out = safe_output_path(str(tmp_path / "elsewhere" / "out.csv"), cwd=tmp_path)
    assert out.name == "out.csv"


def test_existing_directory_rejected(tmp_path) -> None:
    (tmp_path / "sub").mkdir()
    with pytest.raises(UnsafePath):
        safe_output_path("sub", cwd=tmp_path)


def test_user_expansion(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    out = safe_output_path("~/out.csv", cwd=tmp_path)
    assert out == tmp_path / "out.csv"
