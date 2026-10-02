"""Unit tests for the perspective-aware calibration model."""

import pytest

from speedcheck.config import Calibration


def test_ppm_at_endpoints() -> None:
    cal = Calibration(far_ppm=4.0, far_y=100, near_ppm=12.0, near_y=300)
    assert cal.ppm_at(100) == pytest.approx(4.0)
    assert cal.ppm_at(300) == pytest.approx(12.0)


def test_ppm_at_midpoint_is_linear() -> None:
    cal = Calibration(far_ppm=4.0, far_y=100, near_ppm=12.0, near_y=300)
    assert cal.ppm_at(200) == pytest.approx(8.0)


def test_ppm_clamped_towards_horizon() -> None:
    cal = Calibration(far_ppm=4.0, far_y=100, near_ppm=12.0, near_y=300)
    # Above the far line (towards the horizon) the model clamps to far_ppm.
    assert cal.ppm_at(20) == pytest.approx(4.0)
    assert cal.ppm_at(-50) == pytest.approx(4.0)


def test_ppm_limited_extrapolation_below_near_line() -> None:
    cal = Calibration(far_ppm=4.0, far_y=100, near_ppm=12.0, near_y=300)
    # Half a span past the near line: t = 1.5 -> ppm = 12 + 0.5 * 8.
    assert cal.ppm_at(400) == pytest.approx(16.0)
    # Far beyond the extrapolation cap the value stays at t = 1.5.
    assert cal.ppm_at(1000) == pytest.approx(16.0)


def test_invalid_calibration_rejected() -> None:
    with pytest.raises(ValueError):
        Calibration(far_ppm=0, far_y=100, near_ppm=12.0, near_y=300)
    with pytest.raises(ValueError):
        Calibration(far_ppm=4.0, far_y=300, near_ppm=12.0, near_y=100)
