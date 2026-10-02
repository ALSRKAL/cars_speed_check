"""Unit tests for the sliding-window speed estimator."""

import pytest

from speedcheck.config import Calibration, SpeedConfig
from speedcheck.speed import SpeedEstimator, kmh_to

CAL = Calibration(far_ppm=10.0, far_y=100, near_ppm=10.0, near_y=300)
FPS = 25.0


def feed(est: SpeedEstimator, start: int, end: int, px_per_frame: float) -> None:
    for f in range(start, end):
        est.push(f, x=100.0 + px_per_frame * f, y=200.0, ppm_at=CAL.ppm_at)


def test_constant_motion_converges_to_true_speed() -> None:
    # 2 px/frame at 25 fps with 10 px/m -> 0.2 m/frame * 25 * 3.6 = 18 km/h.
    est = SpeedEstimator(SpeedConfig(), FPS)
    feed(est, 0, 60, px_per_frame=2.0)
    assert est.value == pytest.approx(18.0, rel=0.1)


def test_returns_none_until_window_is_reached() -> None:
    est = SpeedEstimator(SpeedConfig(), FPS)
    assert est.value is None
    est.push(0, 100.0, 200.0, CAL.ppm_at)
    assert est.value is None  # only one sample so far
    feed(est, 1, 3, px_per_frame=2.0)
    assert est.value is None  # 3 frames @ 25 fps = 0.12 s < min_dt 0.3 s
    feed(est, 3, 12, px_per_frame=2.0)
    assert est.value is not None


def test_jitter_is_suppressed() -> None:
    est = SpeedEstimator(SpeedConfig(), FPS)
    for f in range(80):
        jitter = 4.0 if f % 2 else -4.0
        est.push(f, 100.0 + 2.0 * f + jitter, 200.0, CAL.ppm_at)
    assert est.value == pytest.approx(18.0, rel=0.2)


def test_tracking_glitch_does_not_explode_the_estimate() -> None:
    est = SpeedEstimator(SpeedConfig(), FPS)
    feed(est, 0, 40, px_per_frame=2.0)
    baseline = est.value
    # One wildly wrong sample (a teleport of 400 px in one frame).
    est.push(40, 500.0, 200.0, CAL.ppm_at)
    assert est.value == pytest.approx(baseline, rel=0.35)


def test_kmh_to_mph() -> None:
    assert kmh_to(100.0, "kmh") == pytest.approx(100.0)
    assert kmh_to(100.0, "mph") == pytest.approx(62.1371, abs=0.01)
