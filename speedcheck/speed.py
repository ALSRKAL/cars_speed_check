"""Per-vehicle speed estimation from a sliding window of tracked positions."""

from __future__ import annotations

import math
import statistics
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from .config import SpeedConfig

MPH_PER_KMH = 0.621371


def kmh_to(kmh: float, units: str) -> float:
    """Convert km/h to the requested display units."""
    if units == "mph":
        return kmh * MPH_PER_KMH
    return kmh


@dataclass
class Sample:
    """One tracked position sample."""

    frame: int
    x: float  # box centre x
    y: float  # ground contact y (box bottom) - drives the ppm lookup


class SpeedEstimator:
    """Median-filtered, EMA-smoothed speed from a sliding position window.

    Each call to :meth:`push` appends a sample and, once the window spans at
    least ``min_dt_seconds``, converts the displacement to a speed using the
    ppm at the *midpoint* row of the travelled segment.  Individual window
    estimates are median-filtered and then exponentially smoothed, which
    suppresses both single-frame tracker jitter and detection jumps.
    """

    def __init__(self, cfg: SpeedConfig, fps: float) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.cfg = cfg
        self.fps = fps
        self.samples: deque[Sample] = deque(
            maxlen=max(2, int(round(cfg.window_seconds * fps)))
        )
        self.recent: deque[float] = deque(maxlen=max(1, cfg.median_window))
        self._ema: float | None = None

    @property
    def value(self) -> float | None:
        """Current smoothed estimate, or ``None`` until one is available."""
        return self._ema

    def push(
        self, frame: int, x: float, y: float, ppm_at: Callable[[float], float]
    ) -> float | None:
        """Add a sample and return the smoothed speed (km/h) if available."""
        self.samples.append(Sample(frame, x, y))
        first = self.samples[0]
        dt = (frame - first.frame) / self.fps
        if dt < self.cfg.min_dt_seconds:
            return self._ema

        dist = math.hypot(x - first.x, y - first.y)
        if dist < self.cfg.min_displacement_px:
            return self._ema

        mid_ppm = ppm_at((y + first.y) / 2.0)
        if mid_ppm <= 0:
            return self._ema

        kmh = (dist / mid_ppm) / dt * 3.6
        if kmh > self.cfg.max_plausible_kmh:
            return self._ema  # almost certainly a tracking glitch

        self.recent.append(kmh)
        median = statistics.median(self.recent)
        if self._ema is None:
            self._ema = median
        else:
            self._ema += self.cfg.ema_alpha * (median - self._ema)
        return self._ema
