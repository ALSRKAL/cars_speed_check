"""cars-speed-check: vehicle speed estimation from video.

Detects road vehicles with YOLO, follows them with ByteTrack persistent IDs
and estimates each vehicle's speed with a perspective-aware pixels-per-metre
calibration model.
"""

from .config import Calibration, RunConfig, SpeedConfig
from .speed import SpeedEstimator

__version__ = "2.0.0"

__all__ = [
    "Calibration",
    "RunConfig",
    "SpeedConfig",
    "SpeedEstimator",
    "__version__",
]
