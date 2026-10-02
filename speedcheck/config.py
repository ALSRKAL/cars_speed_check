"""Typed configuration models for cars-speed-check."""

from __future__ import annotations

from dataclasses import dataclass, field

# COCO classes treated as road vehicles: car, motorcycle, bus, truck.
DEFAULT_CLASSES: tuple[int, ...] = (2, 3, 5, 7)

CLASS_NAMES: dict[int, str] = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass(frozen=True)
class Calibration:
    """Perspective-aware pixels-per-metre (ppm) model.

    A road seen by a fixed camera has no single ppm: a metre near the camera
    projects to many more pixels than a metre close to the horizon.  The model
    is anchored by two horizontal reference lines measured during calibration:

    * the **far** line (smaller ``y``, closer to the horizon) with ``far_ppm``
    * the **near** line (larger ``y``, closer to the camera) with ``near_ppm``

    Between the two lines ppm is interpolated linearly.  Below the far line
    (towards the horizon) the value is clamped to ``far_ppm`` because the
    linear model degenerates there; past the near line it may extrapolate up
    to ``MAX_EXTRAPOLATION`` times the near/far gap before being clamped.
    """

    far_ppm: float
    far_y: float
    near_ppm: float
    near_y: float

    MAX_EXTRAPOLATION: float = 1.5

    def __post_init__(self) -> None:
        if self.far_ppm <= 0 or self.near_ppm <= 0:
            raise ValueError("ppm values must be positive")
        if self.near_y <= self.far_y:
            raise ValueError("near_y must be below (larger than) far_y")

    def ppm_at(self, y: float) -> float:
        """Return pixels-per-metre at image row ``y``."""
        span = self.near_y - self.far_y
        t = (y - self.far_y) / span
        t = min(max(t, 0.0), self.MAX_EXTRAPOLATION)
        return self.far_ppm + t * (self.near_ppm - self.far_ppm)


@dataclass(frozen=True)
class SpeedConfig:
    """Tuning of the per-vehicle speed estimator."""

    # Length of the sliding window used to measure displacement.
    window_seconds: float = 0.6
    # Minimum time span before a window estimate is trusted.
    min_dt_seconds: float = 0.3
    # Ignore windows where the vehicle barely moved (tracker jitter).
    min_displacement_px: float = 2.0
    # Number of recent window estimates combined by median filtering.
    median_window: int = 5
    # Exponential smoothing factor applied on top of the median.
    ema_alpha: float = 0.35
    # Estimates above this are treated as tracking glitches and discarded.
    max_plausible_kmh: float = 200.0


@dataclass(frozen=True)
class RunConfig:
    """Everything the processing pipeline needs for one run."""

    source: str
    calibration: Calibration
    speed: SpeedConfig = field(default_factory=SpeedConfig)

    model: str = "yolov8n.pt"
    tracker: str = "bytetrack.yaml"
    conf: float = 0.35
    iou: float = 0.5
    imgsz: int = 640
    classes: tuple[int, ...] = DEFAULT_CLASSES

    units: str = "kmh"  # "kmh" or "mph"
    speed_limit: float | None = None

    output: str | None = None
    csv_path: str | None = None
    save_dir: str = "outputs"

    resize_width: int | None = None
    max_frames: int | None = None
    display: bool = True
    trails: bool = True
    trail_length: int = 30
    show_calibration: bool = False

    # Drop a track after this many frames without a detection update.
    stale_after_frames: int = 40
