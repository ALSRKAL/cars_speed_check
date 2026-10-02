"""Per-vehicle state maintained on top of the detector's ByteTrack IDs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .config import Calibration, SpeedConfig
from .speed import SpeedEstimator


@dataclass
class Detection:
    """One vehicle detection for the current frame."""

    x: float  # top-left
    y: float
    w: float
    h: float
    tid: int
    cls_name: str

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def ground_y(self) -> float:
        """Bottom of the box - where the vehicle touches the road plane."""
        return self.y + self.h


@dataclass
class VehicleTrack:
    """Everything known about one vehicle.

    ``samples`` keeps the recent (frame, cx, ground_y) history used for the
    speed window and the trail overlay; ``speed_history`` records validated
    speed estimates for the end-of-run report.
    """

    tid: int
    fps: float
    calibration: Calibration
    speed_cfg: SpeedConfig
    stale_after: int
    trail_length: int = 30

    cls_name: str = "car"
    created_frame: int = -1
    last_frame: int = -1
    last_box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    samples: deque[tuple[int, float, float]] = field(default_factory=deque)
    speed_history: deque[tuple[int, float]] = field(default_factory=deque)
    estimator: SpeedEstimator = field(init=False)
    max_speed: float = 0.0
    violation: bool = False

    def __post_init__(self) -> None:
        self.estimator = SpeedEstimator(self.speed_cfg, self.fps)
        self.samples = deque(maxlen=max(self.trail_length, 2))

    @property
    def speed(self) -> float | None:
        """Current smoothed speed in km/h (``None`` until measured)."""
        return self.estimator.value

    def update(self, det: Detection, frame_idx: int, limit: float | None) -> float | None:
        """Fold a fresh detection into the track; return the smoothed speed."""
        self.cls_name = det.cls_name
        if self.created_frame < 0:
            self.created_frame = frame_idx
        self.last_frame = frame_idx
        self.last_box = (det.x, det.y, det.w, det.h)
        self.samples.append((frame_idx, det.cx, det.ground_y))

        speed = self.estimator.push(frame_idx, det.cx, det.ground_y, self.calibration.ppm_at)
        if speed is not None:
            self.speed_history.append((frame_idx, speed))
            self.max_speed = max(self.max_speed, speed)
            if limit is not None and speed > limit:
                self.violation = True
        return speed

    @property
    def direction(self) -> str:
        """Coarse travel direction from the horizontal drift of the track."""
        if len(self.samples) < 2:
            return "?"
        dx = self.samples[-1][1] - self.samples[0][1]
        if dx > 15:
            return "L->R"
        if dx < -15:
            return "R->L"
        return "->"

    @property
    def is_stale(self) -> bool:
        return self.last_frame < 0  # pragma: no cover - set on creation


class TrackManager:
    """Owns live tracks, counts vehicles and retires stale ones."""

    def __init__(
        self,
        fps: float,
        calibration: Calibration,
        speed_cfg: SpeedConfig,
        stale_after: int,
        trail_length: int,
    ) -> None:
        self.fps = fps
        self.calibration = calibration
        self.speed_cfg = speed_cfg
        self.stale_after = stale_after
        self.trail_length = trail_length
        self.tracks: dict[int, VehicleTrack] = {}
        self.finished: list[VehicleTrack] = []
        self.total_seen = 0
        self.violators: set[int] = set()

    def update(
        self, detections: list[Detection], frame_idx: int, limit: float | None
    ) -> list[VehicleTrack]:
        """Apply this frame's detections; return tracks seen in it."""
        for det in detections:
            track = self.tracks.get(det.tid)
            if track is None:
                track = VehicleTrack(
                    tid=det.tid,
                    fps=self.fps,
                    calibration=self.calibration,
                    speed_cfg=self.speed_cfg,
                    stale_after=self.stale_after,
                    trail_length=self.trail_length,
                )
                self.tracks[det.tid] = track
                self.total_seen += 1
            speed = track.update(det, frame_idx, limit)
            if limit is not None and speed is not None and speed > limit:
                self.violators.add(det.tid)

        stale = [
            tid
            for tid, t in self.tracks.items()
            if frame_idx - t.last_frame > self.stale_after
        ]
        for tid in stale:
            self.finished.append(self.tracks.pop(tid))
        return list(self.tracks.values())

    def finish(self) -> list[VehicleTrack]:
        """Retire every remaining track (end of the video)."""
        self.finished.extend(self.tracks.values())
        self.tracks.clear()
        return self.finished

    @property
    def violation_count(self) -> int:
        return len(self.violators)
