"""End-of-run per-vehicle report and CSV export."""

from __future__ import annotations

import csv
from dataclasses import dataclass

from .paths import safe_output_path
from .speed import kmh_to
from .tracker import VehicleTrack


@dataclass
class VehicleSummary:
    """Aggregated stats for one vehicle track."""

    tid: int
    cls_name: str
    direction: str
    first_s: float
    last_s: float
    duration_s: float
    mean_kmh: float
    max_kmh: float
    over_limit: bool


class SessionReport:
    """Collects finished tracks and renders them as a table and CSV."""

    def __init__(self, units: str = "kmh", speed_limit: float | None = None) -> None:
        self.units = units
        self.speed_limit = speed_limit
        self.vehicles: list[VehicleSummary] = []
        self.frames_processed = 0
        self.fps = 0.0

    def add_track(self, track: VehicleTrack) -> None:
        """Fold one finished track into the report (skips speed-less tracks)."""
        if not track.speed_history:
            return
        speeds = [s for _, s in track.speed_history]
        first_s = track.created_frame / self.fps if self.fps else 0.0
        last_s = track.last_frame / self.fps if self.fps else 0.0
        self.vehicles.append(
            VehicleSummary(
                tid=track.tid,
                cls_name=track.cls_name,
                direction=track.direction,
                first_s=first_s,
                last_s=last_s,
                duration_s=max(0.0, last_s - first_s),
                mean_kmh=sum(speeds) / len(speeds),
                max_kmh=track.max_speed,
                over_limit=track.violation,
            )
        )

    def finalize(self, frames_processed: int, fps: float) -> None:
        self.frames_processed = frames_processed
        self.fps = fps

    @property
    def violation_count(self) -> int:
        return sum(1 for v in self.vehicles if v.over_limit)

    def measured_count(self) -> int:
        return len(self.vehicles)

    def save_csv(self, path: str) -> None:
        """Write the per-vehicle table, in the configured units."""
        out = safe_output_path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        unit_lbl = "mph" if self.units == "mph" else "kmh"
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "vehicle_id",
                    "class",
                    "direction",
                    "first_seen_s",
                    "last_seen_s",
                    "duration_s",
                    f"mean_{unit_lbl}",
                    f"max_{unit_lbl}",
                    "over_limit",
                ]
            )
            for v in sorted(self.vehicles, key=lambda x: x.first_s):
                writer.writerow(
                    [
                        v.tid,
                        v.cls_name,
                        v.direction,
                        f"{v.first_s:.1f}",
                        f"{v.last_s:.1f}",
                        f"{v.duration_s:.1f}",
                        f"{kmh_to(v.mean_kmh, self.units):.1f}",
                        f"{kmh_to(v.max_kmh, self.units):.1f}",
                        v.over_limit,
                    ]
                )

    def render_table(self) -> str:
        """Plain-text summary table for terminal output."""
        unit_lbl = "mph" if self.units == "mph" else "km/h"
        header = (
            f"{'ID':>4} {'class':<10} {'dir':<5} {'first':>6} {'last':>6} "
            f"{'mean':>6} {'max':>6} {'over':>4}"
        )
        rows = [header, "-" * len(header)]
        for v in sorted(self.vehicles, key=lambda x: x.first_s):
            rows.append(
                f"{v.tid:>4} {v.cls_name:<10} {v.direction:<5} {v.first_s:>6.1f} "
                f"{v.last_s:>6.1f} {kmh_to(v.mean_kmh, self.units):>6.1f} "
                f"{kmh_to(v.max_kmh, self.units):>6.1f} {('*' if v.over_limit else ''):>4}"
            )
        rows.append("-" * len(header))
        rows.append(
            f"{self.measured_count()} vehicles measured, "
            f"{self.violation_count} over limit "
            f"(limit: {'-' if self.speed_limit is None else f'{self.speed_limit:g} ' + unit_lbl})"
        )
        return "\n".join(rows)
