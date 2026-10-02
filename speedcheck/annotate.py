"""OpenCV overlays: boxes, speed labels, trails, HUD and calibration lines."""

from __future__ import annotations

import cv2
import numpy as np

from .config import Calibration
from .speed import kmh_to
from .tracker import VehicleTrack

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Vehicles smaller than this are too far away to label meaningfully; drawing
# them only adds a pile of overlapping boxes near the horizon.
MIN_BOX_H = 12

COLOR_GREEN = (90, 200, 80)
COLOR_AMBER = (40, 170, 255)
COLOR_RED = (60, 60, 235)
COLOR_PANEL = (30, 30, 30)
COLOR_TEXT = (245, 245, 245)


def speed_color(speed: float | None, limit: float | None) -> tuple[int, int, int]:
    """BGR colour for a speed, traffic-light style when a limit is set."""
    if speed is None:
        return (200, 200, 200)
    if limit is None or limit <= 0:
        return COLOR_GREEN
    ratio = speed / limit
    if ratio > 1.0:
        return COLOR_RED
    if ratio > 0.85:
        return COLOR_AMBER
    return COLOR_GREEN


def draw_track(
    frame: np.ndarray,
    track: VehicleTrack,
    units: str = "kmh",
    limit: float | None = None,
    trails: bool = True,
) -> None:
    """Draw one vehicle: box, trail, ID and speed label."""
    x, y, w, h = (int(v) for v in track.last_box)
    if h < MIN_BOX_H:
        return
    speed = track.speed
    color = speed_color(speed, limit)
    thickness = 1 if h < 22 else 2

    if trails and len(track.samples) >= 2:
        pts = np.array([(int(px), int(py)) for _, px, py in track.samples], np.int32)
        cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)

    unit_lbl = "mph" if units == "mph" else "km/h"
    if speed is None:
        label = f"#{track.tid}"
    else:
        label = f"#{track.tid} {kmh_to(speed, units):.0f} {unit_lbl}"
    scale = 0.45 if h < 22 else 0.52
    (tw, th), _ = cv2.getTextSize(label, FONT, scale, 1)
    lx = min(max(0, x), frame.shape[1] - tw - 8)
    ly = y - 7
    if ly - th < 0:
        ly = y + h + th + 7
    cv2.rectangle(frame, (lx, ly - th - 4), (lx + tw + 6, ly + 4), color, -1)
    cv2.putText(frame, label, (lx + 3, ly), FONT, scale, (20, 20, 20), 1, cv2.LINE_AA)


def draw_speed_limit(frame: np.ndarray, limit: float, units: str = "kmh") -> None:
    """Speed-limit sign (red ring) in the top-right corner."""
    radius = 34
    cx = frame.shape[1] - radius - 14
    cy = radius + 14
    cv2.circle(frame, (cx, cy), radius, (60, 60, 235), -1)
    cv2.circle(frame, (cx, cy), radius - 5, (250, 250, 250), -1)
    text = f"{limit:.0f}"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.9, 2)
    cv2.putText(frame, text, (cx - tw // 2, cy + th // 2 - 2), FONT, 0.9, (20, 20, 20), 2)
    small = "mph" if units == "mph" else "km/h"
    (sw, sh), _ = cv2.getTextSize(small, FONT, 0.35, 1)
    cv2.putText(frame, small, (cx - sw // 2, cy + th // 2 + sh + 6), FONT, 0.35, (60, 60, 60), 1)


def draw_hud(frame: np.ndarray, stats: dict[str, object], units: str = "kmh") -> None:
    """Translucent information panel in the top-left corner."""
    unit_lbl = "mph" if units == "mph" else "km/h"
    lines = [f"cars-speed-check  {stats['proc_fps']:.1f} fps"]
    lines.append(f"vehicles: {stats['total']} total / {stats['active']} active")
    if stats["avg_speed"] is not None:
        lines.append(f"avg speed: {stats['avg_speed']:.0f} {unit_lbl}")
    violations = int(stats["violations"])
    lines.append(f"violations: {violations}")

    pad, lh = 10, 22
    width = max(
        cv2.getTextSize(line, FONT, 0.52, 1)[0][0] for line in lines
    ) + 2 * pad
    height = pad * 2 + lh * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + width, 8 + height), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, line in enumerate(lines):
        color = COLOR_TEXT
        if line.startswith("violations") and violations:
            color = COLOR_RED
        cv2.putText(
            frame, line, (8 + pad, 8 + pad + lh * i + lh - 7), FONT, 0.52, color, 1,
            cv2.LINE_AA,
        )


def draw_calibration(frame: np.ndarray, cal: Calibration) -> None:
    """Show the two calibration reference lines with their ppm values."""
    width = frame.shape[1]
    for ppm, y, name in (
        (cal.far_ppm, cal.far_y, "far"),
        (cal.near_ppm, cal.near_y, "near"),
    ):
        y = int(y)
        for x0 in range(0, width, 24):
            cv2.line(frame, (x0, y), (min(x0 + 12, width), y), (230, 160, 60), 1)
        cv2.putText(
            frame, f"{name}: {ppm:.1f} px/m", (10, y - 6), FONT, 0.45, (230, 160, 60), 1,
            cv2.LINE_AA,
        )
