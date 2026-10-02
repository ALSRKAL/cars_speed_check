"""Main processing pipeline: detect, track, estimate speeds, render, report."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from .annotate import draw_calibration, draw_hud, draw_speed_limit, draw_track
from .config import CLASS_NAMES, RunConfig
from .paths import safe_output_path
from .report import SessionReport
from .tracker import Detection, TrackManager

logger = logging.getLogger("speedcheck")


def open_capture(source: str) -> tuple[cv2.VideoCapture, float, int, int]:
    """Open a video file or webcam index; return (capture, fps, w, h)."""
    src: str | int = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"error: cannot open source {source!r}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 240:  # webcams sometimes report nonsense
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, width, height


def _detections_from(result, class_names: dict[int, str]) -> list[Detection]:
    """Convert one ultralytics result into Detection records."""
    out: list[Detection] = []
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return out
    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.int().cpu().tolist()
    clss = boxes.cls.int().cpu().tolist()
    for (x1, y1, x2, y2), tid, cls in zip(xyxy, ids, clss, strict=False):
        out.append(
            Detection(
                x=float(x1),
                y=float(y1),
                w=float(x2 - x1),
                h=float(y2 - y1),
                tid=tid,
                cls_name=class_names.get(cls, "vehicle"),
            )
        )
    return out


def run(cfg: RunConfig) -> SessionReport:
    """Process the whole source; returns the session report."""
    from ultralytics import YOLO  # imported lazily: keeps startup and tests light

    cap, fps, width, height = open_capture(cfg.source)
    if cfg.resize_width and cfg.resize_width != width:
        scale = cfg.resize_width / width
        proc_size = (cfg.resize_width, round(height * scale))
    else:
        proc_size = (width, height)

    writer: cv2.VideoWriter | None = None
    if cfg.output:
        out_path = safe_output_path(cfg.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, proc_size
        )
        if not writer.isOpened():
            raise SystemExit(f"error: cannot write video to {out_path}")

    model = YOLO(cfg.model)
    manager = TrackManager(
        fps=fps,
        calibration=cfg.calibration,
        speed_cfg=cfg.speed,
        stale_after=cfg.stale_after_frames,
        trail_length=cfg.trail_length,
    )
    report = SessionReport(units=cfg.units, speed_limit=cfg.speed_limit)
    report.fps = fps

    save_dir = Path(cfg.save_dir)
    frame_idx = 0
    proc_fps = 0.0
    logger.info(
        "source=%s %.0fx%.0f@%.1ffps -> processing at %dx%d, model=%s",
        cfg.source, width, height, fps, *proc_size, cfg.model,
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok or (cfg.max_frames is not None and frame_idx >= cfg.max_frames):
                break
            if proc_size[0] != frame.shape[1]:
                frame = cv2.resize(frame, proc_size, interpolation=cv2.INTER_AREA)

            t0 = time.perf_counter()
            result = model.track(
                frame,
                persist=True,
                conf=cfg.conf,
                iou=cfg.iou,
                classes=list(cfg.classes),
                tracker=cfg.tracker,
                imgsz=cfg.imgsz,
                verbose=False,
            )[0]
            detections = _detections_from(result, CLASS_NAMES)
            tracks = manager.update(detections, frame_idx, cfg.speed_limit)

            canvas = frame
            if cfg.show_calibration:
                draw_calibration(canvas, cfg.calibration)
            for track in tracks:
                draw_track(canvas, track, cfg.units, cfg.speed_limit, cfg.trails)

            # Average only over tracks with a settled estimate; freshly born
            # tracks (and far-away ones) would drag it towards zero.
            settled = [t.speed for t in tracks if len(t.speed_history) >= 5]
            draw_hud(
                canvas,
                {
                    "proc_fps": proc_fps,
                    "total": manager.total_seen,
                    "active": len(tracks),
                    "avg_speed": float(np.mean(settled)) if settled else None,
                    "violations": manager.violation_count,
                },
                cfg.units,
            )
            if cfg.speed_limit is not None:
                draw_speed_limit(canvas, cfg.speed_limit, cfg.units)

            if writer is not None:
                writer.write(canvas)

            if cfg.display:
                cv2.imshow("cars-speed-check", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    logger.info("stopped by user at frame %d", frame_idx)
                    break
                if key == ord("s"):
                    save_dir.mkdir(parents=True, exist_ok=True)
                    snap = save_dir / f"snapshot_{frame_idx:05d}.png"
                    cv2.imwrite(str(snap), canvas)
                    logger.info("snapshot saved: %s", snap)

            dt = time.perf_counter() - t0
            proc_fps = (proc_fps or 1 / dt) * 0.9 + (1 / dt) * 0.1
            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info("frame %d, %d active tracks", frame_idx, len(tracks))
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if cfg.display:
            cv2.destroyAllWindows()
        for track in manager.finish():
            report.add_track(track)
        report.finalize(frame_idx, fps)

    if cfg.csv_path:
        report.save_csv(cfg.csv_path)  # validates the path and creates parents
        logger.info("per-vehicle CSV saved: %s", cfg.csv_path)
    return report
