"""Command-line interface: carspeed run | calibrate | info."""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .config import DEFAULT_CLASSES, Calibration, RunConfig, SpeedConfig

DESCRIPTION = (
    "Vehicle speed estimation from video: YOLO detection, ByteTrack tracking, "
    "perspective-aware pixels-per-metre calibration."
)


def parse_ppm(value: str) -> tuple[float, float]:
    """Parse a 'ppm@y' argument such as 8.8@140."""
    try:
        ppm_s, y_s = value.split("@")
        ppm, y = float(ppm_s), float(y_s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected PPM@Y (e.g. 8.8@140), got {value!r}"
        ) from exc
    if ppm <= 0 or y < 0:
        raise argparse.ArgumentTypeError("ppm must be > 0 and y must be >= 0")
    return ppm, y


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carspeed", description=DESCRIPTION
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="process a video file or webcam")
    run_p.add_argument("source", help="path to a video file, or a webcam index")
    run_p.add_argument("-o", "--output", help="save the annotated video to this path")
    run_p.add_argument("--csv", help="save a per-vehicle CSV report to this path")
    run_p.add_argument(
        "--model", default="yolov8n.pt",
        help="YOLO weights (default: %(default)s)",
    )
    run_p.add_argument(
        "--tracker", default="bytetrack.yaml",
        help="tracker config (default: %(default)s)",
    )
    run_p.add_argument(
        "--conf", type=float, default=0.35,
        help="detection confidence (default: %(default)s)",
    )
    run_p.add_argument(
        "--iou", type=float, default=0.5, help="NMS IoU (default: %(default)s)"
    )
    run_p.add_argument(
        "--imgsz", type=int, default=640,
        help="inference size (default: %(default)s)",
    )
    run_p.add_argument(
        "--classes", default="car,motorcycle,bus,truck",
        help="comma-separated vehicle classes (default: %(default)s)",
    )
    run_p.add_argument(
        "--ppm-far", type=parse_ppm, required=True, metavar="PPM@Y",
        help="pixels-per-metre on the far calibration line",
    )
    run_p.add_argument(
        "--ppm-near", type=parse_ppm, required=True, metavar="PPM@Y",
        help="pixels-per-metre on the near calibration line",
    )
    run_p.add_argument(
        "--limit", type=float, default=None,
        help="speed limit for violation highlighting",
    )
    run_p.add_argument("--units", choices=("kmh", "mph"), default="kmh")
    run_p.add_argument(
        "--window", type=float, default=0.6,
        help="speed window seconds (default: %(default)s)",
    )
    run_p.add_argument(
        "--resize", type=int, default=None, help="downscale frames to this width"
    )
    run_p.add_argument(
        "--max-frames", type=int, default=None, help="stop after this many frames"
    )
    run_p.add_argument(
        "--no-display", action="store_true", help="headless run (no preview window)"
    )
    run_p.add_argument(
        "--no-trails", action="store_true", help="hide trajectory trails"
    )
    run_p.add_argument(
        "--show-calibration", action="store_true", help="overlay calibration lines"
    )
    run_p.add_argument(
        "--save-dir", default="outputs", help="where snapshots go (default: %(default)s)"
    )

    cal_p = sub.add_parser("calibrate", help="interactive two-line ppm calibration")
    cal_p.add_argument("source", help="path to a video file")
    cal_p.add_argument(
        "--frame", type=int, default=60, help="frame number to show (default: %(default)s)"
    )

    info_p = sub.add_parser("info", help="print source metadata")
    info_p.add_argument("source", help="path to a video file, or a webcam index")
    return parser


CLASS_ALIASES = {"car": 2, "motorcycle": 3, "bus": 5, "truck": 7}


def parse_classes(value: str) -> tuple[int, ...]:
    out = []
    for token in value.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in CLASS_ALIASES:
            out.append(CLASS_ALIASES[token])
        elif token.isdigit() and int(token) in DEFAULT_CLASSES:
            out.append(int(token))
        else:
            raise SystemExit(f"error: unknown vehicle class {token!r}")
    return tuple(sorted(set(out))) or DEFAULT_CLASSES


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run as run_pipeline

    (far_ppm, far_y), (near_ppm, near_y) = args.ppm_far, args.ppm_near
    calibration = Calibration(
        far_ppm=far_ppm, far_y=far_y, near_ppm=near_ppm, near_y=near_y
    )
    cfg = RunConfig(
        source=args.source,
        calibration=calibration,
        speed=SpeedConfig(window_seconds=args.window),
        model=args.model,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=parse_classes(args.classes),
        units=args.units,
        speed_limit=args.limit,
        output=args.output,
        csv_path=args.csv,
        save_dir=args.save_dir,
        resize_width=args.resize,
        max_frames=args.max_frames,
        display=not args.no_display,
        trails=not args.no_trails,
        show_calibration=args.show_calibration,
    )
    report = run_pipeline(cfg)
    print()
    print(report.render_table())
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from .calibrate import run_calibration

    run_calibration(args.source, args.frame)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    import cv2

    from .pipeline import open_capture

    cap, fps, width, height = open_capture(args.source)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    print(f"source : {args.source}")
    print(f"size   : {width}x{height}")
    print(f"fps    : {fps:.2f}")
    print(f"frames : {frames:.0f}" if frames else "frames : unknown")
    if frames:
        print(f"length : {frames / fps:.1f} s")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args(argv)
    handlers = {"run": cmd_run, "calibrate": cmd_calibrate, "info": cmd_info}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
