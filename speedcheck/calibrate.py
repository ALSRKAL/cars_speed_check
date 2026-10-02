"""Interactive calibration helper.

Click two points that span a known real-world distance *along the road*
(e.g. the length of a clearly visible car, or a dash+gap period of a lane
line), then type that distance in metres in the terminal.  Do this once near
the top of the road (far line) and once near the bottom (near line); the tool
derives the pixels-per-metre at each line and prints the ready-to-use CLI
arguments.
"""

from __future__ import annotations

import math

import cv2

from .pipeline import open_capture

WINDOW = "cars-speed-check calibrate"
INSTRUCTIONS = [
    "click 2 points spanning a KNOWN distance along the road",
    "then type the distance in metres in the terminal",
    "[u] undo point   [q] finish after 2 lines",
]


def _draw(frame, points, lines, current_hint=""):
    canvas = frame.copy()
    for x, y in points:
        cv2.circle(canvas, (int(x), int(y)), 5, (60, 60, 235), -1)
    if len(points) == 2:
        cv2.line(canvas, points[0], points[1], (60, 60, 235), 2)
    for i, line in enumerate(lines):
        y = int(line["y"])
        cv2.line(canvas, (0, y), (canvas.shape[1], y), (230, 160, 60), 2)
        cv2.putText(
            canvas,
            f"line {i + 1}: {line['ppm']:.2f} px/m @ y={y}",
            (10, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 160, 60),
            1,
            cv2.LINE_AA,
        )
    for j, text in enumerate(INSTRUCTIONS):
        cv2.putText(
            canvas, text, (10, canvas.shape[0] - 10 - 22 * (len(INSTRUCTIONS) - 1 - j)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA,
        )
    if current_hint:
        cv2.putText(
            canvas, current_hint, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (90, 200, 80), 1, cv2.LINE_AA,
        )
    return canvas


def run_calibration(source: str, frame_no: int = 60) -> None:
    cap, _fps, _w, _h = open_capture(source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"error: cannot read frame {frame_no} from {source!r}")

    state = {"points": [], "lines": []}

    def on_mouse(event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        state["points"].append((x, y))
        if len(state["points"]) == 2:
            (x1, y1), (x2, y2) = state["points"]
            px = math.hypot(x2 - x1, y2 - y1)
            try:
                metres = float(input("real distance in metres for this segment: "))
            except ValueError:
                print("  not a number - points discarded")
                state["points"] = []
                return
            if metres <= 0 or px < 2:
                print("  invalid - points discarded")
                state["points"] = []
                return
            state["lines"].append(
                {"ppm": px / metres, "y": (y1 + y2) / 2, "px": px, "m": metres}
            )
            print(f"  line {len(state['lines'])}: {px / metres:.2f} px/m at y={(y1 + y2) / 2:.0f}")
            state["points"] = []

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        hint = f"line {len(state['lines']) + 1}: click point {len(state['points']) + 1}/2"
        cv2.imshow(WINDOW, _draw(frame, state["points"], state["lines"], hint))
        key = cv2.waitKey(30) & 0xFF
        if key == ord("u") and state["points"]:
            state["points"].pop()
        if key == ord("q") or key == 27 or len(state["lines"]) >= 2:
            break
    cv2.destroyAllWindows()

    if len(state["lines"]) < 2:
        print("calibration needs two lines - nothing to print")
        return

    first, second = state["lines"][:2]
    far, near = sorted((first, second), key=lambda line: line["y"])
    print("\nCalibration result")
    print(f"  far : {far['ppm']:.2f} px/m at y={far['y']:.0f}")
    print(f"  near: {near['ppm']:.2f} px/m at y={near['y']:.0f}")
    print("\nUse it like this:\n")
    print(
        f"  carspeed run {source} "
        f"--ppm-far {far['ppm']:.2f}@{far['y']:.0f} "
        f"--ppm-near {near['ppm']:.2f}@{near['y']:.0f} --limit 60"
    )
