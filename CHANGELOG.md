# Changelog

## 2.0.0 (2026-10-03)

Complete rewrite of the original single-file experiment into a typed,
tested, installable Python package.

### Detection & tracking

- **YOLO** (`ultralytics`, yolov8n by default) replaces the Haarcascade
  classifier — far fewer false positives and missed vehicles, works across
  cars, motorcycles, buses and trucks.
- **ByteTrack** persistent multi-object tracking replaces the dlib
  correlation tracker — no more per-tracker quality tuning, stable IDs.
- `dlib` (and its painful build) is no longer a dependency.

### Speed estimation

- **Perspective-aware two-line calibration**: pixels-per-metre is measured
  on a near and a far reference line and interpolated linearly in between,
  instead of one global constant for the whole frame.
- Real source **FPS is read from the video** (the old code hardcoded 18 for
  a 25 fps video).
- Sliding-window displacement with **median filtering + EMA smoothing**, so
  tracker jitter no longer shows up as speed spikes.
- Physically implausible estimates (tracking glitches) are discarded.

### Product features

- `carspeed run | calibrate | info` CLI (also `python -m speedcheck`).
- Interactive `carspeed calibrate` helper: click a known distance near/far,
  get the ready-to-use CLI arguments.
- Speed-limit sign, colour-coded boxes (green/amber/red), trajectory trails,
  live HUD with processing FPS, vehicle counts and violation count.
- Per-vehicle **CSV report** (mean/max speed, direction, first/last seen,
  over-limit flag) and a terminal summary table.
- Snapshot key (`s`) and quit (`q`) in the preview window; `--no-display`
  for headless runs; `--units mph` support.

### Engineering

- Installable package (`pip install .` → `carspeed` command) with typed
  dataclass configs.
- Output paths (`--output`, `--csv`) are validated against workspace escape
  (`speedcheck/paths.py`).
- 20 pytest unit tests for calibration, speed estimation, tracking,
  reporting and path hardening; ruff-clean; GitHub Actions CI on
  Python 3.10–3.12.

### Compatibility notes

- The old `speed_check.py`, `myhaar.xml` and the `ppm = 8.8 / fps = 18`
  constants are gone; calibration is now explicit (two lines) and
  configurable per video.
- `outpy.avi` / `output.mp4` demos were regenerated with the new pipeline
  into `assets/`.

## 1.0.0 (2022-11)

- Original release: Haarcascade vehicle detection, dlib correlation
  tracking, single-constant pixels-per-metre speed estimate
  ([original speed_check.py](https://github.com/ALSRKAL/cars_speed_check/blob/6b1f310add0f87d3d9f4a739568f6565aee5aeba/speed_check.py)).
