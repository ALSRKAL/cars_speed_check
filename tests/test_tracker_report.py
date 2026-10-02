"""Tests for track management and the session report."""

import pytest

from speedcheck.config import Calibration, SpeedConfig
from speedcheck.report import SessionReport
from speedcheck.tracker import Detection, TrackManager

CAL = Calibration(far_ppm=10.0, far_y=100, near_ppm=10.0, near_y=300)
FPS = 25.0


def make_manager() -> TrackManager:
    return TrackManager(
        fps=FPS, calibration=CAL, speed_cfg=SpeedConfig(),
        stale_after=10, trail_length=30,
    )


def det(tid: int, x: float, y: float = 100.0, cls: str = "car") -> Detection:
    return Detection(x=x, y=y, w=60.0, h=30.0, tid=tid, cls_name=cls)


def drive(
    mgr: TrackManager, frames: int, x0: float, step: float, tid: int = 1,
    limit: float | None = None,
):
    tracks = []
    for f in range(frames):
        tracks = mgr.update([det(tid, x0 + step * f)], f, limit=limit)
    return tracks


def test_ids_are_tracked_and_counted_once() -> None:
    mgr = make_manager()
    drive(mgr, 30, x0=100.0, step=3.0, tid=1)
    drive(mgr, 30, x0=200.0, step=3.0, tid=1)
    mgr.update([det(7, 400.0)], 60, limit=None)
    assert mgr.total_seen == 2
    assert mgr.violation_count == 0


def test_stale_tracks_are_retired_and_finished_kept() -> None:
    mgr = make_manager()
    drive(mgr, 20, x0=100.0, step=3.0)
    assert len(mgr.tracks) == 1
    mgr.update([], 40, limit=None)  # 20 frames of silence > stale_after=10
    assert mgr.tracks == {}
    assert len(mgr.finished) == 1


def test_direction_from_horizontal_drift() -> None:
    mgr = make_manager()
    tracks = drive(mgr, 30, x0=100.0, step=4.0)
    assert tracks[0].direction == "L->R"
    tracks = drive(mgr, 30, x0=400.0, step=-4.0, tid=2)
    assert tracks[1].direction == "R->L"


def test_violation_is_flagged_per_track() -> None:
    mgr = make_manager()
    # 3 px/frame at 25 fps, 10 px/m -> 27 km/h; limit 20 -> violation.
    tracks = drive(mgr, 40, x0=100.0, step=3.0, limit=20.0)
    assert mgr.violation_count == 1
    assert tracks[0].violation is True


def test_report_csv_contents(tmp_path) -> None:
    mgr = make_manager()
    drive(mgr, 50, x0=100.0, step=3.0, tid=3)
    mgr.finish()
    report = SessionReport(units="kmh", speed_limit=None)
    report.fps = FPS
    for track in mgr.finished:
        report.add_track(track)
    report.finalize(frames_processed=50, fps=FPS)

    csv_path = tmp_path / "vehicles.csv"
    report.save_csv(str(csv_path))
    lines = csv_path.read_text().strip().splitlines()
    assert lines[0].startswith("vehicle_id,class,direction")
    row = lines[1].split(",")
    assert row[0] == "3"
    assert row[1] == "car"
    assert float(row[6]) == pytest.approx(27.0, abs=1.5)

    table = report.render_table()
    assert "1 vehicles measured" in table or "1 vehicles" in table
