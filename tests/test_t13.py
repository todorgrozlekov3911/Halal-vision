"""T13 acceptance: meters measure, the overlay shows and never leaks,
the dump records every shown frame, the glasses output refuses the overlay.

Runs under pytest, or standalone:  python -m tests.test_t13
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from core.config_table import ConfigError, ConfigTable
from core.contracts import (CONCEAL_NIQAB, CensoredFrame, Decision, Detections, MaskGeometry,
                            Track, TrackState)
from core.eval.dump import DecisionDump
from core.render.overlay import DebugOverlay
from core.video.meters import LoopMeters
from run.factory_methods import BuildContext, _openvino_sees, _validate
from tests.test_runner import RAW, SAFE, SHAPE, RecordingSink, build, make_frame

# -- fixtures ----------------------------------------------------------------


def safe_frame(idx: int = 0) -> CensoredFrame:
    return CensoredFrame(image=np.full((*SHAPE, 3), SAFE, dtype=np.uint8), ts=idx / 30.0, idx=idx)


def one_track_detections(idx: int = 0, track_id: int | None = 7) -> Detections:
    track = Track(track_id=track_id, bbox=np.array([1, 1, 6, 6], dtype=np.float32),
                  score=0.9, mask_idx=0)
    return Detections(frame_idx=idx, ts=idx / 30.0, tracks=(track,),
                      masks=np.zeros((1, 4, 4), dtype=np.uint8),
                      geometry=MaskGeometry(scale=1.0, pad_x=0, pad_y=0, frame_shape=SHAPE))


def decision_for(track_id: int = 7) -> dict[int, Decision]:
    return {track_id: Decision(track_id, CONCEAL_NIQAB, TrackState.CONCEALED,
                               rule="stub", confidence=0.9)}


# -- meters ------------------------------------------------------------------


def test_meters_fps_and_latency_live() -> None:
    meters = LoopMeters(live=True, window=30)
    assert meters.fps == 0.0 and meters.latency_ms is None
    for i in range(10):                              # shown every 50 ms, captured 70 ms earlier
        meters.tick(censored_ts=i * 0.05, now=i * 0.05 + 0.07, real=True)
    assert meters.fps == pytest.approx(20.0, rel=0.01)
    assert meters.latency_ms == pytest.approx(70.0, rel=0.01)
    assert meters.avg_latency_ms() == pytest.approx(70.0, rel=0.01)


def test_meters_failsafe_frames_count_for_fps_not_latency() -> None:
    meters = LoopMeters(live=True)
    meters.tick(censored_ts=0.0, now=0.05, real=True)
    meters.tick(censored_ts=0.0, now=0.10, real=False)   # fail-safe: stale ts, no latency
    assert meters.fps > 0.0
    assert meters.latency_ms == pytest.approx(50.0, rel=0.01)


def test_meters_offline_has_no_latency() -> None:
    meters = LoopMeters(live=False)
    meters.tick(censored_ts=0.0, now=100.0, real=True)   # ts is video time: not comparable
    meters.tick(censored_ts=0.1, now=100.1, real=True)
    assert meters.latency_ms is None and meters.avg_latency_ms() is None
    assert meters.fps > 0.0


# -- overlay -----------------------------------------------------------------


def test_overlay_emits_new_censored_frame_and_never_raw() -> None:
    censored = safe_frame()
    out = DebugOverlay().apply(censored, one_track_detections(), decision_for(), LoopMeters(live=True))
    assert isinstance(out, CensoredFrame)
    assert out is not censored and out.image is not censored.image
    assert out.ts == censored.ts and out.idx == censored.idx
    assert not np.any(out.image == RAW)
    assert np.all(censored.image == SAFE), "the input frame must not be drawn on"


def test_overlay_draws_something() -> None:
    censored = safe_frame()
    out = DebugOverlay().apply(censored, one_track_detections(), decision_for(), LoopMeters(live=True))
    assert np.any(out.image != SAFE), "boxes and text must change pixels"


def test_overlay_handles_idless_track_and_missing_decision() -> None:
    overlay = DebugOverlay()
    overlay.apply(safe_frame(), one_track_detections(track_id=None), {}, None)
    overlay.apply(safe_frame(), one_track_detections(track_id=7), {}, None)   # no decision yet


def test_overlay_handles_failsafe_frame() -> None:
    out = DebugOverlay().apply(safe_frame(), None, {}, LoopMeters(live=True))
    assert isinstance(out, CensoredFrame)


# -- dump --------------------------------------------------------------------


def test_dump_writes_one_json_line_per_frame(tmp_path) -> None:
    path = tmp_path / "decisions.jsonl"
    dump = DecisionDump(path)
    meters = LoopMeters(live=True)
    meters.tick(0.0, 0.07, real=True)
    meters.tick(0.05, 0.12, real=True)
    dump.write(safe_frame(0), one_track_detections(0), decision_for(), meters)
    dump.write(safe_frame(1), None, {}, meters)          # a fail-safe frame
    dump.close()

    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 2
    first, second = lines
    assert first["frame"] == 0 and first["failsafe"] is False
    assert first["fps"] > 0 and first["latency_ms"] == pytest.approx(70.0, rel=0.01)
    (track,) = first["tracks"]
    assert track["id"] == 7 and track["state"] == "concealed"
    assert track["rule"] == "stub" and track["action"] == "conceal:niqab"
    assert len(track["bbox"]) == 4
    assert second["failsafe"] is True and second["tracks"] == []


def test_dump_idless_track_is_conceal_by_default(tmp_path) -> None:
    path = tmp_path / "d.jsonl"
    dump = DecisionDump(path)
    dump.write(safe_frame(), one_track_detections(track_id=None), {}, None)
    dump.close()
    (track,) = json.loads(path.read_text())["tracks"]
    assert track["id"] is None and track["state"] is None and track["action"] == "conceal"


# -- wired into the Runner ---------------------------------------------------


def test_runner_with_all_t13_components(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    meters = LoopMeters(live=True)
    runner, rec = build(n_frames=5, meters=meters, overlay=DebugOverlay(), dump=DecisionDump(path))
    stats = runner.run()
    assert stats.frames == 5
    assert stats.avg_latency_ms is not None, "live meters must land in RunStats"
    for frame in rec.frames:
        assert isinstance(frame, CensoredFrame)
        assert not np.any(frame.image == RAW)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 5, "every shown frame must be dumped"
    assert all(line["failsafe"] is False for line in lines)


def test_runner_dump_records_failsafe_frames(tmp_path) -> None:
    from tests.test_runner import FakeDetector
    path = tmp_path / "run.jsonl"
    runner, _ = build(n_frames=4, detector=FakeDetector(fail_at={2}), dump=DecisionDump(path))
    runner.run()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 4
    assert [line["failsafe"] for line in lines] == [False, False, True, False]


# -- the glasses refusal -----------------------------------------------------


def _ctx(device: str = "cpu", **flags) -> BuildContext:
    args = argparse.Namespace(**{"debug_overlay": False, "fullscreen": False, **flags})
    cfg = ConfigTable({"detector": {"device": device}})
    return BuildContext(cfg=cfg, args=args, source=None, live=True)


def test_overlay_refused_on_glasses_output() -> None:
    with pytest.raises(ConfigError, match="glasses"):
        _validate(_ctx(debug_overlay=True, fullscreen=True), sinks=[RecordingSink()])


def test_overlay_allowed_on_dev_window() -> None:
    _validate(_ctx(debug_overlay=True), sinks=[RecordingSink()])


# -- no silent device fallback -------------------------------------------------


def test_source_dispatch_camera_vs_file() -> None:
    from run.factory_methods import source_kind
    assert source_kind("/dev/video0") == "camera"
    assert source_kind("clips/x.mp4") == "file"


def test_openvino_device_matching() -> None:
    assert _openvino_sees("GPU", ["CPU", "GPU"])
    assert _openvino_sees("GPU", ["CPU", "GPU.0"])
    assert not _openvino_sees("NPU", ["CPU", "GPU"])


def test_unavailable_intel_device_is_refused_not_degraded(monkeypatch) -> None:
    import openvino

    class FakeCore:
        available_devices = ["CPU", "GPU"]

    monkeypatch.setattr(openvino, "Core", FakeCore)
    with pytest.raises(ConfigError, match="does not fall back"):
        _validate(_ctx(device="intel:npu"), sinks=[RecordingSink()])
    _validate(_ctx(device="intel:gpu"), sinks=[RecordingSink()])   # a visible device passes


if __name__ == "__main__":
    import sys
    sys.exit(__import__("pytest").main([__file__, "-q"]))