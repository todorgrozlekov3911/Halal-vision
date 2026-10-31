"""T09 acceptance: an injected error never shows a raw frame.

All components are fakes: the Runner is exercised against the contracts only,
so these tests need no model, no renderer implementation, no video files.
Raw frames are filled with RAW=100; the fake renderer outputs SAFE=7. A leak
would be a recorded image containing 100.

Runs under pytest, or standalone:  python -m tests.test_runner
"""
from __future__ import annotations

import itertools
import time

import numpy as np

from core.contracts import (CONCEAL_NIQAB, CensoredFrame, ConcealAllIdentity, Detections,
                            Detector, Frame, MaskGeometry, Renderer, Sink)
from run.runner import FailSafe, PipelineFailure, Runner, RunStats

RAW, SAFE = 100, 7
SHAPE = (8, 8)


def make_frame(idx: int) -> Frame:
    img = np.full((*SHAPE, 3), RAW, dtype=np.uint8)
    return Frame(image=img, ts=idx / 30.0, idx=idx, source_id="test")


def empty_detections(frame: Frame) -> Detections:
    return Detections(frame_idx=frame.idx, ts=frame.ts, tracks=(),
                      masks=np.zeros((0, 4, 4), dtype=np.uint8),
                      geometry=MaskGeometry(scale=1.0, pad_x=0, pad_y=0, frame_shape=SHAPE))


class FakeSupply:
    """Hands out the given frames, then None. Grabber-shaped."""

    def __init__(self, frames) -> None:
        self._iter = iter(frames)
        self.closed = False

    def get(self, timeout=None):
        return next(self._iter, None)

    def close(self) -> None:
        self.closed = True


class FakeDetector(Detector):
    def __init__(self, fail_at: set[int] = frozenset(), fail_always: bool = False) -> None:
        self.fail_at = set(fail_at)
        self.fail_always = fail_always

    def detect(self, frame: Frame) -> Detections:
        if self.fail_always or frame.idx in self.fail_at:
            raise RuntimeError(f"injected detector error at frame {frame.idx}")
        return empty_detections(frame)

    def reset(self) -> None:
        pass


class FakeRenderer(Renderer):
    def __init__(self, fail_at: set[int] = frozenset()) -> None:
        self.fail_at = set(fail_at)

    def render(self, frame, detections, decisions) -> CensoredFrame:
        if frame.idx in self.fail_at:
            raise RuntimeError(f"injected renderer error at frame {frame.idx}")
        return CensoredFrame(image=np.full((*SHAPE, 3), SAFE, dtype=np.uint8),
                             ts=frame.ts, idx=frame.idx)

    def reset(self) -> None:
        pass


class RecordingSink(Sink):
    def __init__(self) -> None:
        self.frames: list[CensoredFrame] = []

    def _write(self, frame: CensoredFrame) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        pass


class FailingSink(Sink):
    def __init__(self, fail_at: int, essential: bool) -> None:
        self.fail_at = fail_at
        self.essential = essential
        self.count = 0

    def _write(self, frame: CensoredFrame) -> None:
        if self.count >= self.fail_at:
            raise OSError("injected sink error (disk full)")
        self.count += 1

    def close(self) -> None:
        pass


def build(n_frames=6, detector=None, renderer=None, sinks=None, failsafe=None, **kw):
    sinks = sinks if sinks is not None else [RecordingSink()]
    runner = Runner(
        frames=FakeSupply([make_frame(i) for i in range(n_frames)]),
        detector=detector or FakeDetector(),
        identity=ConcealAllIdentity(CONCEAL_NIQAB, sync=True),
        renderer=renderer or FakeRenderer(),
        sinks=sinks,
        failsafe=failsafe or FailSafe(SHAPE, hold_seconds=10.0, exit_seconds=60.0),
        **kw,
    )
    return runner, sinks[0]


def assert_no_raw(recorded: list[CensoredFrame]) -> None:
    for i, frame in enumerate(recorded):
        assert isinstance(frame, CensoredFrame), f"frame {i} is {type(frame).__name__}"
        assert not np.any(frame.image == RAW), f"RAW pixels leaked in recorded frame {i}"


def test_clean_run() -> None:
    runner, rec = build()
    stats = runner.run()
    assert stats.frames == 6 and stats.failsafe_frames == 0
    assert stats.stop_reason == "end of stream"
    assert len(rec.frames) == 6
    assert_no_raw(rec.frames)
    assert runner.frames.closed, "supply was not closed"


def test_detector_error_shows_last_censored_never_raw() -> None:
    runner, rec = build(detector=FakeDetector(fail_at={3}))
    stats = runner.run()
    assert stats.frames == 5 and stats.failsafe_frames == 1
    assert len(rec.frames) == 6, "the failed frame must still produce output"
    assert rec.frames[3] is rec.frames[2], "fail-safe must reuse the last censored frame"
    assert_no_raw(rec.frames)


def test_renderer_error_never_shows_raw() -> None:
    runner, rec = build(renderer=FakeRenderer(fail_at={1, 2}))
    runner.run()
    assert_no_raw(rec.frames)


def test_first_frame_error_shows_card() -> None:
    failsafe = FailSafe(SHAPE, hold_seconds=10.0, exit_seconds=60.0)
    runner, rec = build(detector=FakeDetector(fail_at={0}), failsafe=failsafe)
    runner.run()
    assert rec.frames[0] is failsafe.card, "no last-censored exists yet: must show the card"
    assert_no_raw(rec.frames)


def test_hold_expires_to_card() -> None:
    failsafe = FailSafe(SHAPE, hold_seconds=0.0, exit_seconds=60.0)  # hold disabled
    runner, rec = build(detector=FakeDetector(fail_at={3}), failsafe=failsafe)
    runner.run()
    assert rec.frames[3] is failsafe.card, "hold of 0 s must go straight to the card"


def test_persistent_failure_aborts() -> None:
    frames = (make_frame(i) for i in itertools.count())        # endless supply
    failsafe = FailSafe(SHAPE, hold_seconds=0.0, exit_seconds=0.05)
    runner = Runner(frames=FakeSupply(frames), detector=FakeDetector(fail_always=True),
                    identity=ConcealAllIdentity(CONCEAL_NIQAB, sync=True), renderer=FakeRenderer(),
                    sinks=[RecordingSink()], failsafe=failsafe, max_seconds=5.0)
    t0 = time.perf_counter()
    try:
        runner.run()
    except PipelineFailure:
        assert time.perf_counter() - t0 < 2.0, "abort took far longer than exit_seconds"
    else:
        raise AssertionError("persistent failure did not abort the run")


def test_live_recording_failure_drops_sink_and_continues() -> None:
    rec, bad = RecordingSink(), FailingSink(fail_at=2, essential=False)
    runner, _ = build(sinks=[rec, bad], fatal_sink_errors=False)
    stats = runner.run()
    assert stats.sink_failures == 1
    assert bad not in runner.sinks, "failed sink must be removed"
    assert len(rec.frames) == 6, "the surviving sink must see every frame"


def test_live_essential_sink_failure_aborts() -> None:
    bad = FailingSink(fail_at=2, essential=True)
    runner, _ = build(sinks=[RecordingSink(), bad], fatal_sink_errors=False)
    try:
        runner.run()
    except PipelineFailure:
        pass
    else:
        raise AssertionError("essential sink failure did not abort")


def test_offline_any_sink_failure_aborts() -> None:
    bad = FailingSink(fail_at=2, essential=False)
    runner, _ = build(sinks=[RecordingSink(), bad], fatal_sink_errors=True)
    try:
        runner.run()
    except PipelineFailure:
        pass
    else:
        raise AssertionError("offline sink failure did not abort")


if __name__ == "__main__":
    for test in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        test()
        print(f"PASS {test.__name__}")
    print("all runner tests passed")