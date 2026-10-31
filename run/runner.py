"""Main loop (T09): source → detector → identity → renderer → sinks, fail-closed.

The Runner never constructs components; factory_methods.py composes it (one
registry per varying dimension) and __main__.py runs it. The safety claim:

    Between start and exit, a sink only ever receives a CensoredFrame — on any
    error in the frame path, the fail-safe frame (last censored, then the card)
    is shown instead. A raw frame cannot leave run(): sinks type-reject it, and
    the raw Frame never escapes the per-frame try block.

Sink failure policy (decided in review): offline, any sink failure aborts the
run (the file is the product). Live, a non-essential sink (recording) is closed
and dropped and the loop continues; an essential sink (display) aborts.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from core.contracts import CensoredFrame, Detector, IdentityModule, Renderer, Sink

log = logging.getLogger("halal_vision.runner")


class PipelineFailure(RuntimeError):
    """The run cannot continue safely (persistent errors, essential sink gone)."""


class FrameSupply(Protocol):
    """What the Runner reads frames from: a Grabber (live) or a direct source wrapper."""

    def get(self, timeout: float | None = None): ...
    def close(self) -> None: ...


class FailSafe:
    """Owns what is shown when a frame cannot be produced.

    Within ``hold_seconds`` of the first error: the last censored frame (frozen
    image, leaks nothing). After that: the card — a startup-validated image or
    black. ``exit_seconds`` of continuous failure ends the run. Everything here
    is prebuilt: the failure path does no I/O and no allocation.
    """

    def __init__(self, frame_shape: tuple[int, int], hold_seconds: float,
                 exit_seconds: float, image_path: str | None = None) -> None:
        self.hold_seconds = hold_seconds
        self.exit_seconds = exit_seconds
        self.card = CensoredFrame(image=self._make_card(frame_shape, image_path), ts=0.0, idx=-1)
        self._last_good: CensoredFrame | None = None
        self._failing_since: float | None = None

    @staticmethod
    def _make_card(shape: tuple[int, int], image_path: str | None) -> np.ndarray:
        """The 'signal paused' image, letterboxed to the frame shape; black if none."""
        h, w = shape
        card = np.zeros((h, w, 3), dtype=np.uint8)
        if image_path:
            img = cv2.imread(image_path)   # startup-time only; the factory pre-checks the path
            if img is None:
                log.warning("failsafe image %s unreadable, using black", image_path)
            else:
                scale = min(h / img.shape[0], w / img.shape[1])
                ih, iw = round(img.shape[0] * scale), round(img.shape[1] * scale)
                y, x = (h - ih) // 2, (w - iw) // 2
                card[y:y + ih, x:x + iw] = cv2.resize(img, (iw, ih))
        return card

    def good(self, frame: CensoredFrame) -> None:
        """A frame rendered fine: remember it, clear the failure clock."""
        self._last_good = frame
        self._failing_since = None

    def frame(self, now: float) -> CensoredFrame:
        """What to show instead of a failed frame. Never raises."""
        if self._failing_since is None:
            self._failing_since = now
        if self._last_good is not None and now - self._failing_since < self.hold_seconds:
            return self._last_good
        return self.card

    def exhausted(self, now: float) -> bool:
        """True when errors have been continuous for longer than exit_seconds."""
        return self._failing_since is not None and now - self._failing_since > self.exit_seconds


@dataclass
class RunStats:
    frames: int = 0            # frames that went through the full path
    failsafe_frames: int = 0   # frames replaced by the fail-safe
    sink_failures: int = 0     # non-essential sinks dropped mid-run
    seconds: float = 0.0
    stop_reason: str = ""

    def __str__(self) -> str:
        fps = self.frames / self.seconds if self.seconds > 0 else 0.0
        return (f"{self.frames} frames in {self.seconds:.1f} s ({fps:.1f} fps), "
                f"{self.failsafe_frames} fail-safe, {self.sink_failures} sinks lost "
                f"| stopped: {self.stop_reason}")


class Runner:
    """The frame loop. Construct through factory_methods.build_runner()."""

    def __init__(self, *, frames: FrameSupply, detector: Detector, identity: IdentityModule,
                 renderer: Renderer, sinks: list[Sink], failsafe: FailSafe,
                 pacer=None, fatal_sink_errors: bool = True, frame_timeout: float = 2.0,
                 max_frames: int = 0, max_seconds: float = 0.0) -> None:
        self.frames = frames
        self.detector = detector
        self.identity = identity
        self.renderer = renderer
        self.sinks = list(sinks)
        self.failsafe = failsafe
        self.pacer = pacer
        self.fatal_sink_errors = fatal_sink_errors
        self.frame_timeout = frame_timeout
        self.max_frames = max_frames
        self.max_seconds = max_seconds
        self._closed = False

    def run(self) -> RunStats:
        stats = RunStats()
        t_start = time.perf_counter()
        try:
            while True:
                if self.pacer:
                    self.pacer.wait()

                # ---- frame path: the raw Frame lives and dies inside this try ----
                try:
                    frame = self.frames.get(timeout=self.frame_timeout)
                    if frame is None:
                        stats.stop_reason = "end of stream"
                        break
                    detections = self.detector.detect(frame)
                    decisions = self.identity.update(frame, detections)
                    censored = self.renderer.render(frame, detections, decisions)
                    self.failsafe.good(censored)
                    stats.frames += 1
                except Exception:
                    now = time.perf_counter()
                    log.exception("frame path failed, showing fail-safe")
                    censored = self.failsafe.frame(now)
                    stats.failsafe_frames += 1
                    if self.failsafe.exhausted(now):
                        raise PipelineFailure(
                            f"frame path failing for over {self.failsafe.exit_seconds} s") from None

                self._write(censored, stats)

                if (reason := self._should_stop(stats, t_start)):
                    stats.stop_reason = reason
                    break
        except KeyboardInterrupt:
            stats.stop_reason = "interrupted"
        finally:
            stats.seconds = time.perf_counter() - t_start
            self.close()
        return stats

    def _write(self, censored: CensoredFrame, stats: RunStats) -> None:
        for sink in list(self.sinks):
            try:
                sink.write(censored)
            except Exception as exc:
                if self.fatal_sink_errors or getattr(sink, "essential", True):
                    raise PipelineFailure(f"{type(sink).__name__} failed: {exc}") from exc
                log.error("%s failed (%s); recording dropped, run continues",
                          type(sink).__name__, exc)
                self.sinks.remove(sink)
                stats.sink_failures += 1
                try:
                    sink.close()
                except Exception:
                    pass   # it already failed; closing is best-effort

    def _should_stop(self, stats: RunStats, t_start: float) -> str:
        for sink in self.sinks:
            if getattr(sink, "quit_requested", False):
                return getattr(sink, "quit_reason", "") or "quit requested"
        shown = stats.frames + stats.failsafe_frames
        if self.max_frames and shown >= self.max_frames:
            return f"max frames ({self.max_frames})"
        if self.max_seconds and time.perf_counter() - t_start >= self.max_seconds:
            return f"max seconds ({self.max_seconds})"
        return ""

    def close(self) -> None:
        """Ordered, idempotent teardown: input first, decisions, outputs last."""
        if self._closed:
            return
        self._closed = True
        for name, closer in [("frames", self.frames.close), ("identity", self.identity.close)]:
            try:
                closer()
            except Exception:
                log.exception("closing %s failed", name)
        for sink in self.sinks:
            try:
                sink.close()
            except Exception:
                log.exception("closing %s failed", type(sink).__name__)
