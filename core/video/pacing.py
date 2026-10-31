"""Fixed processing rate for live runs: work at ``fps`` on purpose, not as fast as possible.

A steady rate keeps the laptop cooler (the iGPU idles between frames, so it
throttles less) and makes timing predictable: the tracker sees a constant fps,
and runs are comparable. If a frame takes longer than its slot, the pacer does
not try to catch up; it just starts the next slot from now.
"""
from __future__ import annotations

import time


class Pacer:
    def __init__(self, fps: float) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.period = 1.0 / fps
        self.late = 0  # slots where the work took longer than the period
        self._next: float | None = None

    def wait(self) -> None:
        """Call once per loop iteration, before reading the next frame."""
        now = time.perf_counter()
        if self._next is None:
            self._next = now + self.period
            return
        if now < self._next:
            time.sleep(self._next - now)
            self._next += self.period
        else:
            self.late += 1
            self._next = now + self.period
