"""Live fps and camera-to-display latency (T13).

The Runner ticks once per shown frame. Latency is ``now - CensoredFrame.ts``:
the age of the pixels at display time, capture to sink, including every wait
in between (contracts.py promises exactly this on CensoredFrame.ts). It is
only meaningful live — offline, ts is video time, not clock time — so the
meter reports None there. fps counts every shown frame, fail-safe included:
it measures the loop's output rate, not the pipeline's success rate.
"""
from __future__ import annotations

from collections import deque


class LoopMeters:
    def __init__(self, live: bool, window: int = 30) -> None:
        self.live = live
        self._shown: deque[float] = deque(maxlen=window)   # write-clock times of recent frames
        self._latency: deque[float] = deque(maxlen=window) # seconds, fully-rendered frames only
        self._lat_sum = 0.0
        self._lat_count = 0

    def tick(self, censored_ts: float, now: float, real: bool) -> None:
        """Once per shown frame. ``real`` is False for fail-safe frames (stale ts)."""
        self._shown.append(now)
        if self.live and real:
            lat = now - censored_ts
            self._latency.append(lat)
            self._lat_sum += lat
            self._lat_count += 1

    @property
    def fps(self) -> float:
        """Rolling rate over the window; 0.0 until two frames were shown."""
        if len(self._shown) < 2:
            return 0.0
        span = self._shown[-1] - self._shown[0]
        return (len(self._shown) - 1) / span if span > 0 else 0.0

    @property
    def latency_ms(self) -> float | None:
        """Rolling average; None offline or before the first rendered frame."""
        if not self._latency:
            return None
        return 1000.0 * sum(self._latency) / len(self._latency)

    def avg_latency_ms(self) -> float | None:
        """Whole-run average, for RunStats; None offline."""
        if not self._lat_count:
            return None
        return 1000.0 * self._lat_sum / self._lat_count
