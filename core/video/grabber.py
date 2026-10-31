"""Newest-frame grabber (T06). Sits between a live Source and the main loop.

A thread reads the source as fast as it delivers and keeps only the newest
frame (the "newest-frame slot" of the architecture). A slow main loop costs
frame rate, never latency: get() always returns the freshest frame the source
has produced, and every frame the loop was too slow for is dropped and counted.

Offline sources are refused: offline mode must process every frame, so the
main loop reads those directly and no grabber is involved.
"""
from __future__ import annotations

import threading

from core.contracts import Frame, Source


class Grabber:
    """Owns a live Source, reads it on its own thread; each frame is delivered
    at most once. A source error is re-raised from the next get()."""

    def __init__(self, source: Source) -> None:
        if not source.is_live:
            raise ValueError("Grabber is for live sources only; read offline sources directly")
        self.source = source
        self.received = 0        # frames read from the source
        self.dropped = 0         # frames overwritten before the main loop took them
        self._cond = threading.Condition()
        self._frame: Frame | None = None    # the slot: newest undelivered frame
        self._eof = False                   # source ended, or close() called
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="Grabber", daemon=True)
        self._thread.start()

    @property
    def delivered(self) -> int:
        """Frames the main loop actually got."""
        return self.received - self.dropped - (self._frame is not None)

    # -- grabber thread ------------------------------------------------------

    def _run(self) -> None:
        try:
            while True:
                frame = self.source.read()   # may block until the camera has one
                with self._cond:
                    if self._eof:            # close() was called while reading
                        return
                    if frame is None:        # realtime file ended
                        self._eof = True
                        self._cond.notify_all()
                        return
                    if self._frame is not None:
                        self.dropped += 1    # slot still occupied: overwrite
                    self._frame = frame
                    self.received += 1
                    self._cond.notify_all()
        except BaseException as exc:         # surfaced on the main loop's next get()
            with self._cond:
                self._error = exc
                self._cond.notify_all()

    # -- main loop side ------------------------------------------------------

    def get(self, timeout: float | None = None) -> Frame | None:
        """The newest frame not yet delivered; blocks until one arrives.

        Returns None at end of stream. Raises TimeoutError if no frame arrives
        within ``timeout`` seconds (a stalled camera: trigger the fail-safe).
        """
        with self._cond:
            if not self._cond.wait_for(
                lambda: self._error is not None or self._frame is not None or self._eof,
                timeout,
            ):
                raise TimeoutError(f"no frame from {self.source.source_id} in {timeout} s")
            if self._frame is not None:      # a frame that arrived before any error
                frame, self._frame = self._frame, None   # is still the freshest reality:
                return frame                 # deliver it; the error raises on the next call
            if self._error is not None:
                raise RuntimeError(f"source failed in grabber thread: {self._error}") from self._error
            return None                      # eof, and the slot is already empty

    def close(self) -> None:
        """Stop the thread and close the source. Idempotent."""
        with self._cond:
            self._eof = True
            self._cond.notify_all()
        self._thread.join(timeout=2.0)       # usually exits after its current read()
        self.source.close()                  # unblocks a read() stuck on dead hardware
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def __enter__(self) -> Grabber:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
