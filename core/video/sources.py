"""Video sources (T05). Each Source hands out Frames; see core.contracts.Source.

FileSource has two modes:
    offline   every frame, as fast as the caller reads; ts = idx / fps (video time).
              Repeatable: the same file always gives the same Frames.
    realtime  behaves like a camera: frames are released at the video's own pace,
              frames the caller was too slow for are skipped, ts = capture clock.
              Used for daily development and latency testing.

CameraSource is not written yet (rest of T05).
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2

from core.contracts import Frame, Source

MODES = ("offline", "realtime")


class FileSource(Source):
    def __init__(self, path: str | Path, mode: str = "offline") -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise OSError(f"OpenCV cannot open {self.path}")

        self.mode = mode
        self.source_id = f"file:{self.path.name}"
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        if self._fps <= 0:
            raise OSError(f"{self.path} reports no frame rate")
        self._shape = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))  # container estimate
        self.skipped = 0          # realtime only: frames dropped because the caller was slow
        self._next_idx = 0        # index of the next frame in the file
        self._t0: float | None = None  # realtime: clock time of frame 0

    # -- Source interface ---------------------------------------------------

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def is_live(self) -> bool:
        return self.mode == "realtime"

    def read(self) -> Frame | None:
        if self.mode == "offline":
            return self._read_next(ts=self._next_idx / self._fps)
        return self._read_realtime()

    def close(self) -> None:
        self._cap.release()

    # -- internals ------------------------------------------------------------

    def _read_next(self, ts: float) -> Frame | None:
        ok, image = self._cap.read()   # OpenCV allocates a new array per call (rule 3)
        if not ok:
            return None
        frame = Frame(image=image, ts=ts, idx=self._next_idx, source_id=self.source_id)
        self._next_idx += 1
        return frame

    def _read_realtime(self) -> Frame | None:
        now = time.perf_counter()
        if self._t0 is None:
            self._t0 = now
        # Which frame "is on air" right now, as if a camera were filming the video.
        due = int((now - self._t0) * self._fps)
        if due < self._next_idx:
            # Caller is ahead of the video: wait for the next frame, like a camera would.
            time.sleep((self._t0 + self._next_idx / self._fps) - now)
        else:
            # Caller is behind: skip the frames it missed (grab() decodes less than read()).
            while self._next_idx < due:
                if not self._cap.grab():
                    return None
                self._next_idx += 1
                self.skipped += 1
        return self._read_next(ts=time.perf_counter())
