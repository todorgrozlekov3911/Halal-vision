"""Video sources (T05). Each Source hands out Frames; see core.contracts.Source.

FileSource has two modes:
    offline   every frame, as fast as the caller reads; ts = idx / fps (video time).
              Repeatable: the same file always gives the same Frames.
    realtime  behaves like a camera: frames are released at the video's own pace,
              frames the caller was too slow for are skipped, ts = capture clock.
              Used for daily development and latency testing.

CameraSource covers any V4L2 device (/dev/videoN): the laptop webcam, a USB
camera on the glasses, the phone via scrcpy. Always live; ts = capture clock.
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
        self.end_reason = ""      # why read() returned None
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
            self.end_reason = f"read() failed at frame {self._next_idx} of ~{self.frame_count}"
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
                    self.end_reason = f"grab() failed at frame {self._next_idx} of ~{self.frame_count}"
                    return None
                self._next_idx += 1
                self.skipped += 1
        return self._read_next(ts=time.perf_counter())


class CameraSource(Source):
    """A V4L2 camera device. Requests MJPG at the given size and rate; cameras
    grant what they can, so the real shape and rate are probed, not assumed."""

    def __init__(self, device: str | int, width: int = 1280, height: int = 720,
                 fps: float = 30.0) -> None:
        self.device = f"/dev/video{device}" if isinstance(device, int) else str(device)
        self._cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise OSError(f"OpenCV cannot open camera {self.device}")
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        ok, image = self._cap.read()        # probe what the camera actually granted
        if not ok:
            self._cap.release()
            raise OSError(f"camera {self.device} opened but delivers no frames")
        self.source_id = f"camera:{self.device}"
        self._shape = image.shape[:2]
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or fps
        self.end_reason = ""
        self._next_idx = 0

    # -- Source interface ---------------------------------------------------

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def is_live(self) -> bool:
        return True

    def read(self) -> Frame | None:
        ok, image = self._cap.read()       # blocks until the camera has a frame
        if not ok:
            self.end_reason = f"camera read() failed at frame {self._next_idx} (unplugged?)"
            return None
        frame = Frame(image=image, ts=time.perf_counter(), idx=self._next_idx,
                      source_id=self.source_id)
        self._next_idx += 1
        return frame

    def close(self) -> None:
        self._cap.release()
