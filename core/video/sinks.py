"""Video outputs (T07). Both accept only CensoredFrame (checked in contracts.Sink).

DisplaySink  an OpenCV window: the dev window now, the glasses (fullscreen) later.
FileSink     H.264 file through an ffmpeg pipe, written on its own thread so a slow
             encoder never stalls the main loop; copies the original audio back in.
"""
from __future__ import annotations

import queue
import subprocess
import tempfile
import threading
from fractions import Fraction
from pathlib import Path

import cv2

from core.contracts import CensoredFrame, Sink

_QUIT_KEYS = {ord("q"), 27}  # q, Esc


class DisplaySink(Sink):
    """Shows frames in a window. Sets ``quit_requested`` on q / Esc / window closed."""

    def __init__(self, window: str = "halal-vision", fullscreen: bool = False, max_width: int = 1280) -> None:
        self.window = window
        self.fullscreen = fullscreen
        self.max_width = max_width
        self.quit_requested = False
        self._opened = False

    def _open(self, height: int, width: int) -> None:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif width > self.max_width:  # scaled by the window, not by us: no extra copy
            cv2.resizeWindow(self.window, self.max_width, round(height * self.max_width / width))
        self._opened = True

    def _write(self, frame: CensoredFrame) -> None:
        if not self._opened:
            self._open(*frame.image.shape[:2])
        cv2.imshow(self.window, frame.image)
        key = cv2.waitKey(1) & 0xFF   # also lets the window repaint
        if key in _QUIT_KEYS or cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
            self.quit_requested = True

    def close(self) -> None:
        if self._opened:
            cv2.destroyWindow(self.window)
            cv2.waitKey(1)
            self._opened = False


class FileSink(Sink):
    """Encodes frames to ``path`` with ffmpeg (libx264) on a writer thread.

    block=True  (offline export) the main loop waits when the queue is full: no frame is lost.
    block=False (live recording) the oldest queued frame is dropped instead: the loop never waits.

    audio_from: a video whose audio is copied into the output unchanged. Only line-up-correct
    when every frame is written (offline mode); a realtime run skips frames, so leave it None.
    """

    def __init__(self, path: str | Path, fps: float, audio_from: str | Path | None = None,
                 block: bool = True, queue_size: int = 32) -> None:
        self.path = Path(path)
        self.fps = fps
        self.audio_from = Path(audio_from) if audio_from else None
        self.block = block
        self.dropped = 0
        self.written = 0
        self._queue: queue.Queue[CensoredFrame | None] = queue.Queue(maxsize=queue_size)
        self._shape: tuple[int, int] | None = None
        self._proc: subprocess.Popen | None = None
        self._log = tempfile.TemporaryFile()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def _start(self, height: int, width: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(Fraction(self.fps).limit_denominator(1001)),
               "-i", "-"]
        if self.audio_from:
            cmd += ["-i", str(self.audio_from), "-map", "0:v", "-map", "1:a?", "-c:a", "copy"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(self.path)]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._log)
        self._shape = (height, width)
        self._thread = threading.Thread(target=self._run, name="FileSink-writer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while (frame := self._queue.get()) is not None:
                self._proc.stdin.write(frame.image.tobytes())
                self.written += 1
        except BaseException as exc:  # broken pipe = ffmpeg died; surfaced on the next write/close
            self._error = exc

    def _write(self, frame: CensoredFrame) -> None:
        if self._error:
            raise RuntimeError(f"FileSink writer failed: {self._ffmpeg_log() or self._error}")
        h, w = frame.image.shape[:2]
        if self._proc is None:
            self._start(h, w)
        elif (h, w) != self._shape:
            raise ValueError(f"frame size changed from {self._shape} to {(h, w)}")
        if self.block:
            while True:  # wait for room, but notice if the writer died meanwhile
                try:
                    self._queue.put(frame, timeout=0.5)
                    return
                except queue.Full:
                    if self._error or not self._thread.is_alive():
                        raise RuntimeError(f"FileSink writer failed: {self._ffmpeg_log() or self._error}") from None
        while True:
            try:
                self._queue.put_nowait(frame)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    def close(self) -> None:
        if self._proc is None:
            return
        if self._thread.is_alive():
            self._queue.put(None)  # writer is alive, so it will make room
            self._thread.join()
        try:
            self._proc.stdin.close()
        except BrokenPipeError:
            pass
        code = self._proc.wait()
        self._proc = None
        if code != 0 or self._error:
            raise RuntimeError(f"ffmpeg failed (exit {code}): {self._ffmpeg_log()}")

    def _ffmpeg_log(self) -> str:
        self._log.seek(0)
        return self._log.read().decode(errors="replace").strip()[-2000:]
