"""T06 acceptance: slow processing drops frames, latency stays bounded.

Runs under pytest, or standalone:  python -m tests.test_grabber
Needs a short clip; makes a synthetic one next to the system temp dir.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from core.contracts import Frame, Source
from core.video.grabber import Grabber
from core.video.sources import FileSource

FPS = 30.0
N_FRAMES = 75  # 2.5 s clip


def make_clip(path: Path, n: int = N_FRAMES, fps: float = FPS, size: tuple[int, int] = (320, 240)) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened(), "OpenCV cannot create the test clip"
    for i in range(n):
        img = np.full((size[1], size[0], 3), (i * 3) % 255, dtype=np.uint8)
        cv2.putText(img, str(i), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        writer.write(img)
    writer.release()
    return path


def _clip() -> Path:
    path = Path(tempfile.gettempdir()) / "halal_vision_test_clip.mp4"
    if not path.exists():
        make_clip(path)
    return path


def test_fast_consumer_gets_every_frame_once() -> None:
    with Grabber(FileSource(_clip(), mode="realtime")) as grabber:
        indices = []
        while (frame := grabber.get(timeout=2.0)) is not None:
            indices.append(frame.idx)
    assert grabber.dropped == 0, f"fast consumer must drop nothing, dropped {grabber.dropped}"
    assert indices == sorted(set(indices)), "a frame was delivered twice or out of order"
    assert len(indices) >= N_FRAMES * 0.9, f"got only {len(indices)}/{N_FRAMES} frames"


def test_slow_consumer_drops_frames_but_latency_stays_bounded() -> None:
    period = 1.0 / FPS
    max_age = 0.0
    delivered = 0
    with Grabber(FileSource(_clip(), mode="realtime")) as grabber:
        while True:
            try:
                frame = grabber.get(timeout=2.0)
            except TimeoutError:
                break
            if frame is None:
                break
            max_age = max(max_age, time.perf_counter() - frame.ts)
            delivered += 1
            time.sleep(3 * period)  # a main loop running 3x too slow
    assert grabber.dropped > 0, "slow consumer must drop frames"
    assert delivered < N_FRAMES * 0.6, f"delivered {delivered}/{N_FRAMES}, expected far fewer"
    # Bounded latency: every delivered frame is fresh, no matter how far behind we fell.
    assert max_age < 3 * period, f"stale frame delivered: {max_age * 1000:.0f} ms old"


def test_offline_source_is_refused() -> None:
    source = FileSource(_clip(), mode="offline")
    try:
        Grabber(source)
    except ValueError:
        pass
    else:
        raise AssertionError("Grabber accepted an offline source")
    finally:
        source.close()


class _ExplodingSource(Source):
    """A live source that delivers one frame, then dies like an unplugged camera."""

    def __init__(self) -> None:
        self.calls = 0

    def read(self) -> Frame | None:
        self.calls += 1
        if self.calls == 1:
            img = np.zeros((8, 8, 3), dtype=np.uint8)
            return Frame(image=img, ts=time.perf_counter(), idx=0, source_id="test:exploding")
        raise OSError("camera unplugged")

    @property
    def fps(self) -> float:
        return FPS

    @property
    def frame_shape(self) -> tuple[int, int]:
        return (8, 8)

    @property
    def is_live(self) -> bool:
        return True

    def close(self) -> None:
        pass


def test_source_error_reraised_in_main_thread() -> None:
    with Grabber(_ExplodingSource()) as grabber:
        assert grabber.get(timeout=2.0) is not None  # the good frame
        try:
            grabber.get(timeout=2.0)
        except RuntimeError as exc:
            assert "camera unplugged" in str(exc.__cause__)
        else:
            raise AssertionError("source error was swallowed")


if __name__ == "__main__":
    for test in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        test()
        print(f"PASS {test.__name__}")
    print("all grabber tests passed")