"""T08: detector adapter. Fast unit tests, plus one test on a real frame.

Runs under pytest:  .venv/bin/python -m pytest -q tests/test_detector.py

The real-frame test runs the model on the CPU (works on every machine) and is
skipped when the test clip is not on disk (clips/ is gitignored).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.config_table import load_config
from core.contracts import MASK_CUT, Frame
from core.detect.detector import UltralyticsDetector, _imgsz, _mask_geometry, openvino_device, runs_on
from core.detect.masks import outlines_in_frame

CLIP = Path(__file__).resolve().parents[1] / "clips" / "berlin_people_crossing.mp4"


# -- frame <-> mask mapping ----------------------------------------------------

def test_geometry_video_shaped_input():
    # 1920x1080 into a 384x640 model input: scale 1/3, 12 px bars top and bottom; masks are 1/4 of that.
    g = _mask_geometry((1080, 1920), (384, 640), (96, 160))
    assert g.scale == pytest.approx(1 / 12)
    assert (g.pad_x, g.pad_y) == (0.0, 3.0)
    assert g.content_box() == (0, 3, 160, 93)
    corners = g.frame_to_mask(np.array([[0, 0], [1920, 1080]]))
    np.testing.assert_allclose(corners, [[0, 3], [160, 93]], atol=1e-4)


def test_geometry_square_input():
    # 416x416 square: the picture is 416x234, so 91 px bars (Ultralytics rounds 90.9 down
    # on top); in the 104x104 mask grid that is 22.75.
    g = _mask_geometry((1080, 1920), (416, 416), (104, 104))
    assert g.scale == pytest.approx(416 / 1920 / 4)
    assert (g.pad_x, g.pad_y) == (0.0, 22.75)


def test_outlines_map_back_to_frame_pixels():
    g = _mask_geometry((1080, 1920), (384, 640), (96, 160))
    mask = np.zeros((96, 160), np.uint8)
    mask[13:53, 40:61] = 255                         # mask rows 13..52, cols 40..60
    (outline,) = outlines_in_frame(mask, g)
    xs, ys = outline[:, 0, 0], outline[:, 0, 1]
    # frame = (mask - pad) / scale = (mask - pad) * 12
    assert (xs.min(), xs.max()) == (480, 720)
    assert (ys.min(), ys.max()) == (120, 588)


# -- config handling -------------------------------------------------------------

def test_imgsz_accepts_square_and_height_width():
    assert _imgsz(640) == 640
    assert _imgsz([384, 640]) == (384, 640)


@pytest.mark.parametrize("bad", [400, [380, 640], [384, 650]])
def test_imgsz_must_be_multiple_of_32(bad):
    with pytest.raises(ValueError, match="multiple"):
        _imgsz(bad)


def test_int8_on_npu_is_refused():
    # Measured: INT8 on the NPU found nobody far away -> people shown uncensored.
    cfg = load_config("intel", ["detector.device=intel:npu", "detector.int8=true"])
    with pytest.raises(ValueError, match="int8"):
        UltralyticsDetector(cfg, fps=15)


# -- new tracks stay ID-less for confirm_track_seconds ------------------------------

def _bare_detector(confirm_s: float = 0.1, forget_s: float = 2.5) -> UltralyticsDetector:
    det = UltralyticsDetector.__new__(UltralyticsDetector)   # no model needed for this logic
    det.confirm_seconds, det._forget_after, det._seen = confirm_s, forget_s, {}
    return det


def test_new_track_gets_its_id_after_confirm_time():
    det = _bare_detector()
    assert det._confirmed({0: 7}, now=0.00) == {}           # first sight: ID-less
    assert det._confirmed({0: 7}, now=0.05) == {}           # still too young
    assert det._confirmed({2: 7}, now=0.10) == {2: 7}       # old enough (detection index may change)


def test_confirm_time_is_seconds_not_frames():
    det = _bare_detector()
    det._confirmed({0: 3}, now=10.0)
    assert det._confirmed({0: 3}, now=10.09) == {}          # many frames at a high fps: still young
    assert det._confirmed({0: 3}, now=10.11) == {0: 3}


def test_forgotten_track_starts_over():
    det = _bare_detector()
    det._confirmed({0: 5}, now=0.0)
    det._confirmed({0: 5}, now=0.2)
    det._confirmed({}, now=3.0)                              # unseen longer than forget_s
    assert 5 not in det._seen


# -- real frames -------------------------------------------------------------------

@pytest.fixture(scope="module")
def street_frames() -> list[np.ndarray]:
    if not CLIP.is_file():
        pytest.skip(f"test clip missing: {CLIP}")
    cap = cv2.VideoCapture(str(CLIP))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1750)                  # four people, close to the camera
    frames = [cap.read()[1] for _ in range(10)]
    cap.release()
    return frames


def test_people_found_id_less_first_then_tracked(street_frames):
    cfg = load_config("intel", ["detector.device=cpu", "detector.int8=false"])
    det = UltralyticsDetector(cfg, fps=30)
    results = [det.detect(Frame(image=img.copy(), ts=i / 30, idx=i, source_id="test"))
               for i, img in enumerate(street_frames)]

    first, last = results[0], results[-1]
    assert len(first.tracks) >= 3
    assert all(t.track_id is None for t in first.tracks)      # new tracks are not trusted yet
    assert sum(t.track_id is not None for t in last.tracks) >= 3   # 0.3 s later they are

    in_h, in_w = cfg.detector.imgsz                           # masks are 1/4 of the model input
    assert last.masks.shape == (len(last.tracks), in_h // 4, in_w // 4)
    assert last.masks.dtype == np.uint8                       # soft: 0..255 certainty
    assert len(np.unique(last.masks)) > 2, "masks keep the model's certainty, not just yes/no"
    for t in last.tracks:
        ys, xs = np.nonzero(last.mask_of(t) >= MASK_CUT)
        assert len(xs), "every returned person has a non-empty mask"
        cx, cy = (np.array([xs.mean(), ys.mean()]) - [last.geometry.pad_x, last.geometry.pad_y]) / last.geometry.scale
        x1, y1, x2, y2 = t.bbox
        assert x1 <= cx <= x2 and y1 <= cy <= y2, "mask centre lies inside the person's box"


# -- which OpenVINO device really runs the model ---------------------------------------

@pytest.mark.parametrize("available, device, expected", [
    (["CPU", "GPU", "NPU"], "intel:npu", "NPU"),
    (["CPU", "GPU"], "intel:npu", "AUTO"),         # no render group: the NPU is invisible
    (["CPU"], "intel:gpu", "CPU"),
    (["CPU", "GPU.0", "GPU.1"], "intel:gpu", "GPU"),
])
def test_openvino_device_falls_back_like_ultralytics(monkeypatch, available, device, expected):
    import openvino as ov
    monkeypatch.setattr(ov, "Core", lambda: type("C", (), {"available_devices": available})())
    assert openvino_device(device) == expected


def test_models_that_break_auto_fall_back_to_the_gpu(monkeypatch):
    import openvino as ov
    monkeypatch.setattr(ov, "Core", lambda: type("C", (), {"available_devices": ["CPU", "GPU"]})())
    assert openvino_device("intel:npu", auto_ok=False) == "GPU"
    monkeypatch.setattr(ov, "Core", lambda: type("C", (), {"available_devices": ["CPU"]})())
    assert openvino_device("intel:npu", auto_ok=False) == "CPU"


def test_runs_on_says_when_the_npu_is_missing():
    assert runs_on("intel:npu", "NPU") == "NPU"
    assert runs_on("intel:npu", "AUTO") == "AUTO: NPU unavailable (no render group)"
    assert runs_on("intel:npu", "GPU") == "GPU: NPU unavailable (no render group)"
    assert runs_on("cuda:0", None) == "cuda:0"
