"""RF-DETR-Seg detector: post-processing unit tests (no model), plus one test on real frames.

Runs under pytest:  .venv/bin/python -m pytest -q tests/test_rf_detr.py

The real-frame test runs the nano model on the CPU through PyTorch (works on every machine,
~0.5 s a frame) and is skipped when the test clip is not on disk (clips/ is gitignored) or
rfdetr is not installed.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.config_table import load_config
from core.contracts import MASK_CUT, Detections, Frame, Track
from core.detect.detector import UltralyticsDetector, make_detector
from core.detect.rf_detr import PERSON, people_from_outputs, preprocess

CLIP = Path(__file__).resolve().parents[1] / "clips" / "berlin_people_crossing.mp4"


def _outputs(people: list[tuple[float, tuple[float, float, float, float]]], q: int = 5, m: int = 78):
    """Raw-output arrays with the given (person score, relative cx cy w h box) queries first;
    each person's mask logits are +8 inside their box and -8 elsewhere."""
    boxes = np.full((q, 4), 0.5, np.float32)
    logits = np.full((q, 91), -8.0, np.float32)
    masks = np.full((q, m, m), -8.0, np.float32)
    for i, (score, (cx, cy, w, h)) in enumerate(people):
        boxes[i] = cx, cy, w, h
        logits[i, PERSON] = np.log(score / (1 - score))
        x1, x2 = int((cx - w / 2) * m), int((cx + w / 2) * m)
        y1, y2 = int((cy - h / 2) * m), int((cy + h / 2) * m)
        masks[i, y1:y2, x1:x2] = 8.0
    return boxes, logits, masks


def test_boxes_in_frame_pixels_sorted_and_thresholded():
    boxes, logits, masks = _outputs([(0.4, (0.25, 0.5, 0.1, 0.5)), (0.9, (0.75, 0.5, 0.2, 0.6)),
                                     (0.05, (0.5, 0.5, 0.1, 0.1))])
    people, soft, g = people_from_outputs(boxes, logits, masks, (720, 1280), conf=0.1)
    assert people.shape == (2, 6) and soft.shape[0] == 2          # the 0.05 query is below conf
    np.testing.assert_allclose(people[:, 4], [0.9, 0.4], atol=1e-5)   # highest score first
    np.testing.assert_allclose(people[0, :4], [832, 144, 1088, 576], atol=1e-3)
    assert set(people[:, 5]) == {0.0}


def test_masks_are_widened_to_the_frame_shape_without_padding():
    boxes, logits, masks = _outputs([(0.9, (0.5, 0.5, 0.5, 0.5))])
    people, soft, g = people_from_outputs(boxes, logits, masks, (720, 1280), conf=0.1)
    assert soft.shape == (1, 78, 139)                              # 78 rows, 78 * 16/9 columns
    assert soft.dtype == np.uint8 and len(np.unique(soft)) > 2     # soft certainty, not yes/no
    assert (g.pad_x, g.pad_y) == (0.0, 0.0)
    assert g.scale == pytest.approx(78 / 720)
    assert g.content_box() == (0, 0, 139, 78)
    # The mask's covered area maps back onto the person's box in frame pixels.
    ys, xs = np.nonzero(soft[0] >= MASK_CUT)
    fx, fy = (xs.min() + xs.max() + 1) / 2 / g.scale, (ys.min() + ys.max() + 1) / 2 / g.scale
    assert fx == pytest.approx(640, abs=12) and fy == pytest.approx(360, abs=12)


def test_mask_is_cut_to_the_box_and_empty_masks_are_dropped():
    boxes, logits, masks = _outputs([(0.9, (0.3, 0.5, 0.2, 0.4)), (0.8, (0.7, 0.5, 0.2, 0.4))])
    masks[0, :, :] = 8.0                                          # person 0's mask spills over the frame
    masks[1, :, :] = -8.0                                         # person 1's mask is empty
    people, soft, g = people_from_outputs(boxes, logits, masks, (720, 1280), conf=0.1)
    assert len(people) == 1 and people[0, 4] == pytest.approx(0.9)
    ys, xs = np.nonzero(soft[0])
    x1, y1, x2, y2 = people[0, :4] * g.scale
    assert xs.min() >= np.floor(x1) and xs.max() < np.ceil(x2)
    assert ys.min() >= np.floor(y1) and ys.max() < np.ceil(y2)


def test_nobody_found():
    boxes, logits, masks = _outputs([])
    people, soft, g = people_from_outputs(boxes, logits, masks, (1080, 1920), conf=0.1)
    assert people.shape == (0, 6) and soft.shape == (0, 78, 139)


def test_preprocess_shape_and_normalization():
    x = preprocess(np.full((720, 1280, 3), 255, np.uint8), 312)
    assert x.shape == (1, 3, 312, 312) and x.dtype == np.float32 and x.flags.c_contiguous
    np.testing.assert_allclose(x[0, :, 0, 0], (1 - np.array([0.485, 0.456, 0.406])) / [0.229, 0.224, 0.225], atol=1e-5)


def test_factory_picks_the_class_from_the_model_name():
    cfg = load_config("intel", ["detector.device=cpu"])
    assert isinstance(make_detector(cfg, fps=30, model="yolo26n-seg.pt"), UltralyticsDetector)
    with pytest.raises(ValueError, match="unknown RF-DETR model"):
        make_detector(cfg, fps=30, model="rf-detr-seg-tiny")


# -- real frames -------------------------------------------------------------------

def test_people_found_on_real_frames():
    pytest.importorskip("rfdetr")
    if not CLIP.is_file():
        pytest.skip(f"test clip missing: {CLIP}")
    cap = cv2.VideoCapture(str(CLIP))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1750)                        # four people, close to the camera
    frames = [cap.read()[1] for _ in range(6)]
    cap.release()

    cfg = load_config("intel", ["detector.device=cpu"])
    det = make_detector(cfg, fps=30, model="rf-detr-seg-nano")
    results: list[Detections] = [det.detect(Frame(image=img.copy(), ts=i / 15, idx=i, source_id="test"))
                                 for i, img in enumerate(frames)]
    first, last = results[0], results[-1]
    assert len(first.tracks) >= 3
    assert all(t.track_id is None for t in first.tracks)            # same confirm delay as YOLO
    assert sum(t.track_id is not None for t in last.tracks) >= 3
    h, w = frames[0].shape[:2]
    assert last.masks.shape[1:] == (78, round(78 * w / h))
    for t in last.tracks:
        _mask_centre_inside_box(last, t)


def _mask_centre_inside_box(dets: Detections, t: Track) -> None:
    ys, xs = np.nonzero(dets.mask_of(t) >= MASK_CUT)
    assert len(xs), "every returned person has a non-empty mask"
    cx, cy = (np.array([xs.mean(), ys.mean()]) - [dets.geometry.pad_x, dets.geometry.pad_y]) / dets.geometry.scale
    x1, y1, x2, y2 = t.bbox
    assert x1 <= cx <= x2 and y1 <= cy <= y2, "mask centre lies inside the person's box"
