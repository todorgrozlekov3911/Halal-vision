"""T16: face analysis wrapper.

The model tests use InsightFace's own sample photos (t1.jpg: a group of six; Tom_Hanks_54745.png:
one man) and are skipped when the face models are not downloaded (models/insightface/, gitignored).

    .venv/bin/python -m pytest -q tests/test_faces.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import insightface
import numpy as np
import pytest

from core.config_table import load_config
from core.identity import faces
from core.identity.faces import (FaceAnalyzer, gender_probability, head_pose, pose_quality, sharpness_quality,
                                 size_quality)

SAMPLES = Path(insightface.__file__).resolve().parent / "data" / "images"   # sample photos shipped with InsightFace
FRONTAL = np.array([[40, 50], [80, 50], [60, 72], [45, 90], [75, 90]], np.float32)


# -- no model ------------------------------------------------------------------------

def test_gender_probability_is_a_softmax_with_female_first():
    assert gender_probability(np.array([2.0, -1.0])) > 0.9
    assert gender_probability(np.array([-1.0, 2.0])) < 0.1
    assert gender_probability(np.array([0.3, 0.3])) == pytest.approx(0.5)


def test_head_pose_of_a_frontal_face_is_near_zero():
    pitch, yaw, roll = head_pose(FRONTAL)
    assert abs(yaw) < 5 and abs(roll) < 1 and abs(pitch) < 10


def test_head_pose_turned_face_has_yaw_with_the_nose_direction():
    right = FRONTAL.copy(); right[2, 0] += 15
    left = FRONTAL.copy(); left[2, 0] -= 15
    assert head_pose(right)[1] > 30 and head_pose(left)[1] < -30


def test_head_pose_roll_follows_the_eye_line_and_does_not_leak_into_yaw():
    angle = np.radians(20)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    tilted = (FRONTAL - FRONTAL[2]) @ rot.T + FRONTAL[2]
    pitch, yaw, roll = head_pose(tilted)
    assert roll == pytest.approx(20, abs=0.5) and abs(yaw) < 5


def test_head_pose_looking_down_is_positive_pitch():
    down = FRONTAL.copy(); down[2, 1] += 10      # nose closer to the mouth
    up = FRONTAL.copy(); up[2, 1] -= 10
    assert head_pose(down)[0] > 10 and head_pose(up)[0] < -10


def test_quality_parts():
    assert size_quality(32, 32) == 0 and size_quality(200, 32) == 1 and 0 < size_quality(60, 32) < 1
    assert pose_quality((0, 0, 30)) == 1 and pose_quality((0, 60, 0)) == 0
    flat = np.full((100, 100, 3), 128, np.uint8)
    noisy = np.random.default_rng(0).integers(0, 256, (100, 100, 3), dtype=np.uint8)
    box = np.array([10, 10, 90, 90], np.float32)
    assert sharpness_quality(flat, box) == 0 and sharpness_quality(noisy, box) == 1


# -- with the models ------------------------------------------------------------------

def _analyzer(model: str) -> FaceAnalyzer:
    if not (faces.ROOT / "models" / model / "genderage.onnx").is_file():
        pytest.skip(f"{model} not downloaded")
    return FaceAnalyzer(load_config("intel", [f"identity.face_model={model}"]))


@pytest.fixture(scope="module", params=["buffalo_s", "buffalo_l"])
def analyzer(request):
    return _analyzer(request.param)


def test_every_face_in_the_group_photo_is_described(analyzer):
    image = cv2.imread(str(SAMPLES / "t1.jpg"))
    found = analyzer.analyze(image, ts=1.5, frame_idx=7)
    assert len(found) == 6
    for f in found:
        assert f.bbox.shape == (4,) and f.landmarks.shape == (5, 2)
        assert np.all((f.landmarks >= f.bbox[:2] - 5) & (f.landmarks <= f.bbox[2:] + 5))   # inside the face
        assert np.linalg.norm(f.embedding) == pytest.approx(1, abs=1e-4)
        assert 0 <= f.p_female <= 1 and 0 < f.age < 100 and 0 <= f.quality <= 1
        assert f.embedding_model == analyzer.model_name and f.ts == 1.5 and f.frame_idx == 7
        assert f.track_id is None
    heights = [f.bbox[3] - f.bbox[1] for f in found]
    assert heights == sorted(heights, reverse=True)
    # t1.jpg: three women, three men; the model's call should match its own argmax, and split 3/3
    assert sum(f.p_female > 0.5 for f in found) == 3


def test_a_region_gives_frame_coordinates(analyzer):
    image = cv2.imread(str(SAMPLES / "t1.jpg"))
    whole = analyzer.analyze(image, 0.0, 0)
    target = whole[0]
    x1, y1, x2, y2 = target.bbox
    pad = 60
    region = (x1 - pad, y1 - pad, x2 + pad, y2 + pad)
    inside = analyzer.analyze(image, 0.0, 0, region=region)
    assert len(inside) >= 1
    best = max(inside, key=lambda f: float(f.embedding @ target.embedding))
    assert np.abs(best.bbox - target.bbox).max() < 0.1 * (y2 - y1)
    assert float(best.embedding @ target.embedding) > 0.8                       # same face, same fingerprint


def test_the_same_person_matches_and_others_do_not(analyzer):
    hanks = cv2.imread(str(SAMPLES / "Tom_Hanks_54745.png"))
    a = analyzer.analyze(hanks, 0.0, 0)[0]
    b = analyzer.analyze(cv2.flip(hanks, 1), 0.0, 0)[0]       # mirrored: same person, different pixels
    others = analyzer.analyze(cv2.imread(str(SAMPLES / "t1.jpg")), 0.0, 0)
    assert a.p_female < 0.5
    same = float(a.embedding @ b.embedding)
    assert same > 0.5 and all(float(a.embedding @ o.embedding) < 0.3 for o in others)


def test_faces_below_the_minimum_size_are_skipped(analyzer):
    image = cv2.imread(str(SAMPLES / "t1.jpg"))
    small = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    assert all(f.bbox[3] - f.bbox[1] >= analyzer.min_face_px for f in analyzer.analyze(small, 0.0, 0))


def test_a_region_too_small_for_a_face_finds_nothing(analyzer):
    image = cv2.imread(str(SAMPLES / "t1.jpg"))
    assert analyzer.analyze(image, 0.0, 0, region=(10, 10, 30, 30)) == []
