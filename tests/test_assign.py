"""T17: face-to-subject assignment through the masks (synthetic people and faces, no models).

    .venv/bin/python -m pytest -q tests/test_assign.py
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from core.config_table import load_config
from core.contracts import Detections, FaceObservation, Frame, Track
from core.detect.detector import _mask_geometry
from core.identity.assign import FaceAssigner, head_region, look

GEO = _mask_geometry((1080, 1920), (512, 896), (128, 224))


def person_mask(rects):
    """A mask-space mask covering frame-pixel rectangles (x1, y1, x2, y2)."""
    h, w = 128, 224
    v, u = np.mgrid[0:h, 0:w]
    fx, fy = (u + 0.5 - GEO.pad_x) / GEO.scale, (v + 0.5 - GEO.pad_y) / GEO.scale
    m = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in rects:
        m[(fx >= x1) & (fx < x2) & (fy >= y1) & (fy < y2)] = 255
    return m


def body(x, top=200, w=200, h=700):
    """A standing person: head (60% of the width) on a body. Returns (box, rects)."""
    head = (x + 0.2 * w, top, x + 0.8 * w, top + 0.18 * h)
    trunk = (x, top + 0.18 * h, x + w, top + h)
    return np.array([x, top, x + w, top + h], np.float32), [head, trunk]


def scene(people):
    """people: list of (track_id, box, rects). Later people are in front: they hide earlier ones."""
    masks = [person_mask(rects) for _, _, rects in people]
    for k in range(len(masks)):                        # occlusion: a person is not visible behind nearer ones
        for nearer in masks[k + 1:]:
            masks[k][nearer > 0] = 0
    tracks = tuple(Track(tid, box, 0.9, k) for k, (tid, box, _) in enumerate(people))
    return Detections(0, 0.0, tracks, np.stack(masks) if masks else np.zeros((0, 128, 224), np.uint8), GEO)


def face(x1, y1, size=60):
    box = np.array([x1, y1, x1 + size, y1 + size], np.float32)
    s = size
    kps = np.array([[x1 + 0.3 * s, y1 + 0.38 * s], [x1 + 0.7 * s, y1 + 0.38 * s], [x1 + 0.5 * s, y1 + 0.58 * s],
                    [x1 + 0.35 * s, y1 + 0.78 * s], [x1 + 0.65 * s, y1 + 0.78 * s]], np.float32)
    return FaceObservation(bbox=box, landmarks=kps, head_pose=(0.0, 0.0, 0.0), det_score=0.9, quality=0.9,
                           embedding=np.ones(512, np.float32) / np.sqrt(512), embedding_model="test",
                           p_female=0.5, age=30.0, ts=0.0, frame_idx=0)


@pytest.fixture(scope="module")
def assigner():
    return FaceAssigner(load_config("intel"))


def reasons(assigner, faces, dets):
    return [(a.face.track_id, a.reason) for a in assigner.explain(faces, dets)]


def test_a_face_on_one_persons_head_is_theirs(assigner):
    box, rects = body(400)
    dets = scene([(7, box, rects)])
    assert reasons(assigner, [face(470, 220)], dets) == [(7, "assigned")]


def test_overlapping_people_get_their_own_faces_never_each_others(assigner):
    back_box, back = body(400, top=150)
    front_box, front = body(520, top=260)                # in front, overlapping the back person's right side
    dets = scene([(1, back_box, back), (2, front_box, front)])
    got = reasons(assigner, [face(460, 165), face(590, 275)], dets)
    assert got == [(1, "assigned"), (2, "assigned")]


def test_a_face_across_two_masks_is_ambiguous(assigner):
    a_box, a = body(300)
    b_box, b = body(500)                                  # heads 120 px apart: a face between them touches both
    dets = scene([(1, a_box, a), (2, b_box, b)])
    a_box2, a2 = body(300, w=230)
    dets = scene([(1, a_box2, [(340, 200, 470, 330)] + a2[1:]), (2, b_box, [(470, 200, 600, 330)] + b[1:])])
    assert reasons(assigner, [face(440, 230)], dets)[0] == (None, "shared")


def test_an_id_less_detection_is_a_competitor_and_never_an_owner(assigner):
    box, rects = body(400)
    assert reasons(assigner, [face(470, 220)], scene([(None, box, rects)])) == [(None, "id-less")]


def test_a_face_low_on_the_body_is_not_that_persons(assigner):
    box, rects = body(400)                                # e.g. a child carried in their arms
    dets = scene([(7, box, rects)])
    assert reasons(assigner, [face(470, 520)], dets) == [(None, "not-head")]


def test_small_faces_give_nothing(assigner):
    box, rects = body(400)
    dets = scene([(7, box, rects)])
    assert reasons(assigner, [face(485, 230, size=assigner.min_face_px - 2)], dets) == [(None, "small")]


def test_a_face_off_every_mask_is_unowned(assigner):
    box, rects = body(400)
    dets = scene([(7, box, rects)])
    assert reasons(assigner, [face(1200, 220)], dets) == [(None, "unowned")]


def test_eyes_off_the_mask_are_not_enough(assigner):
    box, _ = body(400)
    # most of the face area is on the mask, but the eyes sit above it (a hood, or bad segmentation)
    dets = scene([(7, box, [(440, 250, 560, 330), (400, 330, 600, 900)])])
    f = face(470, 220, size=80)
    kps = np.array(f.landmarks)
    kps[:2, 1] = 222                                      # eyes at the very top of the face box
    f = dataclasses.replace(f, landmarks=kps)
    assert reasons(assigner, [f], dets) == [(None, "core")]


def test_two_faces_for_one_person_give_nothing(assigner):
    box, rects = body(400, w=300)
    dets = scene([(7, box, rects)])
    got = reasons(assigner, [face(480, 215), face(560, 215)], dets)
    assert got == [(None, "two-faces"), (None, "two-faces")]


def test_assign_keeps_everything_but_the_track_id(assigner):
    box, rects = body(400)
    f = face(470, 220)
    out = assigner.assign([f], scene([(7, box, rects)]))[0]
    assert out.track_id == 7 and f.track_id is None
    np.testing.assert_array_equal(out.bbox, f.bbox)
    assert out.embedding is not f.embedding or np.array_equal(out.embedding, f.embedding)


def test_look_searches_the_head_region_and_returns_only_this_persons_face(assigner):
    a_box, a = body(300)
    b_box, b = body(700)
    dets = scene([(1, a_box, a), (2, b_box, b)])
    frame = Frame(np.zeros((1080, 1920, 3), np.uint8), 0.0, 0, "test")

    class Job:
        track_id, track, detections = 2, dets.tracks[1], dets
    Job.frame = frame

    class Analyzer:                                       # stands in for FaceAnalyzer: two faces in the frame
        region = None

        def analyze(self, image, ts, idx, region=None):
            Analyzer.region = region
            return [face(370, 220), face(770, 220)]

    got = look(Job, Analyzer(), assigner)
    assert got is not None and got.track_id == 2 and got.bbox[0] == 770
    x1, y1, x2, y2 = Analyzer.region
    assert x1 < 700 and x2 > 900 and y2 < 200 + 0.6 * 700      # around the top of person 2's box


def test_head_region_stays_in_the_frame():
    t = Track(1, np.array([1850, 10, 1920, 400], np.float32), 0.9, 0)
    x1, y1, x2, y2 = head_region(t, 0.35, (1080, 1920))
    assert 0 <= x1 < x2 <= 1920 and 0 <= y1 < y2 <= 1080
