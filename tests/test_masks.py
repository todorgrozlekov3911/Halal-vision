"""T10: mask pipeline. Who is covered, growth, and that masks land on the right frame pixels.

    .venv/bin/python -m pytest -q tests/test_masks.py
"""
from __future__ import annotations

import numpy as np
import pytest

from core.contracts import (CONCEAL_NIQAB, PASS, Action, ActionKind, Decision, Detections, Track,
                            TrackState)
from core.detect.detector import _mask_geometry
from core.detect.masks import MaskPipeline, covered_tracks, frame_region, is_covered, to_frame_resolution

GEO_1080 = _mask_geometry((1080, 1920), (384, 640), (96, 160))   # scale 1/12, 3-row bars top and bottom


def track(track_id, idx=0, bbox=(480, 120, 732, 600)):
    return Track(track_id=track_id, bbox=np.array(bbox, np.float32), score=0.9, mask_idx=idx)


def decision(track_id, action):
    return Decision(track_id, action, TrackState.PASSED, rule="test", confidence=1.0)


def rect_mask(rows, cols, shape=(96, 160)):
    m = np.zeros(shape, np.uint8)
    m[rows[0]:rows[1], cols[0]:cols[1]] = 255
    return m


def covered_box(full):
    ys, xs = np.nonzero(full)
    return xs.min(), ys.min(), xs.max(), ys.max()


# -- who is covered ---------------------------------------------------------------

def test_cover_rule_fails_closed():
    decorate = Action(ActionKind.DECORATE, asset="hat.png")
    assert is_covered(track(None), {})                               # ID-less
    assert is_covered(track(7), {})                                  # no decision yet
    assert is_covered(track(7), {7: decision(7, CONCEAL_NIQAB)})
    assert not is_covered(track(7), {7: decision(7, PASS)})
    assert not is_covered(track(7), {7: decision(7, decorate)})      # visible, with an overlay (later)
    assert is_covered(track(7), {8: decision(8, PASS)})              # someone else's pass


def test_covered_tracks_keeps_only_covered_people():
    tracks = (track(1, 0), track(2, 1), track(None, 2))
    dets = Detections(0, 0.0, tracks, np.zeros((3, 96, 160), np.uint8), GEO_1080)
    assert [t.track_id for t in covered_tracks(dets, {1: decision(1, PASS)})] == [2, None]


# -- mask space -> frame pixels -------------------------------------------------------

def test_full_resolution_mask_lands_on_the_right_pixels():
    # mask rows 13..52, cols 40..60 -> frame x 480..731, y (13-3)*12=120 .. (53-3)*12-1=599
    full = to_frame_resolution(rect_mask((13, 53), (40, 61)), GEO_1080)
    assert full.shape == (1080, 1920) and full.dtype == np.uint8
    assert set(np.unique(full)) == {0, 255}
    assert covered_box(full) == (480, 120, 731, 599)


def test_letterbox_bars_are_removed_with_fractional_padding():
    # 416x416 input: 22.75 mask rows of bar on top. A mask starting right below the bar
    # starts at the very top of the frame, not 0.75 mask rows (9 px) lower or higher.
    geo = _mask_geometry((1080, 1920), (416, 416), (104, 104))
    full = to_frame_resolution(rect_mask((23, 50), (10, 30), (104, 104)), geo)
    x1, y1, x2, y2 = covered_box(full)
    s = geo.scale
    assert y1 == pytest.approx((23 - geo.pad_y) / s, abs=1.5)
    assert y2 == pytest.approx((50 - geo.pad_y) / s - 1, abs=1.5)
    assert x1 == pytest.approx(10 / s, abs=1.5) and x2 == pytest.approx(30 / s - 1, abs=1.5)


def test_person_at_the_frame_edge_stays_covered_to_the_edge():
    full = to_frame_resolution(rect_mask((3, 93), (0, 10)), GEO_1080)   # touches left, top and bottom
    x1, y1, _, y2 = covered_box(full)
    assert (x1, y1, y2) == (0, 0, 1079)


# -- growth and merging ------------------------------------------------------------

def test_growth_is_a_share_of_the_person_with_a_floor():
    p = MaskPipeline(grow_fraction=0.04, grow_min_fraction=0.01)
    assert p.grow_radius(track(1, bbox=(0, 0, 100, 900)), GEO_1080) == 3   # 36 px = 3 mask px
    assert p.grow_radius(track(1, bbox=(0, 0, 10, 50)), GEO_1080) == 1     # floor: 10.8 px -> 1 mask px
    assert MaskPipeline(0, 0).grow_radius(track(1), GEO_1080) == 0


def test_grown_mask_is_bigger_by_the_radius():
    dets = Detections(0, 0.0, (track(1, bbox=(480, 120, 732, 900)),),
                      rect_mask((13, 53), (40, 61))[None], GEO_1080)
    plain = MaskPipeline(0, 0).cover_mask(dets, dets.tracks).full((1080, 1920))
    grown = MaskPipeline(0.04, 0).cover_mask(dets, dets.tracks).full((1080, 1920))          # 780 px tall -> 31.2 px -> 3 mask px
    assert grown[plain > 0].all()                                          # never smaller
    px1, _, px2, _ = covered_box(plain)
    gx1, _, gx2, _ = covered_box(grown)
    assert (px1 - gx1, gx2 - px2) == (36, 36)


def test_merge_covers_everyone_and_nobody_gives_none():
    masks = np.stack([rect_mask((10, 20), (10, 20)), rect_mask((50, 60), (100, 110))])
    dets = Detections(0, 0.0, (track(1, 0), track(2, 1)), masks, GEO_1080)
    p = MaskPipeline(0, 0)
    both = p.cover_mask(dets, dets.tracks).full((1080, 1920))
    assert both[(15 - 3) * 12, 15 * 12] and both[(55 - 3) * 12, 105 * 12]
    only_second = p.cover_mask(dets, dets.tracks[1:]).full((1080, 1920))
    assert not only_second[(15 - 3) * 12, 15 * 12]
    assert p.cover_mask(dets, []) is None
    assert dets.masks.flags.writeable is False                           # inputs untouched (read-only)


def test_region_only_gives_the_same_pixels_as_the_whole_frame():
    small = np.zeros((96, 160), np.uint8)
    small[13:53, 40:61] = 255
    small[60:70, 0:5] = 255                                               # second blob at the frame edge
    region = frame_region(small, GEO_1080)
    whole = to_frame_resolution(small, GEO_1080)
    x1, y1, x2, y2 = region
    np.testing.assert_array_equal(to_frame_resolution(small, GEO_1080, region), whole[y1:y2, x1:x2])
    outside = whole.copy()
    outside[y1:y2, x1:x2] = 0
    assert not outside.any()                                              # nothing covered outside the region
    assert frame_region(np.zeros((96, 160), np.uint8), GEO_1080) is None


def test_soft_masks_are_cut_after_enlarging():
    # A soft edge (certainty falling 255 -> 0 across one mask pixel) is cut between mask pixels,
    # not on the mask-pixel grid: the cut moves when the certainty moves.
    m = np.zeros((96, 160), np.uint8)
    m[13:53, 40:60] = 255
    m[13:53, 60] = 200                                                    # column 60 mostly the person
    wide = covered_box(to_frame_resolution(m, GEO_1080))[2]
    m[13:53, 60] = 140                                                    # column 60 barely the person
    narrow = covered_box(to_frame_resolution(m, GEO_1080))[2]
    assert 3 <= wide - narrow < 12                                       # moved, by less than one 12-px mask pixel


def test_edge_fade_only_adds_cover_outside_the_full_cover():
    dets = Detections(0, 0.0, (track(1, bbox=(480, 120, 732, 900)),),
                      rect_mask((13, 53), (40, 61))[None], GEO_1080)
    hard = MaskPipeline(0.04, 0, edge_fade=0).cover_mask(dets, dets.tracks).full((1080, 1920))
    soft = MaskPipeline(0.04, 0, edge_fade=64).cover_mask(dets, dets.tracks).full((1080, 1920))
    np.testing.assert_array_equal(soft == 255, hard == 255)              # full cover unchanged
    fade = (soft > 0) & (soft < 255)
    assert fade.any() and not (fade & (hard == 255)).any()               # fade lies outside it


def test_growth_rounding_never_uncovers_the_person():
    m = np.zeros((96, 160), np.uint8)
    m[20:80, 50] = 255                                                    # a one-mask-pixel-wide limb
    dets = Detections(0, 0.0, (track(1, bbox=(0, 0, 10, 50)),), m[None], GEO_1080)
    full = MaskPipeline(0.0, 0.01).cover_mask(dets, dets.tracks).full((1080, 1920))
    raw = to_frame_resolution(m, GEO_1080)
    assert (full[raw == 255] == 255).all()
