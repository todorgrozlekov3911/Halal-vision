"""T11 (+ the mask hold of T12): renderer.

Drawing tests run without mask smoothing, so they test the drawing alone; the memory tests
use the configured rules (minimum size, smoothing, hold). Everyone is covered from their
first frame: there is no cover delay.

    .venv/bin/python -m pytest -q tests/test_renderer.py
"""
from __future__ import annotations

import numpy as np
import pytest

from core.config_table import ConfigTable, load_config
from core.contracts import (PASS, Action, ActionKind, CensoredFrame, Decision, Detections, Frame, Track,
                            TrackState)
from core.detect.detector import _mask_geometry
from core.detect.masks import MaskPipeline
from core.render.renderer import STYLES, StyleRenderer

GEO = _mask_geometry((1080, 1920), (384, 640), (96, 160))


def make_scene():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    frame = Frame(image=image, ts=12.5, idx=40, source_id="test")
    masks = np.zeros((2, 96, 160), np.uint8)
    masks[0, 13:53, 40:61] = 255            # person 1
    masks[1, 13:53, 100:121] = 255          # ID-less person
    tracks = (Track(1, np.array([480, 120, 732, 600], np.float32), 0.9, 0),
              Track(None, np.array([1200, 120, 1452, 600], np.float32), 0.4, 1))
    return frame, Detections(frame.idx, frame.ts, tracks, masks, GEO)


def renderer(style="niqab", smooth=False, fade=0.0):
    sets = [f"policy.conceal_style={style}"]
    sets += [f"timing.mask_smooth_seconds={0.15 if smooth else 0}"]   # the smoothing tests need it on, whatever the config
    sets += [f"timing.fade_out_seconds={fade}"]                       # off unless a test is about the fade
    return StyleRenderer(load_config("intel", sets))


@pytest.mark.parametrize("style", STYLES)
def test_style_changes_exactly_the_covered_pixels(style):
    frame, dets = make_scene()
    original = frame.image.copy()
    r = renderer(style)
    out = r.render(frame, dets, {})

    assert isinstance(out, CensoredFrame) and (out.ts, out.idx) == (frame.ts, frame.idx)
    np.testing.assert_array_equal(frame.image, original)                 # original untouched
    alpha = MaskPipeline.from_config(load_config("intel")).cover_mask(dets, dets.tracks).full((1080, 1920))
    changed = (out.image != original).any(axis=2)
    assert not changed[alpha == 0].any()                                   # nothing outside the mask (and its fade)
    assert changed[alpha == 255].mean() > 0.95                             # (nearly) all of the full cover
    assert ((alpha > 0) & (alpha < 255)).any()                             # a soft edge exists


def test_passed_person_is_shown_id_less_person_is_covered():
    frame, dets = make_scene()
    decisions = {1: Decision(1, PASS, TrackState.PASSED, rule="test", confidence=1.0)}
    out = renderer().render(frame, dets, decisions)
    np.testing.assert_array_equal(out.image[300, 600], frame.image[300, 600])   # person 1: shown
    assert (out.image[300, 1320] != frame.image[300, 1320]).any()              # ID-less: covered


def test_style_comes_from_the_decision_and_unknown_styles_still_cover():
    frame, dets = make_scene()
    r = renderer("niqab")
    solid = {1: Decision(1, Action(ActionKind.CONCEAL, style="solid"), TrackState.CONCEALED, "t", 1.0)}
    out = r.render(frame, dets, solid)
    np.testing.assert_array_equal(out.image[300, 600], r.solid_color)
    odd = {1: Decision(1, Action(ActionKind.CONCEAL, style="glitter"), TrackState.CONCEALED, "t", 1.0)}
    out = r.render(frame, dets, odd)
    np.testing.assert_array_equal(out.image[300, 600], r.solid_color)    # unknown style -> solid fallback


def test_unknown_default_style_is_refused():
    data = load_config("intel").as_dict()
    data["policy"]["conceal_style"] = "glitter"
    with pytest.raises(ValueError):
        StyleRenderer(ConfigTable(data))


def test_people_without_a_decision_follow_the_default_style():
    # The player's style buttons set the default style; people without a decision of their own
    # must get that style too, not the config's niqab.
    frame, dets = make_scene()
    r = renderer("niqab")
    r.set_default_style("solid")
    out = r.render(frame, dets, {})
    np.testing.assert_array_equal(out.image[300, 600], r.solid_color)    # person 1: no decision, sure
    with pytest.raises(ValueError):
        r.set_default_style("glitter")


# -- outlines --------------------------------------------------------------------

def scene_at(ts, cols_by_id, image=None, score=0.9, height=480):
    """A frame at time ts with one person per (track_id -> (col1, col2)) in mask space, rows 13..52.
    ``height``: the person's box height in frame pixels (the box ends at y=600)."""
    image = np.full((1080, 1920, 3), 200, np.uint8) if image is None else image
    frame = Frame(image=image, ts=ts, idx=int(ts * 30), source_id="test")
    masks = np.zeros((len(cols_by_id), 96, 160), np.uint8)
    tracks = []
    for k, (tid, (c1, c2)) in enumerate(cols_by_id.items()):
        masks[k, 13:53, c1:c2] = 255
        tracks.append(Track(tid, np.array([c1 * 12, 600 - height, c2 * 12, 600], np.float32), score, k))
    return frame, Detections(frame.idx, ts, tuple(tracks), masks, GEO)


def is_outline(px, color, tol=40):
    return (np.abs(px.astype(int) - color) <= tol).all(axis=-1)


def test_people_side_by_side_are_separated_by_an_outline():
    frame, dets = scene_at(0.0, {1: (40, 61), 2: (61, 82)})     # touching: one blob without outlines
    r = renderer("niqab")
    out = r.render(frame, dets, {}).image
    between = out[300, 50 * 12:71 * 12]                          # from one person's middle to the other's
    assert is_outline(between, r.outline_color).any()


def test_no_outline_when_disabled():
    r = StyleRenderer(load_config("intel", ["render.outline_fraction=0"]))
    frame, dets = scene_at(0.0, {1: (40, 61), 2: (61, 82)})
    out = r.render(frame, dets, {}).image
    assert not is_outline(out[300, 50 * 12:71 * 12], r.outline_color, tol=20).any()


# -- who is covered, and memory (configured rules) -------------------------------------

def covered_cols(out, row=300, background=200):
    cols = np.nonzero((out[row] != background).any(axis=-1))[0]
    return (cols.min(), cols.max()) if len(cols) else None


def run(r, steps, cols_by_id, start=0.0, dt=0.1, **scene):
    """Render ``steps`` frames dt apart; returns (last ts, last covered cols)."""
    ts, cols = start, None
    for k in range(steps):
        ts = start + k * dt
        frame, dets = scene_at(ts, cols_by_id(k) if callable(cols_by_id) else cols_by_id, **scene)
        cols = covered_cols(r.render(frame, dets, {}).image)
    return ts, cols


def test_tracked_person_is_covered_from_the_first_frame():
    r = renderer("solid", smooth=True)
    for k in range(6):
        frame, dets = scene_at(k * 0.1, {7: (40, 61)})
        assert covered_cols(r.render(frame, dets, {}).image) is not None, f"at {k * 0.1:.1f} s"


def test_detections_without_an_id_are_covered_from_the_first_frame_and_followed():
    r = renderer("solid", smooth=True)
    for k in range(6):                                           # the tracker never gives an ID; slowly moving
        frame, dets = scene_at(k * 0.1, {None: (40 + k, 61 + k)})
        assert covered_cols(r.render(frame, dets, {}).image) is not None, f"at {k * 0.1:.1f} s"
    assert len(r._memory) == 1, "followed as one person by box overlap"


def test_a_one_frame_detection_without_an_id_is_covered_at_once():
    r = renderer("solid", smooth=True)
    for k in range(6):                                           # a different spot every frame: no history
        frame, dets = scene_at(k * 0.1, {None: (10 + 20 * k, 20 + 20 * k)})
        assert covered_cols(r.render(frame, dets, {}).image) is not None, "fail closed: covered, even if false"


def test_getting_a_track_id_keeps_the_history():
    r = renderer("solid", smooth=True)
    run(r, 4, {None: (40, 61)})                                  # 0.0 .. 0.3 s without an ID
    frame, dets = scene_at(0.4, {12: (40, 61)})                  # the tracker now gives it ID 12
    assert covered_cols(r.render(frame, dets, {}).image) is not None
    assert list(r._memory) == [12], "the ID-less history moved to the new ID"


def test_far_people_are_not_covered_and_size_has_hysteresis():
    r = renderer("solid", smooth=True)
    limit = r.min_height * 1080
    _, cols = run(r, 5, {7: (40, 61)}, height=limit * 0.9)       # too small
    assert cols is None
    _, cols = run(r, 2, {7: (40, 61)}, start=0.5, height=limit * 1.1)
    assert cols is not None                                      # big enough: covered
    _, cols = run(r, 2, {7: (40, 61)}, start=0.7, height=limit * 0.9)
    assert cols is not None, "once covered, a little smaller does not blink it off"
    _, cols = run(r, 2, {7: (40, 61)}, start=0.9, height=limit * 0.6)
    assert cols is None, "much smaller (further away) does"


def test_missed_person_is_held_and_keeps_moving_then_released():
    r = renderer("solid", smooth=True)
    last_ts, last = run(r, 6, lambda k: {7: (40 + 3 * k, 61 + 3 * k)})   # moving right, 36 px per 0.1 s
    assert last is not None
    frame, empty = scene_at(last_ts + 0.1, {})                   # the detector misses person 7
    held = covered_cols(r.render(frame, empty, {}).image)
    assert held is not None, "a missed person stays covered"
    assert held[0] > last[0] and held[1] > last[1], "and keeps moving the way they were going"
    frame, empty = scene_at(last_ts + r.hold_s + 0.05, {})       # longer than mask_hold_seconds after last seen
    assert covered_cols(r.render(frame, empty, {}).image) is None


def test_held_person_who_is_now_passed_is_not_covered():
    r = renderer("solid", smooth=True)
    last_ts, _ = run(r, 5, {7: (40, 61)})
    frame, empty = scene_at(last_ts + 0.1, {})
    passed = {7: Decision(7, PASS, TrackState.PASSED, rule="test", confidence=1.0)}
    assert covered_cols(r.render(frame, empty, passed).image) is None


def test_mask_shape_changes_gradually_both_ways():
    r = renderer("solid", smooth=True)
    ts, wide = run(r, 12, {7: (40, 70)}, dt=1 / 30)              # steady, covered
    _, narrow_now = run(renderer("solid"), 1, {7: (40, 55)})
    ts, after_one = run(r, 1, {7: (40, 55)}, start=ts + 1 / 30)  # the mask suddenly shrinks
    assert narrow_now[1] < after_one[1] < wide[1], "one frame later the edge is only part of the way"
    ts, settled = run(r, 30, {7: (40, 55)}, start=ts + 1 / 30, dt=1 / 30)
    assert abs(settled[1] - narrow_now[1]) <= 2, "and settles on the new shape"
    ts, grown_one = run(r, 1, {7: (40, 70)}, start=ts + 1 / 30)  # and grows back
    assert settled[1] < grown_one[1] < wide[1]


def test_redrawing_the_same_frame_changes_nothing():
    r = renderer("niqab", smooth=True)
    run(r, 5, {7: (40, 61)})
    frame, dets = scene_at(0.4, {7: (40, 61)})
    first = r.render(frame, dets, {}).image
    np.testing.assert_array_equal(r.render(frame, dets, {}).image, first)   # paused redraw


def test_a_box_that_only_shrinks_is_not_movement():
    from core.render.renderer import _box_motion
    np.testing.assert_array_equal(_box_motion(np.array([480, 120, 840, 600]), np.array([480, 120, 660, 600])), [0, 0])
    np.testing.assert_array_equal(_box_motion(np.array([480, 120, 840, 600]), np.array([500, 120, 870, 600])), [20, 0])


# -- fade-out when a conceal is lifted (T12) ---------------------------------------------

PASSED_7 = {7: Decision(7, PASS, TrackState.PASSED, rule="test", confidence=1.0)}


def centre(r, ts, decisions, cols=None):
    """Pixel in the middle of person 7 (cols 40..61) at time ts."""
    frame, dets = scene_at(ts, {7: (40, 61)} if cols is None else cols)
    return r.render(frame, dets, decisions).image[300, 600].astype(int)


def test_lifting_a_conceal_fades_out_over_the_configured_time():
    r = renderer("solid", fade=0.2)
    run(r, 5, {7: (40, 61)})                                     # covered 0.0 .. 0.4 s
    solid, background = np.array(r.solid_color), np.array([200, 200, 200])
    at_lift = centre(r, 0.5, PASSED_7)
    halfway = centre(r, 0.6, PASSED_7)
    done = centre(r, 0.7, PASSED_7)
    np.testing.assert_array_equal(at_lift, solid)                # still fully covered the moment it is lifted
    assert np.all(np.abs(halfway - (solid + background) / 2) <= 2), "half faded after half the time"
    np.testing.assert_array_equal(done, background)              # gone after fade_out_seconds


def test_concealing_again_is_instant_and_ends_the_fade():
    r = renderer("solid", fade=0.2)
    run(r, 5, {7: (40, 61)})
    centre(r, 0.5, PASSED_7)
    centre(r, 0.6, PASSED_7)                                     # half faded
    np.testing.assert_array_equal(centre(r, 0.65, {}), r.solid_color)   # concealed again: fully, at once
    np.testing.assert_array_equal(centre(r, 0.7, {}), r.solid_color)


def test_fade_is_timed_in_seconds_not_frames():
    seen = []
    for fps in (15, 30):
        r = renderer("solid", fade=0.2)
        last, _ = run(r, fps // 2, {7: (40, 61)}, dt=1 / fps)    # half a second covered
        lift = last + 1 / fps
        for k in range(fps):                                     # passed from `lift` on
            ts = lift + k / fps
            px = centre(r, ts, PASSED_7)
            if abs(ts - (lift + 2 / 15)) < 1e-6:                  # 2 frames at 15 fps, 4 at 30
                seen.append(px)
    assert len(seen) == 2
    np.testing.assert_allclose(seen[0], seen[1], atol=1)        # same look 0.13 s into the fade at 15 and 30 fps
    assert not np.array_equal(seen[0], [200, 200, 200])         # (and it is mid-fade, not gone)


def test_a_fading_person_never_thins_a_neighbours_cover():
    r = renderer("solid", fade=0.2)
    people = {7: (40, 61), 8: (50, 71)}                          # overlapping in columns 50..60
    run(r, 5, people)
    frame, dets = scene_at(0.6, people)
    r.render(scene_at(0.5, people)[0], scene_at(0.5, people)[1], PASSED_7)
    out = r.render(frame, dets, PASSED_7).image                  # 7 fading, 8 still concealed
    np.testing.assert_array_equal(out[300, 55 * 12], r.solid_color)


def test_no_fade_when_fade_out_seconds_is_zero():
    r = renderer("solid", fade=0.0)
    run(r, 5, {7: (40, 61)})
    np.testing.assert_array_equal(centre(r, 0.5, PASSED_7), [200, 200, 200])
