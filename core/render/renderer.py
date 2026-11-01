"""Renderer (T11, plus the mask hold and fade-out of T12): paints the conceal styles where the cover masks say.

Styles (policy.conceal_style, or the style on a person's conceal action):
    niqab     a plain dark cover, full body (the eye slit is a later task, T32)
    blur      the frame shrunk to a few dozen rows and enlarged again: unrecognisable
    pixelate  big square blocks
    solid     one flat colour

Each style is painted once through the combined cover of its people: fully opaque where
a person is (grown a little), fading out over a few pixels outside that. Then each
person gets an outline, so people next to each other stay distinguishable; outlines are
layered by depth (lower in the frame = nearer the camera), so a nearer person hides the
outline of the one behind.

Who is covered:
    every detected person at least render.min_person_height tall, from their first frame,
    ID-less detections included (no delay: a person is never shown while being confirmed).
    Detections the tracker gives no ID are followed by box overlap with recent frames, so
    their masks are smoothed and held like tracked people's. render.min_person_height: 0
    covers everyone, whatever their size.

Memory per tracked person (track ID), all timed on Frame.ts:
    smooth   the mask is a running average over about timing.mask_smooth_seconds, with the
             previous mask moved along with the person, so its shape changes gradually
    hold     a covered person the detector misses stays covered for timing.mask_hold_seconds,
             their last mask sliding on at their last speed
    fade     when a covered person's conceal is lifted (a pass decision, or they shrank below the
             size threshold), their cover fades from opaque to nothing over timing.fade_out_seconds
             (0 = at once). Concealing is always instant: a conceal decision ends a fade at once.
             Fading people are painted before the full covers, so a fade never thins anyone
             else's cover, and they get no outline.

Style images are made once per style for the rectangle around that style's people
(Ultralytics runs OpenCV single-threaded, so full-frame work is expensive). Pixel blocks
are anchored to the frame, so they don't shimmer as people move.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from core.contracts import (MASK_CUT, ActionKind, CensoredFrame, Decision, Detections, Frame, MaskGeometry, Renderer,
                            Track)
from core.detect.masks import MaskPipeline, is_covered, shows

Region = tuple[int, int, int, int]   # x1, y1, x2, y2 in frame pixels

STYLES = ("niqab", "blur", "pixelate", "solid")
FALLBACK_STYLE = "solid"             # an unknown style on a decision still covers the person
MAX_SPEED = 1.5                      # frame heights per second: caps how far a held mask can slide
SIZE_HYSTERESIS = 0.8                # once covered, stay covered down to this share of min_person_height
MATCH_IOU = 0.3                      # box overlap that makes a detection without an ID "the same person as before"


@dataclass
class _Person:
    """One person to cover this frame: a detection, or a held one the detector just missed."""
    style: str
    small: np.ndarray                # soft mask-space mask
    bbox: np.ndarray                 # frame pixels
    geometry: MaskGeometry
    alpha: float = 1.0               # < 1: fading out (conceal lifted)


@dataclass
class _Memory:
    small: np.ndarray                # smoothed mask, as of ts
    bbox: np.ndarray
    velocity: np.ndarray             # frame pixels per second (x, y)
    ts: float                        # last seen
    geometry: MaskGeometry
    style: str = ""
    covered: bool = False            # drawn last time it was seen (only covered people are held)
    fade_from: float | None = None   # Frame.ts the conceal was lifted, while the fade-out runs


class StyleRenderer(Renderer):
    def __init__(self, cfg: Any) -> None:
        r, t = cfg.render, cfg.timing
        self.masks = MaskPipeline.from_config(cfg)
        self.blur_rows = int(r.blur_rows)
        self.pixelate_rows = int(r.pixelate_rows)
        self.solid_color = tuple(int(c) for c in r.solid_color)
        self.niqab_color = tuple(int(c) for c in r.niqab_color)
        self.outline_fraction = float(r.outline_fraction)
        self.outline_color = tuple(int(c) for c in r.outline_color)
        self.min_height = float(r.min_person_height)
        self.hold_s = float(t.mask_hold_seconds)
        self.smooth_s = float(t.mask_smooth_seconds)
        self.fade_s = float(t.get("fade_out_seconds", 0.0))
        # painter(frame image, region) -> styled pixels for that region
        self._painters: dict[str, Callable[[np.ndarray, Region], np.ndarray]] = {
            "niqab": self._niqab, "blur": self._blur, "pixelate": self._pixelate, "solid": self._solid,
        }
        self._fills: dict[tuple, np.ndarray] = {}   # flat colour images per (shape, colour)
        self._memory: dict[int, _Memory] = {}   # key: track ID, or negative for detections without one
        self._next_key = 0
        self.last_timing: dict[str, float] = {}
        self.keep_cover = False                  # evaluation: keep each frame's cover in last_cover
        self.last_cover: np.ndarray | None = None   # (H, W) uint8 alpha, 255 = fully covered
        self.set_default_style(str(cfg.policy.conceal_style))

    def set_default_style(self, style: str) -> None:
        """Style for people with no conceal decision of their own (ID-less, undecided)."""
        if style not in STYLES:
            raise ValueError(f"conceal style {style!r} is not one of {STYLES}")
        self.default_style = style

    # -- Renderer interface ---------------------------------------------------

    def render(self, frame: Frame, detections: Detections, decisions: dict[int, Decision]) -> CensoredFrame:
        t0 = time.perf_counter()
        out = frame.image.copy()                       # never draw on the original (rule 2)
        geometry = detections.geometry
        people = self._people(frame.ts, detections, decisions)
        grown = [self.masks.grow_person(p.small, p.bbox, geometry) for p in people]
        t1 = time.perf_counter()
        if self.keep_cover:
            self.last_cover = np.zeros(frame.image.shape[:2], np.uint8)

        for p, g in zip(people, grown):                 # fading people first: full covers go on top
            if p.alpha < 1:
                cover = self.masks.cover_from_small(g, geometry)
                if cover is not None:
                    x1, y1, x2, y2 = cover.region
                    alpha = (cover.mask.astype(np.float32) * p.alpha).astype(np.uint8)
                    _paint(out[y1:y2, x1:x2], self._painters[p.style](frame.image, cover.region), alpha)
                    if self.keep_cover:
                        np.maximum(self.last_cover[y1:y2, x1:x2], alpha, out=self.last_cover[y1:y2, x1:x2])

        for style in {p.style for p in people if p.alpha >= 1}:   # one combined cover and one paint per style
            union = None
            for p, g in zip(people, grown):
                if p.style == style and p.alpha >= 1:
                    union = g.copy() if union is None else np.maximum(union, g, out=union)
            cover = self.masks.cover_from_small(union, geometry)
            if cover is not None:
                x1, y1, x2, y2 = cover.region
                _paint(out[y1:y2, x1:x2], self._painters[style](frame.image, cover.region), cover.mask)
                if self.keep_cover:
                    np.maximum(self.last_cover[y1:y2, x1:x2], cover.mask, out=self.last_cover[y1:y2, x1:x2])
        t2 = time.perf_counter()

        thickness = self.outline_fraction * frame.image.shape[0]
        solid = [(p, g) for p, g in zip(people, grown) if p.alpha >= 1]
        if thickness > 0 and solid:
            far_to_near = [g for _, g in sorted(solid, key=lambda pg: float(pg[0].bbox[3]))]
            _draw_outlines(out, far_to_near, geometry, max(1.0, thickness), self.outline_color)
        t3 = time.perf_counter()
        self.last_timing = {"mask": 1000 * (t1 - t0), "paint": 1000 * (t2 - t1),
                            "outline": 1000 * (t3 - t2), "render (total)": 1000 * (t3 - t0)}
        return CensoredFrame(image=out, ts=frame.ts, idx=frame.idx)

    def reset(self) -> None:
        self._memory.clear()
        self._next_key = 0

    # -- who to cover, with memory ----------------------------------------------

    def _people(self, now: float, detections: Detections, decisions: dict[int, Decision]) -> list[_Person]:
        """Who to draw this frame. Every detection is followed over time under a key: its track ID,
        or, for detections the tracker has not given an ID, a key of our own (negative) found by
        box overlap with recent frames. A new key takes over the history of the overlapping person
        it replaces, so an ID appearing or disappearing keeps their smoothed mask and size state."""
        geometry = detections.geometry
        frame_h = geometry.frame_shape[0]
        people, seen = [], set()
        tracked = [t for t in detections.tracks if t.track_id is not None]
        unconfirmed = [t for t in detections.tracks if t.track_id is None]
        for track in tracked + unconfirmed:                     # real IDs first: they claim history first
            key = track.track_id if track.track_id is not None else self._unconfirmed_key(track, seen, now, geometry)
            if key not in self._memory:
                self._take_over(key, track, seen, now, geometry)
            seen.add(key)
            m = self._remember(key, track, detections.mask_of(track), geometry, now)
            height = float(track.bbox[3] - track.bbox[1]) / frame_h
            needed = self.min_height * (SIZE_HYSTERESIS if m.covered else 1.0)
            if not is_covered(track, decisions) or height < needed:   # shown (pass), or too far away for now
                alpha = self._lift(m, now)
                if alpha is not None:
                    people.append(_Person(m.style, m.small, track.bbox, geometry, alpha))
                continue
            m.covered, m.style, m.fade_from = True, self._style_of(track, decisions), None
            people.append(_Person(m.style, m.small, track.bbox, geometry))

        for key, m in list(self._memory.items()):               # not detected this frame
            if key in seen:
                continue
            age = now - m.ts
            if not 0 <= age <= self.hold_s or m.geometry != geometry:
                del self._memory[key]                            # gone for good: forget
                continue
            alpha = 1.0
            if key >= 0 and shows(decisions.get(key)) or not m.covered:
                alpha = self._lift(m, now)                       # passed while out of sight, or already fading
                if alpha is None:
                    continue                                     # nothing to hold
            moved = m.velocity * age
            people.append(_Person(m.style, _shift(m.small, moved * geometry.scale),
                                  m.bbox + np.tile(moved, 2), geometry, alpha))
        return people

    def _lift(self, m: _Memory, now: float) -> float | None:
        """This person is no longer to be covered: their fade-out opacity now, or None once gone."""
        if m.covered:                                            # the conceal is lifted now
            m.covered, m.fade_from = False, now
        if m.fade_from is None:
            return None
        alpha = 1.0 - (now - m.fade_from) / self.fade_s if self.fade_s > 0 else 0.0
        if not 0 < alpha <= 1:                                   # done (or time went backwards: no fade)
            m.fade_from = None
            return None
        return alpha

    def _missing_overlapping(self, track: Track, seen: set[int], now: float,
                             geometry: MaskGeometry) -> int | None:
        """Key of the remembered, not-yet-seen-this-frame person whose box (moved on at their speed)
        overlaps this detection most, if at least MATCH_IOU."""
        best, best_iou = None, MATCH_IOU
        for key, m in self._memory.items():
            age = now - m.ts
            if key in seen or m.geometry != geometry or not 0 <= age <= self.hold_s:
                continue
            iou = _iou(m.bbox + np.tile(m.velocity * age, 2), track.bbox)
            if iou >= best_iou:
                best, best_iou = key, iou
        return best

    def _unconfirmed_key(self, track: Track, seen: set[int], now: float, geometry: MaskGeometry) -> int:
        key = self._missing_overlapping(track, seen, now, geometry)
        if key is not None:
            return key                                           # the same person as before (any key)
        self._next_key -= 1
        return self._next_key

    def _take_over(self, key: int, track: Track, seen: set[int], now: float, geometry: MaskGeometry) -> None:
        """A new key for someone already remembered under another key: move the history over."""
        old = self._missing_overlapping(track, seen, now, geometry)
        if old is not None:
            self._memory[key] = self._memory.pop(old)

    def _remember(self, key: int, track: Track, small: np.ndarray, geometry: MaskGeometry, now: float) -> _Memory:
        """Update this person's memory: smoothed mask, speed."""
        prev = self._memory.get(key)
        if prev is None or prev.geometry != geometry or not 0 <= now - prev.ts <= self.hold_s:
            m = self._memory[key] = _Memory(small, track.bbox.copy(), np.zeros(2, np.float32), now, geometry)
            return m
        dt = now - prev.ts
        if dt > 0:                                               # (dt == 0: the same frame again, paused)
            moved = _box_motion(prev.bbox, track.bbox)
            velocity = 0.5 * prev.velocity + 0.5 * moved / dt
            limit = MAX_SPEED * geometry.frame_shape[0]
            speed = float(np.hypot(*velocity))
            if speed > limit:
                velocity *= limit / speed
            if self.smooth_s > 0:                                # running average, previous moved along
                weight = 1 - math.exp(-dt / self.smooth_s)       # share of the new mask
                small = cv2.addWeighted(small, weight, _shift(prev.small, moved * geometry.scale), 1 - weight, 0)
            prev.small, prev.velocity = small, velocity.astype(np.float32)
            prev.bbox, prev.ts = track.bbox.copy(), now
        return prev

    def _style_of(self, track: Track, decisions: dict[int, Decision]) -> str:
        d = decisions.get(track.track_id) if track.track_id is not None else None
        if d is None or d.action.kind is not ActionKind.CONCEAL:
            return self.default_style
        return d.action.style if d.action.style in STYLES else FALLBACK_STYLE

    # -- styles ---------------------------------------------------------------

    def _blur(self, image: np.ndarray, region: Region) -> np.ndarray:
        x1, y1, x2, y2 = region
        cell = image.shape[0] / max(2, self.blur_rows)        # strength set by the frame, not the region
        roi = image[y1:y2, x1:x2]
        small = cv2.resize(roi, (max(1, round((x2 - x1) / cell)), max(1, round((y2 - y1) / cell))),
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 1.0)
        return cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)

    def _pixelate(self, image: np.ndarray, region: Region) -> np.ndarray:
        x1, y1, x2, y2 = region
        h, w = image.shape[:2]
        b = max(2, round(h / max(2, self.pixelate_rows)))     # block size in frame pixels
        # Blocks on a fixed frame grid: widen the region to whole blocks, average each block
        # (INTER_AREA at an exact 1/b), enlarge, cut back to the region.
        ax1, ay1 = x1 // b * b, y1 // b * b
        cols, rows = -(-(x2 - ax1) // b), -(-(y2 - ay1) // b)
        roi = image[ay1:min(h, ay1 + rows * b), ax1:min(w, ax1 + cols * b)]
        if roi.shape[0] != rows * b or roi.shape[1] != cols * b:   # last blocks run past the frame edge
            roi = cv2.copyMakeBorder(roi, 0, rows * b - roi.shape[0], 0, cols * b - roi.shape[1],
                                     cv2.BORDER_REPLICATE)
        small = cv2.resize(roi, (cols, rows), interpolation=cv2.INTER_AREA)
        big = cv2.resize(small, (cols * b, rows * b), interpolation=cv2.INTER_NEAREST)
        return big[y1 - ay1:y2 - ay1, x1 - ax1:x2 - ax1]

    def _solid(self, image: np.ndarray, region: Region) -> np.ndarray:
        return self._fill(image, region, self.solid_color)

    def _niqab(self, image: np.ndarray, region: Region) -> np.ndarray:
        return self._fill(image, region, self.niqab_color)

    def _fill(self, image: np.ndarray, region: Region, color: tuple[int, ...]) -> np.ndarray:
        key = (image.shape, color)
        if key not in self._fills:
            self._fills[key] = np.full(image.shape, color, dtype=np.uint8)
        x1, y1, x2, y2 = region
        return self._fills[key][y1:y2, x1:x2]


# -- drawing helpers -------------------------------------------------------------------

def _paint(dst: np.ndarray, styled: np.ndarray, alpha: np.ndarray) -> None:
    """dst = styled where alpha is 255, a mix on the fading edge (1..254), untouched where 0.

    OpenCV, not numpy fancy indexing: ~5x faster on a close-up person. Results are always
    assigned back into ``dst`` explicitly, so the cover can never be lost to an OpenCV call
    that returns a new array instead of writing in place.
    """
    dst[...] = cv2.copyTo(styled, cv2.compare(alpha, 254, cv2.CMP_GT), dst)      # full cover
    edge = cv2.inRange(alpha, 1, 254)
    x, y, w, h = cv2.boundingRect(edge)
    if w == 0:
        return
    sub, edge = dst[y:y + h, x:x + w], edge[y:y + h, x:x + w]
    a = alpha[y:y + h, x:x + w].astype(np.float32) * (1 / 255)
    mixed = cv2.blendLinear(styled[y:y + h, x:x + w], sub, a, 1 - a)
    sub[...] = cv2.copyTo(mixed, edge, sub)


def _draw_outlines(out: np.ndarray, far_to_near: list[np.ndarray], geometry: MaskGeometry,
                   thickness: float, color: tuple[int, ...]) -> None:
    """An outline along each person's full-cover edge; nearer people hide the outlines behind them.

    Edges are traced on a grid k times finer than mask space (one cell is about ``thickness``
    frame pixels), nearest person first; outline points inside someone nearer are dropped.
    The rest is drawn straight onto the frame as anti-aliased lines at sub-pixel precision.
    """
    s = geometry.scale
    k = max(1, round(1 / (s * thickness)))
    mh, mw = far_to_near[0].shape
    nearer = np.zeros((mh * k, mw * k), np.uint8)               # cells covered by someone nearer
    lines: list[np.ndarray] = []
    for grown in reversed(far_to_near):                         # nearest first
        box = _bounding_box(grown >= MASK_CUT, 1)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        fine = cv2.resize(grown[y1:y2, x1:x2], ((x2 - x1) * k, (y2 - y1) * k), interpolation=cv2.INTER_LINEAR)
        full = cv2.compare(fine, MASK_CUT - 1, cv2.CMP_GT)
        contours, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=(x1 * k, y1 * k))
        for c in contours:
            pts = c[:, 0, :]
            visible = nearer[pts[:, 1], pts[:, 0]] == 0
            lines.extend(_visible_runs(_smooth_closed(pts), visible))
        view = nearer[y1 * k:y2 * k, x1 * k:x2 * k]
        np.maximum(view, full, out=view)
    if not lines:
        return
    # fine cell v (pixel centres) -> frame pixel X:  X = ((v + 0.5) / k - 0.5 - pad + 0.5 - s / 2) / s
    shift_bits = 4                                              # draw at 1/16 pixel precision
    offset = np.array([0.5 / k - geometry.pad_x - s / 2, 0.5 / k - geometry.pad_y - s / 2], np.float32)
    scale = (1 << shift_bits) / s
    polys = [np.round((run.astype(np.float32) / k + offset) * scale).astype(np.int32) for run in lines]
    cv2.polylines(out, [p for p in polys if len(p) > 1], False, color, max(1, round(thickness)),
                  cv2.LINE_AA, shift_bits)


def _smooth_closed(pts: np.ndarray, window: int = 5) -> np.ndarray:
    """Average each point of a closed contour with its neighbours: grid steps -> smooth curve."""
    if len(pts) <= window:
        return pts.astype(np.float32)
    half = window // 2
    padded = np.vstack([pts[-half:], pts, pts[:half]]).astype(np.float32)
    kernel = np.full(window, 1 / window, np.float32)
    return np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in (0, 1)], axis=1)


def _visible_runs(pts: np.ndarray, visible: np.ndarray) -> list[np.ndarray]:
    """Split a closed contour into the runs of consecutive visible points (wrapping around)."""
    if visible.all():
        return [np.vstack([pts, pts[:1]])]                      # closed
    if not visible.any():
        return []
    start = int(np.argmin(visible))                             # begin at a hidden point, so runs never wrap
    pts, visible = np.roll(pts, -start, axis=0), np.roll(visible, -start)
    edges = np.flatnonzero(np.diff(visible.astype(np.int8)))
    bounds = np.split(np.arange(len(pts)), edges + 1)
    return [pts[b] for b in bounds if visible[b[0]] and len(b) > 1]


def _bounding_box(mask: np.ndarray, spare: int) -> tuple[int, int, int, int] | None:
    """x1, y1, x2, y2 around the True pixels, ``spare`` pixels wider, clipped. None if empty."""
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    if w == 0:
        return None
    mh, mw = mask.shape
    return max(0, x - spare), max(0, y - spare), min(mw, x + w + spare), min(mh, y + h + spare)


def _shift(small: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Move a mask-space mask by (dx, dy) mask pixels, sub-pixel."""
    if not d.any():
        return small
    m = np.float32([[1, 0, d[0]], [0, 1, d[1]]])
    return cv2.warpAffine(small, m, (small.shape[1], small.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _box_motion(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """How far a person moved (dx, dy), from their box edges.

    Both edges moving the same way is movement (the smaller of the two moves, to be safe);
    one edge moving alone is the box growing or shrinking (an arm, a step out of view),
    which is not movement. The box centre would wrongly read that as moving.
    """
    d = np.asarray(after, np.float32) - np.asarray(before, np.float32)   # dx1, dy1, dx2, dy2
    out = np.zeros(2, np.float32)
    for axis in (0, 1):
        a, b = d[axis], d[axis + 2]
        if a * b > 0:
            out[axis] = a if abs(a) < abs(b) else b
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return float(inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter))


def _union(regions: list[Region]) -> Region:
    return (min(r[0] for r in regions), min(r[1] for r in regions),
            max(r[2] for r in regions), max(r[3] for r in regions))
