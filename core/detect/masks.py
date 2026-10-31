"""Mask pipeline (T10): the detector's small masks -> one full-resolution cover mask.

    pick who is covered -> grow each person's mask -> combine -> remove the letterbox
    and enlarge once, straight to the frame -> cut at MASK_CUT

Masks are soft (0..255 certainty) until the very last step, so the cut lands between
mask pixels and outlines move smoothly instead of jumping in mask-pixel steps.

The detector's masks are tiny (96x160 for a 1080p frame, so one mask pixel covers
a 12x12 block of the frame). Everything per person happens at that size, where it
costs microseconds; only the final enlargement touches full-resolution pixels, once,
however many people are covered, and only inside the rectangle around them
(Ultralytics runs OpenCV single-threaded, where a full 1080p enlargement costs ~5 ms).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from core.contracts import MASK_CUT, ActionKind, Decision, Detections, MaskGeometry, Track


# -- who is covered ----------------------------------------------------------------

def shows(decision: Decision | None) -> bool:
    """True only for an explicit pass (or decorate, which keeps the person visible). Fail closed."""
    if decision is None:
        return False                     # no decision yet
    match decision.action.kind:
        case ActionKind.CONCEAL:
            return False
        case ActionKind.PASS | ActionKind.DECORATE:
            return True
        case _:
            return False                 # a kind added later and not handled here


def is_covered(track: Track, decisions: Mapping[int, Decision]) -> bool:
    """ID-less detections are always covered; tracked people unless their decision shows them."""
    if track.track_id is None:
        return True                      # not tracked yet
    return not shows(decisions.get(track.track_id))


def covered_tracks(detections: Detections, decisions: Mapping[int, Decision]) -> list[Track]:
    return [t for t in detections.tracks if is_covered(t, decisions)]


# -- small masks -> one full-resolution mask -----------------------------------------

ROUND_SIGMA = 0.8   # mask px: rounds the corners growth leaves behind (an ellipse on a coarse grid is a polygon)


class MaskPipeline:
    """Builds the full-resolution cover mask for a set of tracks.

    grow_fraction       each person's mask grows by this share of their box height
                        (covers hair, loose clothes, a mask that lags a fast move)
    grow_min_fraction   ...but by at least this share of the frame height (far people)
    edge_fade           certainty levels just below MASK_CUT that fade out instead of ending in
                        a hard edge (0 = hard edge). Only adds partial cover outside the full cover.
    """

    def __init__(self, grow_fraction: float = 0.04, grow_min_fraction: float = 0.01, edge_fade: int = 0) -> None:
        if grow_fraction < 0 or grow_min_fraction < 0:
            raise ValueError("mask growth cannot be negative")
        if not 0 <= edge_fade < MASK_CUT:
            raise ValueError(f"edge_fade must be 0..{MASK_CUT - 1}")
        self.grow_fraction = grow_fraction
        self.grow_min_fraction = grow_min_fraction
        self.edge_fade = edge_fade
        self._kernels: dict[int, np.ndarray] = {}

    @classmethod
    def from_config(cls, cfg: Any) -> MaskPipeline:
        r = cfg.render
        return cls(float(r.mask_grow_fraction), float(r.mask_grow_min_fraction), int(r.edge_fade))

    def cover_mask(self, detections: Detections, tracks: Iterable[Track]) -> CoverMask | None:
        """Where to cover, at frame resolution. None when nobody is covered."""
        small = self.merged_small(detections, tracks)
        return None if small is None else self.cover_from_small(small, detections.geometry)

    def grow_person(self, small: np.ndarray, bbox: np.ndarray, geometry: MaskGeometry) -> np.ndarray:
        """One person's grown soft mask, in mask space (a detection's mask, or a held one)."""
        return self._grow(small, self._radius(float(bbox[3] - bbox[1]), geometry))

    def cover_from_small(self, small: np.ndarray, geometry: MaskGeometry) -> CoverMask | None:
        """A grown (and possibly combined) mask-space mask -> its cover at frame resolution."""
        region = frame_region(small, geometry, MASK_CUT - self.edge_fade)
        if region is None:
            return None
        return CoverMask(region, to_frame_resolution(small, geometry, region, self.edge_fade))

    def merged_small(self, detections: Detections, tracks: Iterable[Track]) -> np.ndarray | None:
        """The grown, combined mask in mask space, or None for no tracks."""
        merged = None
        for track in tracks:
            grown = self._grow(detections.mask_of(track), self.grow_radius(track, detections.geometry))
            merged = grown.copy() if merged is None else np.maximum(merged, grown, out=merged)
        return merged

    def grow_radius(self, track: Track, geometry: MaskGeometry) -> int:
        """Growth in mask pixels, rounded up (fail closed: never less than asked)."""
        return self._radius(float(track.bbox[3] - track.bbox[1]), geometry)

    def _radius(self, box_h: float, geometry: MaskGeometry) -> int:
        frame_px = max(self.grow_fraction * box_h, self.grow_min_fraction * geometry.frame_shape[0])
        return math.ceil(frame_px * geometry.scale - 1e-6)

    def _grow(self, mask: np.ndarray, radius: int) -> np.ndarray:
        """Grow a soft mask by ``radius`` mask pixels, with rounded edges. Never smaller than the input."""
        if radius <= 0:
            return mask
        kernel = self._kernels.get(radius)
        if kernel is None:
            kernel = self._kernels[radius] = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        # Work only on the patch around the person (masks are zero outside their box), with room
        # for the growth and the rounding blur.
        x, y, w, h = cv2.boundingRect(mask)
        if w == 0:
            return mask
        spare = radius + 3 * math.ceil(ROUND_SIGMA) + 1
        mh, mw = mask.shape
        x1, y1, x2, y2 = max(0, x - spare), max(0, y - spare), min(mw, x + w + spare), min(mh, y + h + spare)
        patch = mask[y1:y2, x1:x2]
        grown_patch = cv2.GaussianBlur(cv2.dilate(patch, kernel), (0, 0), ROUND_SIGMA)
        grown = np.zeros_like(mask)
        grown[y1:y2, x1:x2] = np.maximum(grown_patch, patch)   # rounding must never uncover the person's own mask
        return grown


@dataclass(frozen=True, eq=False)
class CoverMask:
    """The cover mask at frame resolution, stored only for the rectangle around the covered people."""

    region: tuple[int, int, int, int]   # x1, y1, x2, y2 in frame pixels (x2, y2 exclusive)
    mask: np.ndarray                    # uint8 (y2-y1, x2-x1): 255 = full cover, 1..254 = fading edge

    def full(self, frame_shape: tuple[int, int]) -> np.ndarray:
        """The whole-frame mask (for tests and overlays; the renderer works on the region)."""
        x1, y1, x2, y2 = self.region
        out = np.zeros(frame_shape, np.uint8)
        out[y1:y2, x1:x2] = self.mask
        return out


def frame_region(mask: np.ndarray, geometry: MaskGeometry, level: int = MASK_CUT) -> tuple[int, int, int, int] | None:
    """Frame rectangle around the mask pixels >= ``level``, with two mask pixels to spare (the smooth
    enlargement reaches past the last such pixel). None for an empty mask."""
    person = (mask >= level).astype(np.uint8)
    if not person.any():
        return None
    bx, by, bw, bh = cv2.boundingRect(person)
    s = geometry.scale
    h, w = geometry.frame_shape
    x1 = max(0, math.floor((bx - 2 - geometry.pad_x) / s))
    y1 = max(0, math.floor((by - 2 - geometry.pad_y) / s))
    x2 = min(w, math.ceil((bx + bw + 2 - geometry.pad_x) / s))
    y2 = min(h, math.ceil((by + bh + 2 - geometry.pad_y) / s))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def to_frame_resolution(mask: np.ndarray, geometry: MaskGeometry,
                        region: tuple[int, int, int, int] | None = None, edge_fade: int = 0) -> np.ndarray:
    """A mask-space mask -> frame resolution, letterbox removed, in one exact step.

    One affine warp does both the crop and the enlargement, so fractional padding
    (e.g. 22.75 mask pixels for a 416x416 model input) lands exactly instead of being
    rounded to whole mask pixels (12 frame pixels each). Smooth enlargement then a cut
    at the midpoint gives clean diagonal edges instead of 12-pixel stair steps.

    region: only compute this frame rectangle (x1, y1, x2, y2); default the whole frame.
    edge_fade: certainty levels below MASK_CUT that ramp from 0 to 254 instead of 0 (a soft outer edge).
    """
    h, w = geometry.frame_shape
    x1, y1, x2, y2 = region or (0, 0, w, h)
    s = geometry.scale
    # frame pixel (x, y) -> mask pixel (x*s + off, y*s + off'), pixel centres lined up
    to_mask = np.float32([
        [s, 0, geometry.pad_x + 0.5 * s - 0.5 + s * x1],
        [0, s, geometry.pad_y + 0.5 * s - 0.5 + s * y1],
    ])
    full = cv2.warpAffine(mask, to_mask, (x2 - x1, y2 - y1), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)   # a person at the frame edge stays covered to the edge
    return cv2.LUT(full, _cut_table(edge_fade))   # >= MASK_CUT -> 255 (full cover), just below it -> fade


_CUT_TABLES: dict[int, np.ndarray] = {}


def _cut_table(edge_fade: int) -> np.ndarray:
    table = _CUT_TABLES.get(edge_fade)
    if table is None:
        v = np.arange(256, dtype=np.float32)
        table = np.where(v >= MASK_CUT, 255.0, 0.0)
        if edge_fade:
            start = MASK_CUT - edge_fade
            ramp = (v >= start) & (v < MASK_CUT)
            table[ramp] = np.minimum(254, np.round((v[ramp] - start + 1) * 254 / (edge_fade + 1)))
        table = _CUT_TABLES[edge_fade] = table.astype(np.uint8)
    return table


# -- drawing helpers -------------------------------------------------------------------

def outlines_in_frame(mask: np.ndarray, geometry: MaskGeometry) -> list[np.ndarray]:
    """Outer contours of a mask-space mask, as int32 point arrays in frame pixels.

    Traced at mask resolution and scaled up, which is far cheaper than resizing
    the mask to full resolution first.
    """
    person = (mask >= MASK_CUT).astype(np.uint8)
    contours, _ = cv2.findContours(person, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    offset = np.array([geometry.pad_x, geometry.pad_y], dtype=np.float32)
    return [np.round((c.astype(np.float32) - offset) / geometry.scale).astype(np.int32) for c in contours]
