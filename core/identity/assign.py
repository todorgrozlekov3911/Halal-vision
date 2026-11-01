"""Face-to-subject assignment (T17): which tracked person a face belongs to, through the masks.

    assigner = FaceAssigner(cfg)
    faces = assigner.assign(faces, detections)          # FaceObservation.track_id set, or None
    face = look(job, analyzer, assigner)                 # the identity worker: this job's person's face

Boxes overlap; masks follow the body (decision #31). A face goes to a person only when all hold:
  size       the face is at least identity.min_face_px tall
  owned      at least identity.assign_own_min of the face lies on that person's mask (grown by
             identity.assign_grow_cells mask cells, for small faces and edges), and at most
             identity.assign_other_max on any other detection's mask, ID-less ones included
  core       both eyes and the nose lie on that person's (grown) mask
  head       the face centre is in the top identity.head_zone of that person's box, inside it
             sideways (a child carried in someone's arms is on the adult's mask, lower down)
  tracked    that person has a track ID (an ID-less detection never holds evidence)
  unique     no other face passes for the same person; if two do, neither counts

Anything else is ambiguous and gets track_id None: no gender, identity or eye-slit evidence
(rule 6, "ambiguous faces give nothing"). Every call says why per face (Assignment.reason).
"""
from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from core.contracts import MASK_CUT, Detections, FaceObservation, Frame, Track

GRID = 7                 # face area sampled as GRID x GRID points over the inner 80% of the face box
EYES_NOSE = (0, 1, 2)    # landmark rows: left eye, right eye, nose


@dataclass(frozen=True)
class Assignment:
    face: FaceObservation          # track_id set when assigned
    reason: str                    # "assigned" or why not: small, unowned, shared, core, not-head, id-less, two-faces
    shares: dict[int, float]       # mask_idx -> share of the face on that detection's mask (for debugging)


class FaceAssigner:
    def __init__(self, cfg: Any) -> None:
        i = cfg.identity
        self.min_face_px = float(i.min_face_px)
        self.own_min = float(i.get("assign_own_min", 0.6))
        self.other_max = float(i.get("assign_other_max", 0.3))
        self.grow_cells = int(i.get("assign_grow_cells", 1))
        self.head_zone = float(i.get("head_zone", 0.35))

    def assign(self, faces: Sequence[FaceObservation], detections: Detections) -> list[FaceObservation]:
        return [a.face for a in self.explain(faces, detections)]

    def explain(self, faces: Sequence[FaceObservation], detections: Detections) -> list[Assignment]:
        """assign(), with the reason for every face."""
        g = detections.geometry
        grown = self._grown_masks(detections)
        results: list[tuple[FaceObservation, str, dict[int, float], Track | None]] = []
        for face in faces:
            x1, y1, x2, y2 = (float(v) for v in face.bbox)
            if y2 - y1 < self.min_face_px:
                results.append((face, "small", {}, None))
                continue
            pts = g.frame_to_mask(_face_points(face.bbox))
            shares = {t.mask_idx: _share(grown[t.mask_idx], pts) for t in detections.tracks}
            ranked = sorted(detections.tracks, key=lambda t: -shares[t.mask_idx])
            if not ranked or shares[ranked[0].mask_idx] < self.own_min:
                results.append((face, "unowned", shares, None))
                continue
            owner = ranked[0]
            if len(ranked) > 1 and shares[ranked[1].mask_idx] > self.other_max:
                results.append((face, "shared", shares, None))
                continue
            core = g.frame_to_mask(np.asarray(face.landmarks)[list(EYES_NOSE)])
            if _share(grown[owner.mask_idx], core) < 1.0:
                results.append((face, "core", shares, None))
                continue
            bx1, by1, bx2, by2 = (float(v) for v in owner.bbox)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if not (bx1 <= cx <= bx2 and by1 <= cy <= by1 + self.head_zone * (by2 - by1)):
                results.append((face, "not-head", shares, None))
                continue
            if owner.track_id is None:
                results.append((face, "id-less", shares, None))
                continue
            results.append((face, "assigned", shares, owner))

        owners = [o.track_id for _, reason, _, o in results if reason == "assigned"]
        out = []
        for face, reason, shares, owner in results:
            if reason == "assigned" and owners.count(owner.track_id) > 1:
                reason, owner = "two-faces", None
            tid = owner.track_id if reason == "assigned" else None
            out.append(Assignment(dataclasses.replace(face, track_id=tid), reason, shares))
        return out

    def _grown_masks(self, detections: Detections) -> list[np.ndarray]:
        k = 2 * self.grow_cells + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)) if self.grow_cells > 0 else None
        out = []
        for m in detections.masks:
            binary = (m >= MASK_CUT).astype(np.uint8)
            out.append(cv2.dilate(binary, kernel) > 0 if kernel is not None else binary > 0)
        return out


def _face_points(bbox: np.ndarray) -> np.ndarray:
    """GRID x GRID points over the inner 80% of the face box (frame pixels)."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    fx = np.linspace(0.1, 0.9, GRID)
    xs, ys = np.meshgrid(x1 + fx * (x2 - x1), y1 + fx * (y2 - y1))
    return np.stack([xs.ravel(), ys.ravel()], axis=1)


def _share(mask: np.ndarray, pts: np.ndarray) -> float:
    """Share of points (mask pixels) that land on the mask; points outside the grid count as off it."""
    h, w = mask.shape
    ix, iy = np.floor(pts[:, 0]).astype(int), np.floor(pts[:, 1]).astype(int)
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    on = np.zeros(len(pts), bool)
    on[inside] = mask[iy[inside], ix[inside]]
    return float(on.mean()) if len(pts) else 0.0


# -- for the identity worker -------------------------------------------------------------

HEAD_REGION_PAD = 0.25    # the face search region: the top of the box, widened by this share of its width


def head_region(track: Track, head_zone: float, frame_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """Where to look for a person's face: the top head_zone of their box, a little wider and taller
    (a face may stick out of a tight box)."""
    x1, y1, x2, y2 = (float(v) for v in track.bbox)
    w, h = x2 - x1, y2 - y1
    pad = HEAD_REGION_PAD * w
    return (max(0.0, x1 - pad), max(0.0, y1 - 0.1 * h),
            min(float(frame_shape[1]), x2 + pad), min(float(frame_shape[0]), y1 + (head_zone + 0.1) * h))


def look(job: Any, analyzer: Any, assigner: FaceAssigner) -> FaceObservation | None:
    """The face of this job's person (a CheckJob, core/identity/component.py), or None when there
    is none, it is too small, or it is ambiguous. Faces of others found in the region take part in
    the assignment, so a neighbour's face is never taken for this person's."""
    frame: Frame = job.frame
    region = head_region(job.track, assigner.head_zone, frame.image.shape[:2])
    faces = analyzer.analyze(frame.image, frame.ts, frame.idx, region=region)
    mine = [f for f in assigner.assign(faces, job.detections) if f.track_id == job.track_id]
    return mine[0] if mine else None


__all__ = ["Assignment", "FaceAssigner", "head_region", "look"]
