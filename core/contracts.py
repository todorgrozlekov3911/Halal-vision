"""Contracts shared by every component (roadmap task T02).

The data types passed between components, and the interfaces each component
implements. Outside libraries (Ultralytics, InsightFace, ...) are used only
inside their wrapper; everything else speaks these types.

Rules every component relies on:

1. Only censored frames are shown. Sinks accept only ``CensoredFrame``, which
   only the renderer (and the debug overlay, on top of one) creates.
2. Frames are read-only. ``Frame.image`` is locked when the Frame is created;
   the renderer draws on a copy. The identity worker may still be reading a
   frame several frames later.
3. Sources hand out a fresh image for every frame, never a reused buffer, or
   rule 2 is broken from the other side.
4. Fail closed. ID-less detections, tracks without a Decision and anything
   undecided are concealed.
5. Time, not frames. Every "how long" rule is measured in seconds on
   ``Frame.ts``: capture clock live, video time offline (idx / video fps).
6. The detector owns track IDs. The identity module and the renderer keep
   per-track memory keyed by those IDs, and every component with memory has
   ``reset()`` so that all of them forget together when the source changes.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


def _lock(array: np.ndarray) -> np.ndarray:
    """Make an array read-only so an accidental write raises instead of corrupting."""
    array.flags.writeable = False
    return array


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Frame:
    """One input frame. Created by a Source, never modified afterwards."""

    image: np.ndarray        # BGR uint8 (H, W, 3), full resolution; locked on creation
    ts: float                # seconds: time.perf_counter() at capture live, idx / video fps offline
    idx: int                 # frame number within the source
    source_id: str

    def __post_init__(self) -> None:
        _lock(self.image)


@dataclass(frozen=True, eq=False)
class CensoredFrame:
    """A frame that is safe to show. Only the renderer creates these."""

    image: np.ndarray        # BGR uint8 (H, W, 3); locked on creation
    ts: float                # copied from the Frame it was rendered from (latency = now - ts)
    idx: int

    def __post_init__(self) -> None:
        _lock(self.image)


# --------------------------------------------------------------------------
# Detection output: one bundle per frame
# --------------------------------------------------------------------------

MASK_CUT = 128   # a mask pixel at or above this certainty (0..255) is the person


@dataclass(frozen=True)
class MaskGeometry:
    """Maps frame pixels to mask pixels (the pixel grid of ``Detections.masks``).

    The detector shrinks the frame and pads it (letterbox) before the model:
    ``mask_xy = frame_xy * scale + (pad_x, pad_y)``.
    """

    scale: float
    pad_x: float
    pad_y: float
    frame_shape: tuple[int, int]     # (height, width) of the full frame, like image.shape[:2]

    def frame_to_mask(self, xy: np.ndarray) -> np.ndarray:
        """Map points (..., 2) from frame pixels to mask pixels."""
        return np.asarray(xy, dtype=np.float32) * self.scale + np.array(
            [self.pad_x, self.pad_y], dtype=np.float32
        )

    def content_box(self) -> tuple[int, int, int, int]:
        """(x1, y1, x2, y2) of the frame's content in mask pixels, without the padding."""
        h, w = self.frame_shape
        return (
            int(round(self.pad_x)),
            int(round(self.pad_y)),
            int(round(self.pad_x + w * self.scale)),
            int(round(self.pad_y + h * self.scale)),
        )


@dataclass(frozen=True, eq=False)
class Track:
    """One detected person in one frame."""

    track_id: int | None     # None = detected but not tracked yet: always concealed, never gets a Decision
    bbox: np.ndarray         # (4,) float32 x1, y1, x2, y2 in frame pixels; locked
    score: float             # detector confidence
    mask_idx: int            # index into Detections.masks; use Detections.mask_of(track)

    def __post_init__(self) -> None:
        _lock(self.bbox)


@dataclass(frozen=True, eq=False)
class Detections:
    """Everything the detector found in one frame.

    Masks are whole-image masks at the model's small resolution (mask space),
    uint8 on the CPU, soft: 0..255 is how sure the model is that the pixel is this
    person, and >= MASK_CUT counts as the person. Soft values let the yes/no cut
    happen after enlarging, so outlines move smoothly. One mask per track, ID-less
    tracks included. Read them through ``mask_of`` and
    ``union_of`` so the storage format can change later without touching callers.
    """

    frame_idx: int
    ts: float
    tracks: tuple[Track, ...]
    masks: np.ndarray        # (N, h, w) uint8 0..255 certainty, N == len(tracks); locked
    geometry: MaskGeometry

    def __post_init__(self) -> None:
        if self.masks.ndim != 3 or self.masks.dtype != np.uint8:
            raise ValueError(f"masks must be (N, h, w) uint8, got {self.masks.shape} {self.masks.dtype}")
        if self.masks.shape[0] != len(self.tracks):
            raise ValueError(f"{self.masks.shape[0]} masks for {len(self.tracks)} tracks")
        _lock(self.masks)

    @property
    def mask_shape(self) -> tuple[int, int]:
        """(height, width) of the mask space."""
        return self.masks.shape[1], self.masks.shape[2]

    def mask_of(self, track: Track) -> np.ndarray:
        """One person's mask in mask space."""
        return self.masks[track.mask_idx]

    def union_of(self, tracks: Iterable[Track]) -> np.ndarray:
        """Combined mask of several people in mask space (all zeros for none)."""
        indices = [t.mask_idx for t in tracks]
        if not indices:
            return np.zeros(self.mask_shape, dtype=np.uint8)
        return self.masks[indices].max(axis=0)


# --------------------------------------------------------------------------
# Faces and evidence (produced and used inside the identity component)
# --------------------------------------------------------------------------

@dataclass(frozen=True, eq=False, kw_only=True)
class FaceObservation:
    """One face found by face analysis, converted from the library's own format.

    All coordinates are full-frame pixels, even when the face model ran on a crop.

    Privacy: embeddings of people who are not enrolled live in memory only, for
    the life of their track. Never cache or save them.
    """

    bbox: np.ndarray                         # (4,) float32 x1, y1, x2, y2; locked
    landmarks: np.ndarray                    # (5, 2) float32: left eye, right eye, nose, left mouth, right mouth; locked
    head_pose: tuple[float, float, float]    # (pitch, yaw, roll) degrees, 0 = facing the camera upright;
                                             # approximate, from the 5 landmarks
    det_score: float                         # face detector confidence
    quality: float                           # 0..1, how good this view is (size, sharpness, head pose)
    embedding: np.ndarray                    # (D,) float32, L2-normalized face fingerprint; locked
    embedding_model: str                     # e.g. "buffalo_l"; fingerprints from different models never compare
    p_female: float                          # 0..1, from the model's raw scores, not its rounded 0/1
    age: float                               # years, unrounded
    track_id: int | None = None              # set by face-to-subject assignment; None = ambiguous → no evidence
    ts: float
    frame_idx: int

    def __post_init__(self) -> None:
        _lock(self.bbox)
        _lock(self.landmarks)
        _lock(self.embedding)


class Modality(enum.Enum):
    """What the evidence was taken from. Body-only evidence never lifts a conceal by default."""

    FACE = "face"
    BODY = "body"


@dataclass(frozen=True, kw_only=True)
class Evidence:
    """One claim about one tracked person, from one look by one model."""

    track_id: int
    source: str              # which model produced it, e.g. "buffalo_l"; thresholds are tuned per model
    modality: Modality
    quality: float           # 0..1, how much to trust this look (not the claim's own probability)
    ts: float


@dataclass(frozen=True, kw_only=True)
class GenderEvidence(Evidence):
    p_female: float          # 0..1


@dataclass(frozen=True, kw_only=True)
class AgeEvidence(Evidence):
    years: float


@dataclass(frozen=True, kw_only=True)
class IdentityEvidence(Evidence):
    person_id: str           # the enrolled family member this look matched best
    similarity: float        # matcher score; accept / reject thresholds are per model


# --------------------------------------------------------------------------
# Decisions (identity → renderer)
# --------------------------------------------------------------------------

class TrackState(enum.Enum):
    """Where a tracked person is in the decision process. Which rule decided is ``Decision.rule``."""

    UNDECIDED = "undecided"  # new, or evidence below threshold: concealed, retried soon
    CONCEALED = "concealed"  # decided: conceal; re-checked on a slow timer
    PASSED = "passed"        # decided: show; re-checked on a timer
    REVERIFY = "reverify"    # identity in doubt (crossing, or a passed person gone briefly):
                             # concealed, first in line for a re-check


class ActionKind(enum.Enum):
    PASS = "pass"
    CONCEAL = "conceal"      # needs a style: "niqab" | "blur" | "pixelate" | "solid"
    DECORATE = "decorate"    # needs an asset; later phase


@dataclass(frozen=True)
class Action:
    """What the renderer does with one person."""

    kind: ActionKind
    style: str | None = None
    asset: str | None = None

    def __post_init__(self) -> None:
        if self.kind is ActionKind.CONCEAL and not self.style:
            raise ValueError("a conceal action needs a style")
        if self.kind is ActionKind.DECORATE and not self.asset:
            raise ValueError("a decorate action needs an asset")
        if self.kind is ActionKind.PASS and (self.style or self.asset):
            raise ValueError("a pass action takes no style or asset")


PASS = Action(ActionKind.PASS)
CONCEAL_NIQAB = Action(ActionKind.CONCEAL, style="niqab")


@dataclass(frozen=True)
class Decision:
    """The current action for one tracked person."""

    track_id: int
    action: Action           # the only field the renderer needs
    state: TrackState
    rule: str                # name of the rule that decided, from config: "whitelist", "male", "child",
                             # "crossing", "default", ...; for the debug overlay and evaluation
    confidence: float


# --------------------------------------------------------------------------
# Component interfaces
# --------------------------------------------------------------------------

class Source(ABC):
    """Camera or video file. Produces Frames; one fresh image per frame (rule 3)."""

    @abstractmethod
    def read(self) -> Frame | None:
        """The next frame, or None when a file has ended."""

    @property
    @abstractmethod
    def fps(self) -> float:
        """Nominal frames per second."""

    @property
    @abstractmethod
    def frame_shape(self) -> tuple[int, int]:
        """(height, width) of the frames."""

    @property
    @abstractmethod
    def is_live(self) -> bool:
        """True for cameras (ts from the clock), False for files read offline (ts from video time)."""

    @abstractmethod
    def close(self) -> None:
        """Release the camera or file."""

    def __enter__(self) -> Source:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class Sink(ABC):
    """Display or file output. Accepts only CensoredFrame; the check is built in."""

    def write(self, frame: CensoredFrame) -> None:
        if not isinstance(frame, CensoredFrame):
            raise TypeError(f"sinks accept only CensoredFrame, got {type(frame).__name__}")
        self._write(frame)

    @abstractmethod
    def _write(self, frame: CensoredFrame) -> None:
        """Show or store the frame. Must never block the main loop for long (file sinks use a writer thread)."""

    @abstractmethod
    def close(self) -> None:
        """Flush and release the output."""

    def __enter__(self) -> Sink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class Detector(ABC):
    """Finds, segments and tracks people. Owns the track IDs."""

    @abstractmethod
    def detect(self, frame: Frame) -> Detections:
        """Run on every processed frame. Returns ID-less detections too (track_id=None)."""

    @abstractmethod
    def reset(self) -> None:
        """Forget all tracks, e.g. when the source changes."""


class IdentityModule(ABC):
    """Decides, per tracked person, whether to conceal. Owns the decision state.

    Implementations take ``sync: bool`` in their constructor: False (live) never
    blocks; True (offline evaluation and export) finishes each frame's work before
    returning so results do not depend on timing.
    """

    @abstractmethod
    def update(self, frame: Frame, detections: Detections) -> dict[int, Decision]:
        """Called every processed frame.

        Must return a Decision for every track with a track_id, and none for
        ID-less ones. Anything not yet decided is concealed. Never blocks unless
        sync. Crossing and gap rules run here, in the same frame.
        """

    @abstractmethod
    def reset(self) -> None:
        """Forget all per-track state (evidence, decisions, timers)."""

    @abstractmethod
    def close(self) -> None:
        """Stop the worker thread."""


class Renderer(ABC):
    """Draws the conceal styles. The only creator of CensoredFrame.

    Keeps its own per-track memory (last masks for the mask hold, fade-out
    timing), all measured on Frame.ts.
    """

    @abstractmethod
    def render(self, frame: Frame, detections: Detections, decisions: dict[int, Decision]) -> CensoredFrame:
        """Draw on a copy of ``frame.image``, never on the original.

        Conceals ID-less tracks and tracks missing from ``decisions``. Concealing
        is instant; lifting a conceal may fade out.
        """

    @abstractmethod
    def reset(self) -> None:
        """Forget held masks and fades."""


# --------------------------------------------------------------------------
# Stubs
# --------------------------------------------------------------------------

class ConcealAllIdentity(IdentityModule):
    """Conceals everyone. Used by the main loop until the real identity module exists."""

    def __init__(self, action: Action = CONCEAL_NIQAB, sync: bool = False) -> None:
        if action.kind is not ActionKind.CONCEAL:
            raise ValueError("ConcealAllIdentity needs a conceal action")
        self.action = action
        self.sync = sync

    def update(self, frame: Frame, detections: Detections) -> dict[int, Decision]:
        return {
            t.track_id: Decision(t.track_id, self.action, TrackState.UNDECIDED, rule="stub", confidence=0.0)
            for t in detections.tracks
            if t.track_id is not None
        }

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass
