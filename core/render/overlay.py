"""Debug overlay (T13): pipeline internals drawn onto an already-censored frame.

Per tracked person: box, ID, state, the rule that decided, confidence. Header:
rolling fps, latency, frame index. It copies a CensoredFrame's image and emits
a new CensoredFrame — the raw frame never passes through here, so nothing it
draws can leak pixels. Dev window only: the factory refuses it on the glasses
output (README §6.4).
"""
from __future__ import annotations

import cv2

from core.contracts import CensoredFrame, Decision, Detections, TrackState
from core.video.meters import LoopMeters

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_STATE_COLORS = {                           # BGR
    TrackState.UNDECIDED: (0, 170, 255),    # amber: concealed, being decided
    TrackState.CONCEALED: (0, 0, 220),      # red: decided conceal
    TrackState.REVERIFY: (0, 90, 255),      # orange: identity in doubt
    TrackState.PASSED: (0, 200, 0),         # green: decided show
}
_NO_DECISION = (0, 0, 220)                  # ID-less / undecided: concealed by default


class DebugOverlay:
    def __init__(self, text_scale: float = 0.45) -> None:
        self.text_scale = text_scale

    def apply(self, censored: CensoredFrame, detections: Detections | None,
              decisions: dict[int, Decision], meters: LoopMeters | None) -> CensoredFrame:
        """A new CensoredFrame with the debug drawing. ``detections=None`` = fail-safe frame."""
        img = censored.image.copy()
        cv2.putText(img, self._header(censored, detections, meters), (8, 20),
                    _FONT, self.text_scale, (255, 255, 255), 1, cv2.LINE_AA)
        for track in detections.tracks if detections is not None else ():
            x1, y1 = int(track.bbox[0]), int(track.bbox[1])
            x2, y2 = int(track.bbox[2]), int(track.bbox[3])
            decision = decisions.get(track.track_id) if track.track_id is not None else None
            if decision is None:
                color = _NO_DECISION
                who = "?" if track.track_id is None else str(track.track_id)
                label = f"#{who} no-decision {track.score:.2f}"
            else:
                color = _STATE_COLORS.get(decision.state, (255, 255, 255))
                label = (f"#{decision.track_id} {decision.state.value} "
                         f"{decision.rule} {decision.confidence:.2f}")
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            cv2.putText(img, label, (x1, max(12, y1 - 5)),
                        _FONT, self.text_scale, color, 1, cv2.LINE_AA)
        return CensoredFrame(image=img, ts=censored.ts, idx=censored.idx)

    @staticmethod
    def _header(censored: CensoredFrame, detections: Detections | None,
                meters: LoopMeters | None) -> str:
        parts = []
        if meters is not None:
            parts.append(f"{meters.fps:.1f} fps")
            if (lat := meters.latency_ms) is not None:
                parts.append(f"{lat:.0f} ms")
        parts.append(f"frame {censored.idx}")
        if detections is None:
            parts.append("FAIL-SAFE")
        return "  ".join(parts)
