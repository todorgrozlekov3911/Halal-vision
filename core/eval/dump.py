"""Per-frame decision dump (T13): JSON Lines, one object per shown frame.

The machine-readable twin of the debug overlay, for the evaluation agent:
what the pipeline knew and decided each frame (track IDs, boxes, state, rule,
confidence) plus the meters' fps/latency. Join it with per-track ground-truth
labels to score a run. Pixels never enter the file. Each line is flushed so a
killed run still leaves a usable record.

Line shape:
    {"frame": 12, "ts": 0.4, "failsafe": false, "fps": 28.1, "latency_ms": 71.2,
     "tracks": [{"id": 7, "bbox": [204.0, 88.5, 341.0, 460.2], "score": 0.91,
                 "state": "concealed", "rule": "stub", "confidence": 0.9,
                 "action": "conceal:niqab"}]}
"""
from __future__ import annotations

import json
from pathlib import Path

from core.contracts import CensoredFrame, Decision, Detections, Track
from core.video.meters import LoopMeters


class DecisionDump:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file = self.path.open("w", encoding="utf-8")

    def write(self, censored: CensoredFrame, detections: Detections | None,
              decisions: dict[int, Decision], meters: LoopMeters | None) -> None:
        """One line per shown frame. ``detections=None`` = fail-safe frame."""
        latency = meters.latency_ms if meters is not None else None
        record = {
            "frame": censored.idx,
            "ts": round(censored.ts, 4),
            "failsafe": detections is None,
            "fps": round(meters.fps, 2) if meters is not None else None,
            "latency_ms": round(latency, 1) if latency is not None else None,
            "tracks": [self._track(t, decisions)
                       for t in (detections.tracks if detections is not None else ())],
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    @staticmethod
    def _track(track: Track, decisions: dict[int, Decision]) -> dict:
        rec = {"id": track.track_id,
               "bbox": [round(float(v), 1) for v in track.bbox],
               "score": round(track.score, 3)}
        decision = decisions.get(track.track_id) if track.track_id is not None else None
        if decision is None:
            # ID-less or not yet decided: concealed by default (fail closed)
            rec.update(state=None, rule=None, confidence=None, action="conceal")
        else:
            action = decision.action.kind.value
            if decision.action.style:
                action += f":{decision.action.style}"
            rec.update(state=decision.state.value, rule=decision.rule,
                       confidence=round(decision.confidence, 3), action=action)
        return rec

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()