"""Identity component skeleton (T15): who gets checked when, on a worker thread that never blocks.

    identity = IdentityComponent(cfg, sync=False)      # live: checks run on a worker thread
    decisions = identity.update(frame, detections)     # every processed frame; returns at once

update() only bookkeeps and reads stored decisions; the slow part (face analysis, gender, whitelist:
a "check") runs on the worker. What a check does is plugged in, so later tasks add to it without
touching the scheduling here:

    Checker   (job) -> evidence list       T16-T21: face analysis, gender, age, whitelist
    Policy    (track, new evidence) -> Decision    T22: the rules; until then everyone is concealed

Scheduling, all timed on Frame.ts (rule 5):
  - a track is checked when it is first seen, then again when its re-check time comes:
        undecided  timing.retry_unsure_seconds
        concealed  timing.recheck_gender_seconds
        passed     timing.recheck_whitelist_seconds
        reverify   at once
  - queue order: new tracks and reverify first, then re-checks, oldest first. The queue is short
    (QUEUE_SIZE); when it is full, a more urgent job pushes out the least urgent one, and a job
    that does not fit is simply tried again next frame
  - a track with a check queued or running is "pending" and not queued twice
  - a track not seen for timing.track_buffer_seconds (the tracker has dropped it by then) is
    forgotten, evidence included (privacy: strangers' face data lives only as long as their track);
    its queued jobs are skipped

Fail closed (rule 4): anything without a decision is concealed; a check that raises leaves the
track as it was (concealed unless decided) and retries it later. A Decision is only ever returned
for a track_id in this frame's detections; ID-less detections get none (the renderer conceals them).

sync=True (offline export, evaluation): each frame's checks run inside update() before it returns,
so results do not depend on timing.
"""
from __future__ import annotations

import heapq
import itertools
import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from core.contracts import (Action, ActionKind, Decision, Detections, Evidence, Frame, IdentityModule, Track,
                            TrackState)

log = logging.getLogger("halal_vision.identity")

QUEUE_SIZE = 8                 # jobs waiting at most; each holds a frame (~6 MB at 1080p)
URGENT, RECHECK = 0, 1         # queue priority: new / reverify tracks before re-checks
CLOSE_TIMEOUT = 2.0            # seconds close() waits for a running check


@dataclass(frozen=True)
class CheckJob:
    """One check of one tracked person. Frame and Detections are read-only (rule 2), so the
    worker can use them after the main loop has moved on."""
    track_id: int
    frame: Frame
    detections: Detections
    track: Track
    generation: int            # reset() bumps the generation; results of older jobs are dropped


class Checker(Protocol):
    def __call__(self, job: CheckJob) -> Sequence[Evidence]: ...


class Policy(Protocol):
    def __call__(self, track: TrackRecord, new: Sequence[Evidence]) -> Decision: ...


@dataclass
class TrackRecord:
    """What the component remembers about one track. Evidence stays in memory only."""
    track_id: int
    first_seen: float
    last_seen: float
    decision: Decision
    evidence: list[Evidence] = field(default_factory=list)
    checks: int = 0            # finished checks
    pending: bool = False      # a check is queued or running
    next_check: float = 0.0    # Frame.ts when the next check is due


def no_checks(job: CheckJob) -> Sequence[Evidence]:
    """The checker until face analysis exists (T16): looks at nothing, finds nothing."""
    return ()


class ConcealAllPolicy:
    """The policy until the rule engine exists (T22): everyone stays concealed, undecided."""

    def __init__(self, action: Action) -> None:
        self.action = action

    def __call__(self, track: TrackRecord, new: Sequence[Evidence]) -> Decision:
        return Decision(track.track_id, self.action, TrackState.UNDECIDED, rule="default", confidence=0.0)


class IdentityComponent(IdentityModule):
    def __init__(self, cfg: Any, sync: bool = False, checker: Checker = no_checks,
                 policy: Policy | None = None) -> None:
        t = cfg.timing
        self.sync = sync
        self.conceal = Action(ActionKind.CONCEAL, style=str(cfg.policy.conceal_style))
        self.checker = checker
        self.policy = policy or ConcealAllPolicy(self.conceal)
        self.recheck_after = {
            TrackState.UNDECIDED: float(t.retry_unsure_seconds),
            TrackState.CONCEALED: float(t.recheck_gender_seconds),
            TrackState.PASSED: float(t.recheck_whitelist_seconds),
            TrackState.REVERIFY: 0.0,
        }
        self.forget_after = float(t.track_buffer_seconds)

        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._tracks: dict[int, TrackRecord] = {}
        self._queue: list[tuple[int, float, int, CheckJob]] = []   # heap: (priority, due, seq, job)
        self._seq = itertools.count()
        self._generation = 0
        self._closing = False
        self.errors = 0            # checks that raised (for status lines and tests)
        self._worker: threading.Thread | None = None
        if not sync:
            self._worker = threading.Thread(target=self._work, name="identity", daemon=True)
            self._worker.start()

    # -- IdentityModule ---------------------------------------------------------------

    def update(self, frame: Frame, detections: Detections) -> dict[int, Decision]:
        now = frame.ts
        due: list[CheckJob] = []
        with self._lock:
            for track in detections.tracks:
                if track.track_id is None:
                    continue                                  # ID-less: concealed by the renderer
                rec = self._tracks.get(track.track_id)
                if rec is None or now < rec.last_seen:        # new, or time went backwards (seek)
                    rec = self._tracks[track.track_id] = self._new_record(track.track_id, now)
                rec.last_seen = now
                if not rec.pending and now >= rec.next_check:
                    due.append(CheckJob(track.track_id, frame, detections, track, self._generation))
            self._forget_gone(now)
            if not self.sync:
                for job in due:
                    self._enqueue(job)
        if self.sync:
            for job in due:
                self._run(job)
        with self._lock:
            return {t.track_id: self._tracks[t.track_id].decision
                    for t in detections.tracks if t.track_id is not None and t.track_id in self._tracks}

    def reset(self) -> None:
        with self._lock:
            self._generation += 1
            self._tracks.clear()
            self._queue.clear()

    def close(self) -> None:
        with self._lock:
            self._closing = True
            self._queue.clear()
            self._wake.notify_all()
        if self._worker is not None:
            self._worker.join(CLOSE_TIMEOUT)
            if self._worker.is_alive():
                log.warning("identity worker still busy after %.1f s; left to finish as a daemon", CLOSE_TIMEOUT)

    # -- for the debug overlay and tests ------------------------------------------------

    def snapshot(self) -> dict[int, TrackRecord]:
        """A copy of the per-track records (evidence lists copied too)."""
        with self._lock:
            return {k: TrackRecord(**{**vars(r), "evidence": list(r.evidence)}) for k, r in self._tracks.items()}

    def queued(self) -> int:
        with self._lock:
            return len(self._queue)

    # -- internals (callers hold the lock unless noted) ---------------------------------

    def _new_record(self, track_id: int, now: float) -> TrackRecord:
        decision = Decision(track_id, self.conceal, TrackState.UNDECIDED, rule="default", confidence=0.0)
        return TrackRecord(track_id, first_seen=now, last_seen=now, decision=decision, next_check=now)

    def _forget_gone(self, now: float) -> None:
        for tid in [tid for tid, r in self._tracks.items() if now - r.last_seen > self.forget_after]:
            del self._tracks[tid]

    def _priority(self, job: CheckJob) -> int:
        rec = self._tracks[job.track_id]
        return URGENT if rec.checks == 0 or rec.decision.state is TrackState.REVERIFY else RECHECK

    def _enqueue(self, job: CheckJob) -> None:
        item = (self._priority(job), self._tracks[job.track_id].next_check, next(self._seq), job)
        if len(self._queue) >= QUEUE_SIZE:
            worst = max(self._queue)
            if item >= worst:
                return                                        # full of more urgent work: try next frame
            self._queue.remove(worst)
            heapq.heapify(self._queue)
            if (dropped := self._tracks.get(worst[3].track_id)) is not None:
                dropped.pending = False                       # it will be queued again when due
        heapq.heappush(self._queue, item)
        self._tracks[job.track_id].pending = True
        self._wake.notify()

    def _work(self) -> None:
        """Worker thread: take the most urgent job, run it outside the lock."""
        while True:
            with self._lock:
                while not self._queue and not self._closing:
                    self._wake.wait()
                if self._closing:
                    return
                job = heapq.heappop(self._queue)[3]
                if job.generation != self._generation or job.track_id not in self._tracks:
                    continue                                  # reset since, or the track is gone
            self._run(job)

    def _run(self, job: CheckJob) -> None:
        """Run one check (no lock held while the checker runs) and store what it decided."""
        try:
            evidence = list(self.checker(job))
            failed = False
        except Exception:
            log.exception("identity check of track %d failed; it stays concealed", job.track_id)
            evidence, failed = [], True
        with self._lock:
            self.errors += failed
            rec = self._tracks.get(job.track_id)
            if rec is None or job.generation != self._generation:
                return                                        # forgotten or reset meanwhile
            rec.pending = False
            if failed:
                rec.next_check = job.frame.ts + self.recheck_after[TrackState.UNDECIDED]
                return
            rec.evidence.extend(evidence)
            rec.checks += 1
            try:
                decision = self.policy(rec, evidence)
            except Exception:
                log.exception("identity policy failed for track %d; concealing", job.track_id)
                self.errors += 1
                decision = Decision(job.track_id, self.conceal, TrackState.UNDECIDED, rule="error", confidence=0.0)
            rec.decision = _fail_closed(decision, job.track_id, self.conceal)
            rec.next_check = job.frame.ts + self.recheck_after[rec.decision.state]


def _fail_closed(decision: Decision, track_id: int, conceal: Action) -> Decision:
    """A decision for another track, or a pass that is not in the PASSED state, is a bug: conceal."""
    wrong_track = decision.track_id != track_id
    shows = decision.action.kind is not ActionKind.CONCEAL
    if wrong_track or (shows and decision.state is not TrackState.PASSED):
        log.error("identity policy returned %r for track %d; concealing", decision, track_id)
        return Decision(track_id, conceal, TrackState.UNDECIDED, rule="error", confidence=0.0)
    return decision


def make_identity(cfg: Any, sync: bool, checker: Checker | None = None,
                  policy: Policy | None = None) -> IdentityComponent:
    """The identity component as configured (checker and policy grow with T16-T22)."""
    return IdentityComponent(cfg, sync=sync, checker=checker or no_checks, policy=policy)


__all__ = ["CheckJob", "Checker", "ConcealAllPolicy", "IdentityComponent", "Policy", "TrackRecord",
           "make_identity", "no_checks"]
