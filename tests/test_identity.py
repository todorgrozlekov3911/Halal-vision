"""T15: identity component skeleton (scheduling, worker thread, fail-closed), with stand-in checkers.

    .venv/bin/python -m pytest -q tests/test_identity.py
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from core.config_table import load_config
from core.contracts import (PASS, Action, ActionKind, Decision, Detections, Frame, GenderEvidence, Modality, Track,
                            TrackState)
from core.detect.detector import _mask_geometry
from core.identity import component
from core.identity.component import IdentityComponent

GEO = _mask_geometry((72, 128), (64, 128), (16, 32))
IMAGE = np.zeros((72, 128, 3), np.uint8)
CONCEAL = Action(ActionKind.CONCEAL, style="niqab")


def cfg(*sets):
    return load_config("intel", ["timing.retry_unsure_seconds=0.3", "timing.recheck_gender_seconds=2.0",
                                 "timing.recheck_whitelist_seconds=1.0", "timing.track_buffer_seconds=1.5", *sets])


def scene(ts: float, ids, idx: int = 0):
    """A frame at time ts with one detection per id (None = ID-less)."""
    frame = Frame(IMAGE, ts, idx, "test")
    tracks = tuple(Track(tid, np.array([0, 0, 10, 20], np.float32), 0.9, k) for k, tid in enumerate(ids))
    return frame, Detections(idx, ts, tracks, np.zeros((len(ids), 16, 32), np.uint8), GEO)


class Recorder:
    """Checker that records (track_id, frame ts); can be held back with a gate."""

    def __init__(self, gate: threading.Event | None = None, fail: bool = False) -> None:
        self.calls: list[tuple[int, float]] = []
        self.gate, self.fail = gate, fail
        self.started = threading.Event()

    def __call__(self, job):
        self.started.set()
        if self.gate is not None:
            self.gate.wait(5)
        self.calls.append((job.track_id, job.frame.ts))
        if self.fail:
            raise RuntimeError("model crashed")
        return [GenderEvidence(track_id=job.track_id, source="test", modality=Modality.FACE, quality=1.0,
                               ts=job.frame.ts, p_female=0.1)]


def wait_until(cond, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def closing():
    made = []
    yield made.append
    for identity in made:
        identity.close()


def test_conceals_every_tracked_person_and_gives_id_less_ones_nothing(closing):
    identity = IdentityComponent(cfg(), sync=False)
    closing(identity)
    decisions = identity.update(*scene(0.0, [1, 2, None]))
    assert set(decisions) == {1, 2}
    for tid, d in decisions.items():
        assert d.track_id == tid and d.action.kind is ActionKind.CONCEAL and d.action.style == "niqab"
        assert d.state is TrackState.UNDECIDED


def test_update_never_waits_for_a_slow_check(closing):
    gate = threading.Event()
    checker = Recorder(gate)
    identity = IdentityComponent(cfg(), sync=False, checker=checker)
    closing(identity)
    worst = 0.0
    for i in range(30):
        t0 = time.perf_counter()
        decisions = identity.update(*scene(i / 30, [1, 2, 3], i))
        worst = max(worst, time.perf_counter() - t0)
        assert all(d.action.kind is ActionKind.CONCEAL for d in decisions.values())
    assert checker.started.wait(1)          # the worker is inside a check the whole time
    assert worst < 0.02
    gate.set()


def test_a_pending_track_is_not_queued_twice(closing):
    gate = threading.Event()
    checker = Recorder(gate)
    identity = IdentityComponent(cfg(), sync=False, checker=checker)
    closing(identity)
    for i in range(20):                     # retry time (0.3 s) passes while the first check hangs
        identity.update(*scene(i * 0.05, [1], i))
    gate.set()
    assert wait_until(lambda: len(checker.calls) == 1)
    time.sleep(0.05)
    assert checker.calls == [(1, 0.0)]


def test_new_people_are_checked_before_re_checks(closing):
    gate = threading.Event()
    checker = Recorder(gate)
    identity = IdentityComponent(cfg(), sync=False, checker=checker)
    closing(identity)
    identity.update(*scene(0.0, [1]))                      # 1: checked first (worker holds it)
    assert checker.started.wait(1)
    gate.set()
    assert wait_until(lambda: identity.snapshot()[1].checks == 1)
    gate.clear()
    checker.started.clear()
    identity.update(*scene(0.1, [9]))                      # 9: new, keeps the worker busy
    assert checker.started.wait(1)
    identity.update(*scene(0.5, [1, 9]))                   # 1 due for a re-check (0.3 s)
    identity.update(*scene(0.6, [1, 9, 2, 3]))             # 2, 3: new, queued after the re-check
    gate.set()
    assert wait_until(lambda: len(checker.calls) == 5)
    order = [tid for tid, _ in checker.calls]
    assert order[:2] == [1, 9]
    assert order[2:4] == [2, 3] and order[4] == 1


def test_re_check_times_follow_the_decision_state():
    checker = Recorder()
    identity = IdentityComponent(cfg(), sync=True, checker=checker)
    for i, ts in enumerate([0.0, 0.1, 0.2, 0.29, 0.31, 0.5, 0.62]):
        identity.update(*scene(ts, [1], i))
    assert [ts for _, ts in checker.calls] == [0.0, 0.31, 0.62]   # undecided: every 0.3 s


def test_passed_people_are_re_checked_on_the_whitelist_timer():
    def policy(rec, new):
        return Decision(rec.track_id, PASS, TrackState.PASSED, rule="whitelist", confidence=0.99)
    checker = Recorder()
    identity = IdentityComponent(cfg(), sync=True, checker=checker, policy=policy)
    decisions = {}
    for i, ts in enumerate([0.0, 0.5, 0.99, 1.0, 1.5, 2.0]):
        decisions = identity.update(*scene(ts, [1], i))
    assert [ts for _, ts in checker.calls] == [0.0, 1.0, 2.0]
    assert decisions[1].action is PASS


def test_evidence_is_kept_per_track_and_forgotten_with_the_track():
    identity = IdentityComponent(cfg(), sync=True, checker=Recorder())
    identity.update(*scene(0.0, [1, 2]))
    assert len(identity.snapshot()[1].evidence) == 1
    identity.update(*scene(1.0, [2]))
    assert set(identity.snapshot()) == {1, 2}              # 1 unseen for 1.0 s < track buffer
    identity.update(*scene(1.6, [2]))
    assert set(identity.snapshot()) == {2}                 # 1 unseen for 1.6 s: forgotten


def test_queued_jobs_of_a_forgotten_track_are_skipped(closing):
    gate = threading.Event()
    checker = Recorder(gate)
    identity = IdentityComponent(cfg(), sync=False, checker=checker)
    closing(identity)
    identity.update(*scene(0.0, [1]))
    assert checker.started.wait(1)                         # worker busy with 1
    identity.update(*scene(0.1, [1, 2]))                   # 2 queued
    identity.update(*scene(2.0, [1]))                      # 2 unseen for 1.9 s: forgotten
    gate.set()
    assert wait_until(lambda: identity.queued() == 0)
    time.sleep(0.05)
    assert 2 not in [tid for tid, _ in checker.calls]


def test_a_failing_check_leaves_the_person_concealed_and_is_retried():
    checker = Recorder(fail=True)
    identity = IdentityComponent(cfg(), sync=True, checker=checker)
    d = identity.update(*scene(0.0, [1]))[1]
    assert d.action.kind is ActionKind.CONCEAL and identity.errors == 1
    identity.update(*scene(0.31, [1], 1))
    assert len(checker.calls) == 2


def test_a_policy_that_shows_someone_undecided_is_overruled():
    def bad(rec, new):
        return Decision(rec.track_id, PASS, TrackState.UNDECIDED, rule="bug", confidence=1.0)
    identity = IdentityComponent(cfg(), sync=True, checker=Recorder(), policy=bad)
    d = identity.update(*scene(0.0, [1]))[1]
    assert d.action.kind is ActionKind.CONCEAL and d.rule == "error"


def test_a_policy_that_raises_conceals():
    def broken(rec, new):
        raise ValueError("rule bug")
    identity = IdentityComponent(cfg(), sync=True, checker=Recorder(), policy=broken)
    assert identity.update(*scene(0.0, [1]))[1].action.kind is ActionKind.CONCEAL


def test_reset_drops_results_of_checks_still_running(closing):
    gate = threading.Event()
    checker = Recorder(gate)

    def policy(rec, new):
        return Decision(rec.track_id, PASS, TrackState.PASSED, rule="whitelist", confidence=0.99)
    identity = IdentityComponent(cfg(), sync=False, checker=checker, policy=policy)
    closing(identity)
    identity.update(*scene(5.0, [1]))
    assert checker.started.wait(1)
    identity.reset()                                       # e.g. the source changed
    gate.set()
    assert wait_until(lambda: len(checker.calls) == 1)
    time.sleep(0.05)
    d = identity.update(*scene(0.0, [1]))[1]               # same ID, new source: must not inherit the pass
    assert d.action.kind is ActionKind.CONCEAL


def test_a_full_queue_makes_room_for_new_people(closing, monkeypatch):
    monkeypatch.setattr(component, "QUEUE_SIZE", 2)
    gate = threading.Event()
    checker = Recorder(gate)
    identity = IdentityComponent(cfg("timing.retry_unsure_seconds=0.01"), sync=False, checker=checker)
    closing(identity)
    identity.update(*scene(0.0, [1]))
    assert checker.started.wait(1)                         # worker holds 1
    gate.set()
    assert wait_until(lambda: identity.snapshot()[1].checks == 1)
    gate.clear()
    checker.started.clear()
    identity.update(*scene(0.1, [7]))                      # worker holds 7
    assert checker.started.wait(1)
    identity.update(*scene(0.2, [1, 7, 2]))                # queue: 2 (new), 1 (re-check) -> full
    identity.update(*scene(0.3, [1, 7, 2, 3]))             # 3 (new) pushes out the re-check of 1
    gate.set()
    assert wait_until(lambda: identity.queued() == 0 and len(checker.calls) >= 4)
    assert [tid for tid, _ in checker.calls][:4] == [1, 7, 2, 3]


def test_sync_mode_gives_the_same_decisions_every_run():
    def policy(rec, new):                  # passes a person after two checks
        if rec.checks >= 2:
            return Decision(rec.track_id, PASS, TrackState.PASSED, "test", 1.0)
        return Decision(rec.track_id, CONCEAL, TrackState.UNDECIDED, "test", 0.0)
    runs = []
    for _ in range(2):
        identity = IdentityComponent(cfg(), sync=True, checker=Recorder(), policy=policy)
        runs.append([{k: d.action.kind for k, d in identity.update(*scene(i * 0.1, [1, 2], i)).items()}
                     for i in range(12)])
    assert runs[0] == runs[1]
    assert runs[0][0][1] is ActionKind.CONCEAL and runs[0][-1][1] is ActionKind.PASS


def test_close_stops_the_worker_quickly():
    identity = IdentityComponent(cfg(), sync=False)
    t0 = time.perf_counter()
    identity.close()
    assert time.perf_counter() - t0 < 0.5
    assert not identity._worker.is_alive()
