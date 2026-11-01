"""Search the live setup's settings (YOLO26n fixed) for the cover closest to the teacher labels.

    sg render -c ".venv/bin/python -m eval.sweep --machine intel clips/berlin_people_crossing.mp4"

(sg render: the NPU is only visible to processes in the render group; the model runs on it in
stage 1, so speeds are the live ones.)

  1. sizes    run YOLO26n at each input size (cached, see eval/score.py), note its NPU time per
              frame, score the current settings. Sizes slower than --budget-ms are dropped.
  2. random   --trials random settings from SPACE (sizes included), --workers in parallel
  3. refine   from the best so far, try every value of one setting at a time, keep improvements

Objective (lower is better), on people the renderer is meant to cover (eval/score.py):
    3 x missed% + partial% + leak_px% + over% + 0.1 x id_switch/min
taken as the mean over the two halves of each clip, so a setting that only helps one stretch of
video does not win. Results: clips/labels/_sweep/<time>.jsonl (every trial) and the best settings.
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

from core.config_table import available_machines, load_config
from eval import score

SPACE: dict[str, list] = {
    "detector.imgsz": None,                      # filled with the sizes that passed stage 1
    "detector.conf": [0.05, 0.08, 0.1, 0.15, 0.2, 0.25],
    "detector.tracker": ["bytetrack", "botsort"],
    "tracker.track_high_thresh": [0.15, 0.2, 0.25, 0.35],
    "tracker.track_low_thresh": [0.05, 0.1],
    "tracker.new_track_thresh": [0.25, 0.3, 0.4, 0.5],
    "tracker.match_thresh": [0.7, 0.8, 0.9],
    "timing.track_buffer_seconds": [1.0, 1.5, 2.5],
    "timing.mask_hold_seconds": [0.3, 0.5, 0.8, 1.2],
    "timing.mask_smooth_seconds": [0.0, 0.1, 0.15, 0.25],
    "render.mask_grow_fraction": [0.02, 0.04, 0.06, 0.08],
    "render.min_person_height": [0.10, 0.11, 0.12],
}


def objective(sm: dict) -> float:
    return 3 * sm["missed_%"] + sm["partial_%"] + sm["leak_px_%"] + sm["over_%"] + 0.1 * sm["id_switch/min"]


def _sets(params: dict) -> tuple[list[str], dict]:
    sets = [f"{k}={json.dumps(v)}" for k, v in params.items() if not k.startswith("tracker.")]
    tracker = {k.split(".", 1)[1]: v for k, v in params.items() if k.startswith("tracker.")}
    return sets, tracker


def run_trial(job: tuple) -> dict:
    """One setting, every clip. (Runs in a worker process.)"""
    machine, clips, params, eval_min_height, labels = job
    score.use_labels(Path(labels))
    sets, tracker = _sets(params)
    cfg = load_config(machine, sets)
    t0 = time.perf_counter()
    total, parts = score.evaluate(cfg, [Path(c) for c in clips], None, eval_min_height, tracker)
    halves = [objective(s.summary()) for s in parts.values()]
    return {"params": params, "objective": sum(halves) / len(halves), "halves": halves,
            "total": total.summary(), "seconds": round(time.perf_counter() - t0, 1)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eval.sweep", description=__doc__.split("\n")[0])
    p.add_argument("clips", nargs="*", type=Path)
    p.add_argument("--machine", required=True, choices=available_machines())
    p.add_argument("--sizes", default="384x640,448x800,512x896,576x1024")
    p.add_argument("--budget-ms", type=float, default=35.0, help="max model time per frame (of the 66 ms at 15 fps)")
    p.add_argument("--trials", type=int, default=120)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--eval-min-height", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    clips = [str(c) for c in (args.clips or score.labelled_clips())]
    out = score.LABELS / "_sweep" / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.open("a")
    started = time.time()

    def record(stage: str, r: dict) -> None:
        log.write(json.dumps({"stage": stage, **r}) + "\n"); log.flush()

    def show(r: dict) -> str:
        t = r["total"]
        return (f"obj {r['objective']:6.2f} (halves {r['halves'][0]:.2f} / {r['halves'][1]:.2f})  "
                f"missed {t['missed_%']:.2f}%  partial {t['partial_%']:.2f}%  leak {t['leak_px_%']:.2f}%  "
                f"over {t['over_%']:.2f}%  idsw {t['id_switch/min']:.1f}/min")

    # 1. sizes: model runs here (on the NPU when started under sg render), one size at a time
    print(f"[sizes] {args.sizes}", flush=True)
    feasible, baseline = [], None
    for size in args.sizes.split(","):
        hw = [int(v) for v in size.split("x")]
        params = {"detector.imgsz": hw}
        cfg = load_config(args.machine, _sets(params)[0])
        for c in clips:
            cache = score.ModelCache.load_or_build(cfg, Path(c), max(1, round(
                score.load_labels(Path(c).stem)["fps"] / float(cfg.expected_fps))))
        r = run_trial((args.machine, clips, params, args.eval_min_height, str(score.LABELS)))
        r["infer_ms"], r["device"] = cache.infer_ms, cache.device
        record("size", r)
        ok = cache.infer_ms <= args.budget_ms
        print(f"  {size:>9s}  {cache.infer_ms:5.1f} ms on {cache.device:<6s} {'ok ' if ok else 'too slow'}  {show(r)}",
              flush=True)
        if ok:
            feasible.append(hw)
        if baseline is None:
            baseline = r
    if not feasible:
        print("no size within the time budget"); return 1
    SPACE["detector.imgsz"] = feasible

    # 2. random settings, in parallel (cached model outputs only: no NPU needed)
    rng = random.Random(args.seed)
    trials = [{k: rng.choice(v) for k, v in SPACE.items()} for _ in range(args.trials)]
    jobs = [(args.machine, clips, t, args.eval_min_height, str(score.LABELS)) for t in trials]
    results = []
    print(f"[random] {len(jobs)} trials on {args.workers} workers", flush=True)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_trial, jobs), 1):  # noqa
            record("random", r); results.append(r)
            best = min(results, key=lambda x: x["objective"])
            if i % 10 == 0:
                print(f"  {i}/{len(jobs)}  {(time.time() - started) / 60:.0f} min  best {show(best)}", flush=True)

        # 3. refine: one setting at a time from the best, all its values in parallel
        best = min(results, key=lambda x: x["objective"])
        print("[refine]", flush=True)
        improved = True
        while improved:
            improved = False
            for key, values in SPACE.items():
                variants = [{**best["params"], key: v} for v in values if v != best["params"][key]]
                for r in pool.imap_unordered(run_trial, [(args.machine, clips, v, args.eval_min_height,
                                                          str(score.LABELS)) for v in variants]):
                    record("refine", r)
                    if r["objective"] < best["objective"] - 1e-9:
                        best, improved = r, True
                        print(f"  {key} = {r['params'][key]}: {show(r)}", flush=True)

    print(f"\nbaseline (current settings): {show(baseline)}")
    print(f"best:                        {show(best)}")
    print("best settings:")
    for k, v in best["params"].items():
        print(f"  {k}: {v}")
    print(f"all trials: {out}   ({(time.time() - started) / 60:.0f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
