"""Side-by-side video: the live cover with the current settings vs the tuned ones, on the teacher labels.

    python -m eval.compare --machine intel clips/berlin_people_crossing.mp4                 # best of the latest sweep
    python -m eval.compare --machine intel clips/... --tuned clips/labels/_sweep/<time>.jsonl

Label frames only (5 per second, so it plays at real speed). In each half:
    dark fill       where the live pipeline covers
    green outline   a teacher person the renderer must cover (tall enough, not "ignore")
    red             the part of such a person left uncovered (a leak)
    grey outline    teacher people too small or too unsure to be scored
Output: clips/labels/<clip>/compare.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from core.config_table import available_machines, load_config
from core.contracts import Frame
from core.render.renderer import StyleRenderer
from eval import score
from eval.sweep import _sets

FILL = np.array([40, 30, 25], np.float32)      # near-black (BGR), shown at 75% so the scene stays readable
GREEN, RED, GREY = (60, 220, 60), (40, 40, 255), (170, 170, 170)


class Pipeline:
    def __init__(self, name: str, machine: str, params: dict, fps: float, stride: int, clip: Path) -> None:
        sets, tracker = _sets(params)
        cfg = load_config(machine, sets)
        self.name, self.cache = name, score.ModelCache.load_or_build(cfg, clip, stride)
        self.detector = score.ReplayDetector(cfg, fps / stride, self.cache)
        if tracker:
            from ultralytics.utils import IterableSimpleNamespace
            args = {**vars(self.detector.tracker_args), **tracker}
            self.detector.tracker_args = IterableSimpleNamespace(**args)
            self.detector._tracker_cls = score.TRACKER_MAP[args["tracker_type"]]
            self.detector._predict_conf = min(self.detector.conf, float(args["track_low_thresh"]))
            self.detector.reset()
        self.renderer = StyleRenderer(cfg)
        self.renderer.set_default_style("solid")
        self.renderer.keep_cover = True
        self.missed = 0

    def step(self, frame: Frame) -> np.ndarray:
        dets = self.detector.detect(frame)
        self.renderer.render(frame, dets, {})
        return self.renderer.last_cover >= score.COVERED


def best_params(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return min(rows, key=lambda r: r["objective"])["params"]


def draw(image: np.ndarray, covered: np.ndarray, inst: np.ndarray, frame_labels: dict, h: int,
         eval_min_height: float, title: str) -> tuple[np.ndarray, int]:
    out = image.astype(np.float32)
    out[covered] = out[covered] * 0.25 + FILL * 0.75
    out = out.astype(np.uint8)
    missed = 0
    for p in frame_labels["people"]:
        if "id" not in p:
            continue
        mine = inst == p["id"]
        if not mine.any():
            continue
        scored = (not p["ignore"] and p["visible_area"] >= score.MIN_VISIBLE_PX
                  and (p["box"][3] - p["box"][1]) / h >= eval_min_height)
        contours, _ = cv2.findContours(mine.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if scored:
            leak = mine & ~covered
            out[leak] = (out[leak] * 0.3 + np.array(RED) * 0.7).astype(np.uint8)
            missed += leak.sum() > score.LEAK_MISSED * mine.sum()
            cv2.drawContours(out, contours, -1, GREEN, 3)
        else:
            cv2.drawContours(out, contours, -1, GREY, 1)
    text = f"{title}   missed here: {missed}"
    cv2.putText(out, text, (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(out, text, (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
    return out, missed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eval.compare", description=__doc__.split("\n")[0])
    p.add_argument("clip", type=Path)
    p.add_argument("--machine", required=True, choices=available_machines())
    p.add_argument("--tuned", type=Path, help="sweep results (.jsonl); default: the latest")
    p.add_argument("--eval-min-height", type=float, default=0.12)
    args = p.parse_args(argv)
    sweep = args.tuned or sorted((score.LABELS / "_sweep").glob("*.jsonl"))[-1]
    tuned = best_params(sweep)
    labels = score.load_labels(args.clip.stem)
    fps, h = labels["fps"], labels["height"]
    stride = max(1, round(fps / float(load_config(args.machine).expected_fps)))
    sides = [Pipeline("current settings", args.machine, {}, fps, stride, args.clip),
             Pipeline("tuned settings", args.machine, tuned, fps, stride, args.clip)]
    truth = {f["idx"]: f for f in labels["frames"] if f["label"]}
    out_path = score.LABELS / args.clip.stem / "compare.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / labels["teacher"]["label_every"],
                             (1920, 540))
    cap = cv2.VideoCapture(str(args.clip))
    idx, totals = -1, [0, 0]
    while cap.grab():
        idx += 1
        if idx % stride:
            continue
        ok, img = cap.retrieve()
        if not ok:
            break
        frame = Frame(img, idx / fps, idx, args.clip.stem)
        covers = [s.step(frame) for s in sides]
        if idx in truth:
            inst = cv2.imread(str(score.LABELS / args.clip.stem / "masks" / f"{idx:06d}.png"), cv2.IMREAD_UNCHANGED)
            panels = []
            for k, (s, cov) in enumerate(zip(sides, covers)):
                panel, missed = draw(img, cov, inst, truth[idx], h, args.eval_min_height,
                                     f"{s.name}  t={idx / fps:.1f}s")
                totals[k] += missed
                panels.append(cv2.resize(panel, (960, 540), interpolation=cv2.INTER_AREA))
            writer.write(np.hstack(panels))
    cap.release()
    writer.release()
    print(f"{out_path}\nmissed person-frames: current {totals[0]}, tuned {totals[1]}\ntuned settings: {tuned}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
