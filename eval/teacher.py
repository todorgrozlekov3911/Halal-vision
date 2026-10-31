"""Teacher labels: a slow, accurate offline model labels the test clips, as the reference the fast
live setup (YOLO26n) is scored and tuned against.

    python -m eval.teacher                       # every clip in clips/ except the long walk and the 720p copy
    python -m eval.teacher clips/berlin_people_crossing.mp4 --force

Two stages, run side by side (detector on the iGPU, SAM on the CPU):
  1. track   YOLO26x-seg at 736x1280 (OpenVINO, iGPU) + BoT-SORT with appearance ReID, on every
             TRACK_EVERY-th frame (15 fps for 60 fps video): boxes, scores and person IDs
  2. masks   SAM 2.1-B, prompted with those boxes, on every LABEL_EVERY-th frame (5 fps): sharp,
             full-resolution masks. Nearer people (box bottom lower in the frame) are painted
             over farther ones, so each mask is the visible part of that person.

Per person:
  ignore   teacher score in [DROP_BELOW, LABEL_SCORE): the teacher is unsure, so the evaluation
           should neither reward nor punish covering it
  check    SAM's mask and YOLO26x's own mask disagree (IoU < CHECK_IOU): look at it in the review

Output, clips/labels/<clip>/ (gitignored with clips/):
  labels.json        clip info, teacher settings, and per tracked frame the people:
                     {id, track_id, box [x1,y1,x2,y2], score, ignore, check?, mask_iou?}
                     ("id" is the value of that person in the frame's mask PNG; label frames only)
  masks/<idx>.png    label frames: uint8 instance map at full resolution, 0 = background
  review.mp4         label frames with the masks drawn in, 5 fps = real time, for looking at

This tool may import Ultralytics directly: it is offline tooling, not part of core/.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
CLIPS = REPO / "clips"
OUT = CLIPS / "labels"
MODELS = REPO / "models"
TRACKERS = REPO / "system_configurations" / "trackers"

DETECTOR = MODELS / "yolo26x-seg_736x1280_half_openvino_model"   # exported from yolo26x-seg.pt, FP16
DETECTOR_IMGSZ = (736, 1280)
SAM_WEIGHTS = MODELS / "sam2.1_b.pt"
SKIP_BY_DEFAULT = ("berlin_walk_raw_to_frankfurter_tor_1080p60", "berlin_people_crossing_720p30")

TRACK_EVERY = 4          # 60 fps video -> 15 fps tracking
LABEL_EVERY = 12         # -> 5 fps masks (a multiple of TRACK_EVERY)
DROP_BELOW = 0.15        # teacher detections below this are not people
LABEL_SCORE = 0.40       # below this (and >= DROP_BELOW) a person is "ignore"
CHECK_IOU = 0.6          # SAM vs YOLO26x mask agreement below this -> "check"
BOX_SLACK = 0.05         # SAM masks are clipped to the box grown by this share of its height
TRACK_BUFFER_SECONDS = 3.0
PERSON = 0


# -- stage 1: detection + tracking ----------------------------------------------------

def track_clip(clip: Path, max_seconds: float = 0.0) -> dict:
    from ultralytics import YOLO
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import IterableSimpleNamespace

    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if max_seconds:
        n = min(n, int(max_seconds * fps))
    track_fps = fps / TRACK_EVERY

    tcfg = yaml.safe_load((TRACKERS / "botsort_reid.yaml").read_text())
    tcfg.update(track_buffer=round(TRACK_BUFFER_SECONDS * track_fps), model=str(MODELS / tcfg["model"]),
                device="cpu", track_low_thresh=DROP_BELOW)
    tracker = TRACKER_MAP[tcfg["tracker_type"]](args=IterableSimpleNamespace(**tcfg))
    model = YOLO(str(DETECTOR), task="segment")

    frames = []
    t0 = time.time()
    for idx in range(n):
        if not cap.grab():
            break
        if idx % TRACK_EVERY:
            continue
        ok, img = cap.retrieve()
        if not ok:
            break
        r = model.predict(img, imgsz=DETECTOR_IMGSZ, classes=[PERSON], conf=DROP_BELOW,
                          device="intel:gpu", verbose=False)[0]
        data = r.boxes.data.cpu().numpy()
        rows = tracker.update(Boxes(data, img.shape[:2]), img)
        id_of = {int(row[7]): int(row[4]) for row in rows}
        people = [{"track_id": id_of.get(i), "box": [round(float(v), 1) for v in data[i, :4]],
                   "score": round(float(data[i, 4]), 3), "ignore": bool(data[i, 4] < LABEL_SCORE)}
                  for i in range(len(data))]
        frames.append({"idx": idx, "ts": round(idx / fps, 4), "label": idx % LABEL_EVERY == 0, "people": people})
        if len(frames) % 60 == 0:
            print(f"  [track] {clip.stem}: frame {idx}/{n}  {time.time() - t0:.0f}s", flush=True)
    cap.release()
    return {
        "clip": clip.name, "fps": fps, "width": w, "height": h, "frame_count": n,
        "teacher": {
            "detector": DETECTOR.name, "imgsz": list(DETECTOR_IMGSZ), "tracker": "botsort_reid",
            "track_buffer_seconds": TRACK_BUFFER_SECONDS, "segmenter": SAM_WEIGHTS.name,
            "track_every": TRACK_EVERY, "label_every": LABEL_EVERY, "drop_below": DROP_BELOW,
            "label_score": LABEL_SCORE, "check_iou": CHECK_IOU, "box_slack": BOX_SLACK,
        },
        "frames": frames,
    }


# -- stage 2: masks -----------------------------------------------------------------

def mask_clip(clip: Path, labels: dict, out_dir: Path) -> None:
    from ultralytics import SAM, YOLO

    sam = SAM(str(SAM_WEIGHTS))
    yolo = YOLO(str(DETECTOR), task="segment")   # its own masks, only to flag disagreements
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    h, w = labels["height"], labels["width"]

    cap = cv2.VideoCapture(str(clip))
    review = cv2.VideoWriter(str(out_dir / "review.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             labels["fps"] / LABEL_EVERY, (1280, 720))
    todo = {f["idx"]: f for f in labels["frames"] if f["label"]}
    t0, done = time.time(), 0
    for idx in range(labels["frame_count"]):
        if not cap.grab():
            break
        if idx not in todo:
            continue
        ok, img = cap.retrieve()
        frame = todo[idx]
        people = frame["people"]
        inst = np.zeros((h, w), np.uint8)
        if people:
            boxes = np.array([p["box"] for p in people], np.float32)
            sam_masks = sam.predict(img, bboxes=boxes, verbose=False)[0].masks.data.cpu().numpy().astype(bool)
            yolo_r = yolo.predict(img, imgsz=DETECTOR_IMGSZ, classes=[PERSON], conf=DROP_BELOW,
                                  device="intel:gpu", retina_masks=True, verbose=False)[0]
            yolo_boxes = yolo_r.boxes.xyxy.cpu().numpy()
            yolo_masks = yolo_r.masks.data.cpu().numpy().astype(bool) if yolo_r.masks is not None else None
            for p, m, box in zip(people, sam_masks, boxes):
                m &= _box_region(box, (h, w))
                p["area"] = int(m.sum())
                j = _best_box(box, yolo_boxes)
                if j is not None and yolo_masks is not None:
                    p["mask_iou"] = round(_iou(m, yolo_masks[j]), 3)
                    p["check"] = bool(p["mask_iou"] < CHECK_IOU)
            # far to near: nearer people (lower box bottom) paint over the ones behind them
            order = sorted(range(len(people)), key=lambda i: people[i]["box"][3])
            for k, i in enumerate(order, start=1):
                people[i]["id"] = k
                inst[sam_masks[i]] = k
            for p in people:   # visible area after occlusion
                p["visible_area"] = int((inst == p["id"]).sum())
        cv2.imwrite(str(mask_dir / f"{idx:06d}.png"), inst)
        review.write(draw_review(img, inst, frame, labels["fps"]))
        done += 1
        if done % 20 == 0:
            print(f"  [masks] {clip.stem}: {done}/{len(todo)}  {(time.time() - t0) / done:.1f}s/frame", flush=True)
    cap.release()
    review.release()


def _box_region(box: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    s = BOX_SLACK * (y2 - y1)
    region = np.zeros(shape, bool)
    region[max(0, int(y1 - s)):int(y2 + s) + 1, max(0, int(x1 - s)):int(x2 + s) + 1] = True
    return region


def _best_box(box: np.ndarray, others: np.ndarray) -> int | None:
    if not len(others):
        return None
    ix1 = np.maximum(box[0], others[:, 0]); iy1 = np.maximum(box[1], others[:, 1])
    ix2 = np.minimum(box[2], others[:, 2]); iy2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    iou = inter / (area(box) + area(others) - inter + 1e-9)
    j = int(iou.argmax())
    return j if iou[j] > 0.5 else None


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


# -- review video ---------------------------------------------------------------------

def _color(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return (255, 255, 255)
    hue = (track_id * 47) % 180
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def draw_review(img: np.ndarray, inst: np.ndarray, frame: dict, fps: float) -> np.ndarray:
    """Labelled people in their track colour (white: no ID yet), ignored ones hatched grey,
    disagreements ("check") with a red outline and a '?'."""
    out = img.copy()
    hatch = (np.add.outer(np.arange(img.shape[0]), np.arange(img.shape[1])) // 6) % 2 == 0
    for p in frame["people"]:
        m = inst == p.get("id", -1)
        if not m.any():
            continue
        if p["ignore"]:
            out[m & hatch] = (out[m & hatch] * 0.4 + 128 * 0.6).astype(np.uint8)
            color = (160, 160, 160)
        else:
            color = _color(p["track_id"])
            out[m] = (out[m] * 0.45 + np.array(color) * 0.55).astype(np.uint8)
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 0, 255) if p.get("check") else color, 3 if p.get("check") else 2)
        x1, y1 = int(p["box"][0]), int(p["box"][1])
        tag = f"{'#' + str(p['track_id']) if p['track_id'] is not None else 'new'} {p['score']:.2f}"
        tag += " ?" if p.get("check") else ""
        size = float(np.clip((p["box"][3] - p["box"][1]) / 300, 0.5, 1.2))   # small people, small tags
        cv2.putText(out, tag, (x1, max(30, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out, tag, (x1, max(30, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, size, color, 2, cv2.LINE_AA)
    info = f"frame {frame['idx']}  t={frame['idx'] / fps:.2f}s  people {sum(not p['ignore'] for p in frame['people'])}"
    cv2.putText(out, info, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(out, info, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.resize(out, (1280, 720), interpolation=cv2.INTER_AREA)


# -- driver ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eval.teacher", description=__doc__.split("\n")[0])
    p.add_argument("clips", nargs="*", type=Path)
    p.add_argument("--force", action="store_true", help="relabel clips that already have labels")
    p.add_argument("--max-seconds", type=float, default=0.0, help="only the first this many seconds (testing)")
    p.add_argument("--out", type=Path, default=OUT, help="output root (default clips/labels)")
    args = p.parse_args(argv)
    clips = args.clips or sorted(c for c in CLIPS.glob("*.mp4") if c.stem not in SKIP_BY_DEFAULT)
    clips = [c for c in clips if args.force or not (args.out / c.stem / "labels.json").is_file()]
    if not clips:
        print("nothing to do (use --force to relabel)")
        return 0

    # Stage 1 (iGPU) runs one clip ahead of stage 2 (CPU).
    tracked: queue.Queue = queue.Queue(maxsize=1)

    def tracking() -> None:
        for clip in clips:
            print(f"[track] {clip.name}", flush=True)
            tracked.put((clip, track_clip(clip, args.max_seconds)))
        tracked.put(None)

    threading.Thread(target=tracking, daemon=True).start()
    while (item := tracked.get()) is not None:
        clip, labels = item
        out_dir = args.out / clip.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[masks] {clip.name}: {sum(f['label'] for f in labels['frames'])} label frames", flush=True)
        mask_clip(clip, labels, out_dir)
        (out_dir / "labels.json").write_text(json.dumps(labels, indent=1))
        print(f"[done] {clip.name} -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
