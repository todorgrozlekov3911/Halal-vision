"""Score the live pipeline against the teacher labels (eval/teacher.py), everyone concealed.

    python -m eval.score --machine intel                       # all labelled clips, machine config
    python -m eval.score --machine intel --set detector.conf=0.15 --set timing.mask_hold_seconds=0.8

The pipeline is the real one: core/detect (model + Ultralytics tracker + confirm delay) and the
StyleRenderer (growth, smoothing, hold, size threshold), run on every `stride`-th frame, the way
the live loop sees the video at the machine's expected_fps. No identity yet: everyone the
pipeline covers is concealed, which is the conceal-all stage the policy starts from.

Model outputs are cached per clip (clips/labels/_cache/), at a low score floor, so changing
anything after the model (thresholds, tracker, timing, masks) replays in seconds; only a new
model or input size runs the model again.

Metrics, on the label frames (5 per second), for teacher people taller than --eval-min-height
of the frame (people the renderer is meant to cover) and not marked ignore:
  missed      person-frames with more than half of the person's visible pixels uncovered
  partial     person-frames with more than LEAK_PARTIAL of them uncovered
  leak_px     share of all those people's pixels left uncovered
  leak_frames label frames with at least one missed person (README §13.2 "uncensored frames")
  over        covered pixels further than OVER_MARGIN of the frame height from any person
              (teacher people of any size, ignore ones included), as a share of the frame
  id_switch   changes of the live track ID along one teacher track (per teacher track-minute)
  id_cover    share of person-frames whose person has a live track ID (rather than ID-less)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.config_table import ConfigError, available_machines, load_config
from core.contracts import Detections, Frame, MaskGeometry
from core.detect.detector import TrackingDetector, UltralyticsDetector, _imgsz
from ultralytics.trackers.track import TRACKER_MAP
from core.render.renderer import StyleRenderer

REPO = Path(__file__).resolve().parents[1]
CLIPS = REPO / "clips"
LABELS = CLIPS / "labels"
CACHE = LABELS / "_cache"

CONF_FLOOR = 0.05        # cached model outputs go down to this score; sweep thresholds above it
LEAK_MISSED = 0.5
LEAK_PARTIAL = 0.1
MIN_VISIBLE_PX = 400     # people hidden almost entirely behind others are not scored
OVER_MARGIN = 0.03       # of the frame height: cover this close to a person is growth, not a false cover
COVERED = 128            # cover alpha at or above this hides the pixel


# -- model output cache --------------------------------------------------------------

@dataclass
class ModelCache:
    """Model outputs (people rows + soft low-res masks) for every processed frame of one clip."""
    rows: dict[int, np.ndarray]
    masks: dict[int, np.ndarray]
    geometry: MaskGeometry
    infer_ms: float = 0.0         # median model time per frame where the cache was built
    device: str = ""

    @staticmethod
    def path(cfg: Any, clip: Path, stride: int) -> Path:
        imgsz = _imgsz(cfg.detector.imgsz)
        size = f"{imgsz[0]}x{imgsz[1]}" if isinstance(imgsz, tuple) else str(imgsz)
        tag = f"{Path(str(cfg.detector.model)).stem}_{size}{'_int8' if cfg.detector.get('int8', False) else ''}"
        return CACHE / tag / f"{clip.stem}_every{stride}.npz"

    @classmethod
    def load_or_build(cls, cfg: Any, clip: Path, stride: int) -> ModelCache:
        path = cls.path(cfg, clip, stride)
        if not path.is_file():
            cls._build(cfg, clip, stride, path)
        z = np.load(path)
        # Read each array once: every z[...] access decompresses the whole array again, so indexing
        # z["masks"] per frame kept hundreds of full copies alive (this froze a 30 GB laptop once).
        idx, offsets, all_rows, all_masks = z["idx"], z["offsets"], z["rows"], z["masks"]
        rows = {int(i): all_rows[a:b] for i, a, b in zip(idx, offsets[:-1], offsets[1:])}
        masks = {int(i): all_masks[a:b] for i, a, b in zip(idx, offsets[:-1], offsets[1:])}
        g = z["geometry"]
        return cls(rows, masks, MaskGeometry(scale=float(g[0]), pad_x=float(g[1]), pad_y=float(g[2]),
                                             frame_shape=(int(g[3]), int(g[4]))),
                   float(np.median(z["infer_ms"])) if "infer_ms" in z else 0.0,
                   str(z["device"]) if "device" in z else "")

    @staticmethod
    def _build(cfg: Any, clip: Path, stride: int, path: Path) -> None:
        print(f"  running {cfg.detector.model} on {clip.name} (every {stride}. frame), cached for next time", flush=True)
        cap = cv2.VideoCapture(str(clip))
        fps = cap.get(cv2.CAP_PROP_FPS)
        det = UltralyticsDetector(cfg, fps / stride)
        idx_list, rows, masks, geometry, infer_ms = [], [], [], None, []
        idx = -1
        while cap.grab():
            idx += 1
            if idx % stride:
                continue
            ok, img = cap.retrieve()
            if not ok:
                break
            people, m, geometry = det._infer(Frame(img, idx / fps, idx, clip.stem), CONF_FLOOR)
            idx_list.append(idx); rows.append(people); masks.append(m)
            infer_ms.append(det._infer_timing["model_infer"])
        cap.release()
        offsets = np.cumsum([0] + [len(r) for r in rows])
        mask_shape = masks[0].shape[1:]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, idx=np.array(idx_list), offsets=offsets,
            rows=np.concatenate(rows) if offsets[-1] else np.zeros((0, 6), np.float32),
            masks=np.concatenate(masks) if offsets[-1] else np.zeros((0, *mask_shape), np.uint8),
            geometry=np.array([geometry.scale, geometry.pad_x, geometry.pad_y, *geometry.frame_shape], np.float64),
            infer_ms=np.array(infer_ms[5:] or infer_ms, np.float32), device=np.array(det.runs_on))   # first frames warm up


class ReplayDetector(TrackingDetector):
    """The real tracking detector, fed from the cache instead of the model."""

    def __init__(self, cfg: Any, fps: float, cache: ModelCache) -> None:
        super().__init__(cfg, fps)
        self.cache = cache

    def _infer(self, frame: Frame, conf: float):
        rows, masks = self.cache.rows[frame.idx], self.cache.masks[frame.idx]
        keep = rows[:, 4] >= conf
        self._infer_timing = {}
        return rows[keep], masks[keep], self.cache.geometry


# -- teacher labels --------------------------------------------------------------------

def use_labels(folder: Path) -> None:
    """Read labels (and keep the model cache) under another folder."""
    global LABELS, CACHE
    LABELS, CACHE = folder, folder / "_cache"


def load_labels(clip_stem: str) -> dict:
    return json.loads((LABELS / clip_stem / "labels.json").read_text())


def labelled_clips() -> list[Path]:
    return sorted(CLIPS / d["clip"] for d in (json.loads(p.read_text()) for p in LABELS.glob("*/labels.json")))


# -- scoring ---------------------------------------------------------------------------

@dataclass
class Score:
    person_frames: int = 0
    missed: int = 0
    partial: int = 0
    gt_px: int = 0
    leak_px: int = 0
    frames: int = 0
    leak_frames: int = 0
    over_px: float = 0.0          # sum over frames of the over-covered share of the frame
    with_id: int = 0
    id_switches: int = 0
    track_minutes: float = 0.0
    seconds: float = 0.0          # wall time of the replay
    worst: list = field(default_factory=list)   # (leak, clip, frame idx, teacher track) of missed people

    def add(self, other: Score) -> None:
        for k in ("person_frames", "missed", "partial", "gt_px", "leak_px", "frames", "leak_frames",
                  "over_px", "with_id", "id_switches", "track_minutes", "seconds"):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        self.worst += other.worst

    def summary(self) -> dict[str, float]:
        pf = max(1, self.person_frames)
        return {
            "missed_%": 100 * self.missed / pf,
            "partial_%": 100 * self.partial / pf,
            "leak_px_%": 100 * self.leak_px / max(1, self.gt_px),
            "leak_frames_%": 100 * self.leak_frames / max(1, self.frames),
            "over_%": 100 * self.over_px / max(1, self.frames),
            "id_switch/min": self.id_switches / max(1e-9, self.track_minutes),
            "id_cover_%": 100 * self.with_id / pf,
            "person_frames": self.person_frames,
        }


@dataclass
class _Person:
    """One scored teacher person on one label frame: their visible pixels, cut to their box."""
    track_id: int | None
    box: np.ndarray
    y1: int
    x1: int
    pixels: np.ndarray        # bool, visible pixels inside (y1, x1) + shape


@dataclass
class _LabelFrame:
    idx: int
    people: list[_Person]
    near: np.ndarray          # packed bits: within OVER_MARGIN of any teacher person (any size, ignore too)


_TRUTH: dict[tuple, dict[int, _LabelFrame]] = {}


def load_truth(clip_stem: str, eval_min_height: float) -> dict[int, _LabelFrame]:
    """The teacher labels of one clip, prepared once per process (masks read, people cut out)."""
    key = (str(LABELS), clip_stem, eval_min_height)
    if key in _TRUTH:
        return _TRUTH[key]
    labels = load_labels(clip_stem)
    h = labels["height"]
    margin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * round(OVER_MARGIN * h) + 1,) * 2)
    out = {}
    for f in labels["frames"]:
        if not f["label"]:
            continue
        inst = cv2.imread(str(LABELS / clip_stem / "masks" / f"{f['idx']:06d}.png"), cv2.IMREAD_UNCHANGED)
        people = []
        for p in f["people"]:
            if p["ignore"] or "id" not in p or p["visible_area"] < MIN_VISIBLE_PX:
                continue
            if (p["box"][3] - p["box"][1]) / h < eval_min_height:
                continue
            ys, xs = np.nonzero(inst == p["id"])
            y1, x1 = int(ys.min()), int(xs.min())
            pixels = np.zeros((int(ys.max()) - y1 + 1, int(xs.max()) - x1 + 1), bool)
            pixels[ys - y1, xs - x1] = True
            people.append(_Person(p["track_id"], np.array(p["box"], np.float32), y1, x1, pixels))
        near = cv2.dilate((inst > 0).astype(np.uint8), margin) > 0
        out[f["idx"]] = _LabelFrame(f["idx"], people, np.packbits(near))
    _TRUTH[key] = out
    return out


def score_clip(cfg: Any, clip: Path, stride: int, eval_min_height: float,
               tracker: dict | None = None) -> tuple[Score, Score]:
    """Run the pipeline over the clip; scores of its first and second half."""
    labels = load_labels(clip.stem)
    fps, h, w = labels["fps"], labels["height"], labels["width"]
    cache = ModelCache.load_or_build(cfg, clip, stride)
    detector = ReplayDetector(cfg, fps / stride, cache)
    if tracker:                                   # tracker settings beyond the yaml file
        from ultralytics.utils import IterableSimpleNamespace
        args = {**vars(detector.tracker_args), **tracker}
        detector.tracker_args = IterableSimpleNamespace(**args)
        detector._tracker_cls = TRACKER_MAP[args["tracker_type"]]
        detector._predict_conf = min(detector.conf, float(args["track_low_thresh"]))
        detector.reset()
    renderer = StyleRenderer(cfg)
    renderer.set_default_style("solid")      # the style does not change the cover; solid is cheapest
    renderer.keep_cover = True
    truth = load_truth(clip.stem, eval_min_height)
    needs_image = detector.tracker_args.tracker_type == "botsort"   # GMC / ReID look at the image
    blank = np.zeros((h, w, 3), np.uint8)
    middle = labels["frame_count"] / 2

    halves = (Score(), Score())
    live_ids: dict[int, list[tuple[int, int | None]]] = {}     # teacher track -> (idx, live ID) per label frame
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(str(clip)) if needs_image else None
    pos = 0
    for idx in sorted(cache.rows):
        img = blank
        if cap is not None:
            while pos < idx:
                cap.grab(); pos += 1
            ok, img = cap.read(); pos += 1
        frame = Frame(img, idx / fps, idx, clip.stem)
        dets = detector.detect(frame)
        renderer.render(frame, dets, {})
        if idx in truth:
            _score_frame(halves[idx >= middle], truth[idx], dets, renderer.last_cover, clip.stem, (h, w), live_ids)
    if cap is not None:
        cap.release()
    halves[0].seconds = time.perf_counter() - t0
    label_dt = labels["teacher"]["label_every"] / fps
    for seq in live_ids.values():
        for half, s in enumerate(halves):
            part = [live for idx, live in seq if (idx >= middle) == half]
            s.track_minutes += len(part) * label_dt / 60
            known = [i for i in part if i is not None]
            s.id_switches += sum(a != b for a, b in zip(known, known[1:]))
    return halves


def _score_frame(s: Score, truth: _LabelFrame, dets: Detections, cover: np.ndarray | None, clip: str,
                 shape: tuple[int, int], live_ids: dict[int, list[tuple[int, int | None]]]) -> None:
    covered = cover >= COVERED if cover is not None else np.zeros(shape, bool)
    s.frames += 1
    frame_missed = False
    for p in truth.people:
        ph, pw = p.pixels.shape
        visible = int(p.pixels.sum())
        uncovered = int((p.pixels & ~covered[p.y1:p.y1 + ph, p.x1:p.x1 + pw]).sum())
        leak = uncovered / visible
        s.person_frames += 1
        s.gt_px += visible
        s.leak_px += uncovered
        s.partial += leak > LEAK_PARTIAL
        if leak > LEAK_MISSED:
            s.missed += 1
            frame_missed = True
            s.worst.append((round(leak, 2), clip, truth.idx, p.track_id))
        live = _live_id(p.box, dets)
        s.with_id += live is not None
        if p.track_id is not None:
            live_ids.setdefault(p.track_id, []).append((truth.idx, live))
    s.leak_frames += frame_missed
    near = np.unpackbits(truth.near, count=covered.size).reshape(covered.shape).astype(bool)
    s.over_px += float((covered & ~near).mean())


def _live_id(box: np.ndarray, dets: Detections) -> int | None:
    best, best_iou = None, 0.5
    for t in dets.tracks:
        b = t.bbox
        iw = max(0.0, min(box[2], b[2]) - max(box[0], b[0]))
        ih = max(0.0, min(box[3], b[3]) - max(box[1], b[1]))
        inter = iw * ih
        iou = inter / ((box[2] - box[0]) * (box[3] - box[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)
        if iou >= best_iou:
            best, best_iou = t.track_id, iou
    return best


def evaluate(cfg: Any, clips: list[Path], stride: int | None = None, eval_min_height: float = 0.12,
             tracker: dict | None = None) -> tuple[Score, dict[str, Score]]:
    """Total and per-part scores; parts are '<clip> 1st half' and '<clip> 2nd half'."""
    total, parts = Score(), {}
    for clip in clips:
        fps = load_labels(clip.stem)["fps"]
        st = stride or max(1, round(fps / float(cfg.expected_fps)))
        first, second = score_clip(cfg, clip, st, eval_min_height, tracker)
        parts[f"{clip.stem} 1st half"], parts[f"{clip.stem} 2nd half"] = first, second
        total.add(first)
        total.add(second)
    return total, parts


def format_table(rows: dict[str, Score]) -> str:
    keys = list(Score().summary())
    head = f"{'':34s}" + "".join(f"{k:>15s}" for k in keys)
    lines = [head]
    for name, s in rows.items():
        sm = s.summary()
        lines.append(f"{name[:34]:34s}" + "".join(f"{sm[k]:15.2f}" if isinstance(sm[k], float) else f"{sm[k]:15d}"
                                                  for k in keys))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eval.score", description=__doc__.split("\n")[0])
    p.add_argument("clips", nargs="*", type=Path, help="default: every labelled clip")
    p.add_argument("--machine", required=True, choices=available_machines())
    p.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY.PATH=VALUE")
    p.add_argument("--stride", type=int, help="process every n-th frame (default: video fps / expected_fps)")
    p.add_argument("--eval-min-height", type=float, default=0.12,
                   help="score teacher people at least this share of the frame height")
    p.add_argument("--worst", type=int, default=10, help="list this many of the worst missed person-frames")
    p.add_argument("--labels", type=Path, default=LABELS, help="teacher label folder (default clips/labels)")
    args = p.parse_args(argv)
    use_labels(args.labels)
    try:
        cfg = load_config(args.machine, args.sets)
    except ConfigError as exc:
        p.exit(2, f"config error: {exc}\n")
    clips = args.clips or labelled_clips()
    if not clips:
        p.exit(2, "no labelled clips: run python -m eval.teacher first\n")
    total, parts = evaluate(cfg, clips, args.stride, args.eval_min_height)
    print(format_table({**parts, "TOTAL": total}))
    for leak, clip, idx, tid in sorted(total.worst, key=lambda w: (w[0], w[1], w[2]), reverse=True)[:args.worst]:
        print(f"  missed: {clip} frame {idx} (t={idx / load_labels(clip)['fps']:.2f}s) teacher #{tid}, {leak:.0%} uncovered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
