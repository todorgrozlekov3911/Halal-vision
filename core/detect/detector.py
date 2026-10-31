"""Person detector + tracker (T08): Ultralytics YOLO26-seg behind core.contracts.Detector.

Per frame:
  1. model.predict()   people only, at the tracker's low threshold so its second
                       (low-score) matching stage has something to work with
  2. tracker.update()  Ultralytics' own BYTETracker / BOTSORT, driven here instead of
                       through model.track(), because model.track() throws away
                       people the tracker has not confirmed yet
  3. Detections        tracked people get their ID once the track is older than
                       timing.confirm_track_seconds; everyone else with
                       score >= detector.conf comes back with track_id=None and is
                       concealed downstream (rule 4, fail closed)

Masks stay at the model's resolution, letterbox padding included; MaskGeometry
says how they map onto the frame (resizing is the mask pipeline's job, T10).

No other module imports Ultralytics: everything leaves here as contracts types.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import IterableSimpleNamespace

from core.contracts import Detections, Detector, Frame, MaskGeometry, Track

REPO = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO / "models"
TRACKERS_DIR = REPO / "system-configurations" / "trackers"
PERSON = 0  # COCO class id


class UltralyticsDetector(Detector):
    def __init__(self, cfg: Any, fps: float) -> None:
        """cfg: the loaded ConfigTable. fps: frames per second this detector will actually
        process (video fps when every frame is processed, expected_fps when live), used to
        turn timing.track_buffer_seconds into the tracker's frame count."""
        det = cfg.detector
        self.conf = float(det.conf)
        self.imgsz = _imgsz(det.imgsz)
        self.device = str(det.device)
        self.half = bool(det.get("half", False))
        self.model = YOLO(str(_weights_for(det.model, self.device, self.imgsz)), task="segment")

        tracker_cfg = yaml.safe_load((TRACKERS_DIR / f"{det.tracker}.yaml").read_text())
        tracker_cfg["track_buffer"] = max(1, round(float(cfg.timing.track_buffer_seconds) * fps))
        if tracker_cfg.get("with_reid") and tracker_cfg.get("model", "auto") != "auto":
            tracker_cfg["model"] = str(MODELS_DIR / tracker_cfg["model"])  # ReID weights live in models/ too
            tracker_cfg["device"] = "cpu" if self.device.startswith("intel") else self.device  # plain torch model
        self.tracker_args = IterableSimpleNamespace(**tracker_cfg)
        self.track_buffer_frames = tracker_cfg["track_buffer"]
        self._tracker_cls = TRACKER_MAP[tracker_cfg["tracker_type"]]
        self._predict_conf = min(self.conf, float(tracker_cfg["track_low_thresh"]))
        self.confirm_seconds = float(cfg.timing.confirm_track_seconds)
        self._forget_after = float(cfg.timing.track_buffer_seconds) + 1.0
        self._letterbox_shape: dict[tuple[int, int], tuple[int, int]] = {}
        self.reset()

    # -- Detector interface ---------------------------------------------------

    def reset(self) -> None:
        self.tracker = self._tracker_cls(args=self.tracker_args)
        self._seen: dict[int, list[float]] = {}  # track_id -> [first ts, last ts]

    def detect(self, frame: Frame) -> Detections:
        result = self.model.predict(
            frame.image, classes=[PERSON], conf=self._predict_conf, imgsz=self.imgsz,
            device=self.device, quantize=16 if self.half else None, verbose=False,
        )[0]
        boxes = result.boxes.cpu().numpy()
        frame_shape = frame.image.shape[:2]

        # Tracker rows: x1, y1, x2, y2, track_id, score, cls, det_idx
        # (called on empty frames too: lost tracks must age so they expire on time)
        rows = self.tracker.update(boxes, frame.image)
        id_of = {int(r[7]): int(r[4]) for r in rows}
        id_of = self._confirmed(id_of, frame.ts)

        keep = [i for i in range(len(boxes)) if i in id_of or boxes.conf[i] >= self.conf]
        tracks = tuple(
            Track(
                track_id=id_of.get(i),
                bbox=boxes.xyxy[i].astype(np.float32).copy(),
                score=float(boxes.conf[i]),
                mask_idx=k,
            )
            for k, i in enumerate(keep)
        )

        if result.masks is not None and keep:
            masks = (result.masks.data[keep] > 0.5).to("cpu").numpy().astype(np.uint8) * 255
            mask_shape = masks.shape[1:]
        else:
            mask_shape = self._model_input_shape(frame)
            masks = np.zeros((0, *mask_shape), dtype=np.uint8)

        return Detections(
            frame_idx=frame.idx, ts=frame.ts, tracks=tracks, masks=np.ascontiguousarray(masks),
            geometry=_letterbox_geometry(frame_shape, mask_shape),
        )

    # -- internals ------------------------------------------------------------

    def _confirmed(self, id_of: dict[int, int], now: float) -> dict[int, int]:
        """Keep only IDs whose track has existed for confirm_track_seconds (time, not frames)."""
        for tid in id_of.values():
            self._seen.setdefault(tid, [now, now])[1] = now
        for tid in [t for t, (_, last) in self._seen.items() if now - last > self._forget_after]:
            del self._seen[tid]
        return {i: tid for i, tid in id_of.items() if now - self._seen[tid][0] >= self.confirm_seconds}

    def _model_input_shape(self, frame: Frame) -> tuple[int, int]:
        """(h, w) of the letterboxed model input, for frames where no masks came back."""
        key = frame.image.shape[:2]
        if key not in self._letterbox_shape:
            self._letterbox_shape[key] = self.model.predictor.pre_transform([frame.image])[0].shape[:2]
        return self._letterbox_shape[key]


def _letterbox_geometry(frame_shape: tuple[int, int], mask_shape: tuple[int, int]) -> MaskGeometry:
    """Same arithmetic as Ultralytics' LetterBox(center=True): scale to fit, pad evenly."""
    h, w = frame_shape
    mh, mw = mask_shape
    r = min(mh / h, mw / w)
    dw = (mw - round(w * r)) / 2
    dh = (mh - round(h * r)) / 2
    return MaskGeometry(scale=r, pad_x=float(round(dw - 0.1)), pad_y=float(round(dh - 0.1)), frame_shape=(h, w))


def _imgsz(value: Any) -> int | tuple[int, int]:
    """detector.imgsz: one number (square, e.g. 640) or [height, width] (e.g. [384, 640] for 16:9).
    Both must be multiples of 32. A shape matching the video avoids wasting model time on padding."""
    if isinstance(value, (list, tuple)):
        h, w = (int(v) for v in value)
        if h % 32 or w % 32:
            raise ValueError(f"detector.imgsz {value}: height and width must be multiples of 32")
        return h, w
    if int(value) % 32:
        raise ValueError(f"detector.imgsz {value}: must be a multiple of 32")
    return int(value)


def _weights_for(model: str, device: str, imgsz: int | tuple[int, int]) -> Path:
    """models/<name>.pt (downloaded by Ultralytics if missing). For intel:* devices the
    .pt is exported once to OpenVINO at this imgsz and that folder is used instead."""
    MODELS_DIR.mkdir(exist_ok=True)
    pt = MODELS_DIR / model
    if not device.startswith("intel"):
        return pt
    size = f"{imgsz[0]}x{imgsz[1]}" if isinstance(imgsz, tuple) else str(imgsz)
    ov_dir = MODELS_DIR / f"{pt.stem}_{size}_openvino_model"  # suffix is how Ultralytics spots the format
    if not (ov_dir / "metadata.yaml").is_file():
        exported = Path(YOLO(str(pt)).export(format="openvino", imgsz=list(imgsz) if isinstance(imgsz, tuple) else imgsz, dynamic=False))
        if ov_dir.exists():
            shutil.rmtree(ov_dir)
        exported.rename(ov_dir)
    return ov_dir
