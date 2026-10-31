"""Person detector + tracker (T08): YOLO26-seg (or RF-DETR-Seg, rf_detr.py) behind core.contracts.Detector.

    make_detector(cfg, fps)   picks the class from detector.model: rf-detr-seg-* or an Ultralytics .pt

Per frame:
  1. model             people only, at the tracker's low threshold so its second
                       (low-score) matching stage has something to work with
  2. tracker.update()  Ultralytics' own BYTETracker / BOTSORT, driven here instead of
                       through model.track(), because model.track() throws away
                       people the tracker has not confirmed yet
  3. Detections        tracked people get their ID once the track is older than
                       timing.confirm_track_seconds; everyone else with
                       score >= detector.conf comes back with track_id=None and is
                       concealed downstream (rule 4, fail closed)

Masks stay at the model's prototype resolution (1/4 of the model input, e.g.
96x160 for 384x640), soft (0..255 certainty), letterbox padding included; MaskGeometry says how they map
onto the frame. Ultralytics would enlarge every person's mask to the full model
input on the CPU; we skip that, because the mask pipeline (T10) merges the
concealed people at low resolution and enlarges once, straight to the frame.

TrackingDetector does steps 2 and 3 for every model; a model class only does step 1
(_infer). RF-DETR's masks map to the frame without padding, see rf_detr.py.

No module outside core/detect imports Ultralytics: everything leaves as contracts types.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results
from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import IterableSimpleNamespace, ops

from core.contracts import MASK_CUT, Detections, Detector, Frame, MaskGeometry, Track

REPO = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO / "models"
TRACKERS_DIR = REPO / "system-configurations" / "trackers"
PERSON = 0  # COCO class id


class _LowResMaskPredictor(SegmentationPredictor):
    """Ultralytics' segmentation predictor, minus the per-person mask enlargement, with soft masks.

    Based on SegmentationPredictor.construct_result (Ultralytics 8.4.194, non-retina
    path): masks stay at prototype resolution, and keep the model's per-pixel
    certainty (sigmoid, as uint8 0..255) instead of Ultralytics' yes/no cut. The cut
    happens after enlarging (mask pipeline), so outlines fall between mask pixels
    and move smoothly instead of in 12-pixel steps.
    """

    def construct_result(self, pred, img, orig_img, img_path, proto):
        masks = None
        if pred.shape[0]:
            masks = _soft_masks(proto, pred[:, 6:], pred[:, :4], img.shape[2:])
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            keep = masks.amax((-2, -1)) >= MASK_CUT  # drop detections whose mask came out empty
            if not keep.all():
                pred, masks = pred[keep], masks[keep]
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)


def _soft_masks(protos, coefficients, boxes, input_shape):
    """ops.process_mask(upsample=False) without the yes/no cut: sigmoid certainty as uint8 0..255,
    zero outside each person's box. 128 is where Ultralytics would cut."""
    c, mh, mw = protos.shape
    masks = (coefficients.float() @ protos.float().view(c, -1)).view(-1, mh, mw).sigmoid_()
    ratios = boxes.new_tensor([[mw / input_shape[1], mh / input_shape[0]] * 2])
    masks = ops.crop_mask(masks, boxes * ratios)   # boxes to prototype resolution
    return masks.mul_(255).round_().byte()


def make_detector(cfg: Any, fps: float, model: str | None = None) -> Detector:
    """The detector for detector.model (or `model`, to override it): rf-detr-seg-* -> RF-DETR-Seg,
    anything else -> an Ultralytics .pt such as yolo26n-seg.pt."""
    name = model or str(cfg.detector.model)
    if name.startswith("rf-detr"):
        from core.detect.rf_detr import RFDETRDetector   # imported only when used: pulls in transformers
        return RFDETRDetector(cfg, fps, model=name)
    return UltralyticsDetector(cfg, fps, model=name)


def openvino_device(device: str, auto_ok: bool = True) -> str:
    """detector.device intel:npu | intel:gpu | intel:cpu -> the OpenVINO device that really runs the
    model in this process. Same rule as Ultralytics: a device the process cannot see falls back to
    AUTO (CPU if that is all there is). The NPU is invisible without the render group: start with
    `sg render -c "..."`, or log out and in once after setup.sh added you to the group.
    auto_ok=False: fall back to the GPU (else CPU) instead of AUTO, for models AUTO breaks."""
    import openvino as ov
    available = ov.Core().available_devices
    wanted = device.partition(":")[2].upper()
    if wanted and any(d == wanted or d.startswith(f"{wanted}.") for d in available):
        return wanted
    if auto_ok:
        return "CPU" if available == ["CPU"] else "AUTO"
    return "GPU" if any(d.startswith("GPU") for d in available) else "CPU"


def runs_on(device: str, ov_device: str | None) -> str:
    """Human-readable device for status lines, saying when the configured one was unavailable."""
    if ov_device is None:
        return device
    wanted = device.partition(":")[2].upper()
    if wanted and wanted != ov_device:
        hint = " (no render group)" if wanted == "NPU" else ""
        return f"{ov_device}: {wanted} unavailable{hint}"
    return ov_device


class TrackingDetector(Detector):
    """What every model shares: Ultralytics' tracker, the confirm delay for new IDs, and the
    Detections it returns. A model only implements _infer(): people in, boxes + soft masks out."""

    AUTO_OK = True   # OpenVINO's AUTO device works for this model (see openvino_device)

    def __init__(self, cfg: Any, fps: float) -> None:
        """cfg: the loaded ConfigTable. fps: frames per second this detector will actually
        process (video fps when every frame is processed, expected_fps when live), used to
        turn timing.track_buffer_seconds into the tracker's frame count."""
        det = cfg.detector
        self.conf = float(det.conf)
        self.device = str(det.device)
        self.half = bool(det.get("half", False))
        self.ov_device = openvino_device(self.device, self.AUTO_OK) if self.device.startswith("intel") else None
        self.runs_on = runs_on(self.device, self.ov_device)   # for status lines: where the model really runs
        self._infer_timing: dict[str, float] = {}

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
        self.reset()

    # -- Detector interface ---------------------------------------------------

    def reset(self) -> None:
        self.tracker = self._tracker_cls(args=self.tracker_args)
        self._seen: dict[int, list[float]] = {}  # track_id -> [first ts, last ts]

    def detect(self, frame: Frame) -> Detections:
        t0 = time.perf_counter()
        people, masks, geometry = self._infer(frame, self._predict_conf)
        t1 = time.perf_counter()
        boxes = Boxes(people, frame.image.shape[:2])

        # Tracker rows: x1, y1, x2, y2, track_id, score, cls, det_idx
        # (called on empty frames too: lost tracks must age so they expire on time)
        rows = self.tracker.update(boxes, frame.image)
        id_of = {int(r[7]): int(r[4]) for r in rows}
        id_of = self._confirmed(id_of, frame.ts)
        t2 = time.perf_counter()

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
        dets = Detections(
            frame_idx=frame.idx, ts=frame.ts, tracks=tracks,
            masks=np.ascontiguousarray(masks[keep]), geometry=geometry,
        )
        # Milliseconds spent in each step of the last call, for profiling.
        self.last_timing = {
            **self._infer_timing,
            "tracker": 1000 * (t2 - t1), "to_contract": 1000 * (time.perf_counter() - t2),
        }
        return dets

    def _infer(self, frame: Frame, conf: float) -> tuple[np.ndarray, np.ndarray, MaskGeometry]:
        """Run the model: people scoring >= conf as (N, 6) float32 rows x1, y1, x2, y2 (frame
        pixels), score, class; their soft masks (N, h, w) uint8 0..255, every one non-empty;
        and the mask geometry. Sets self._infer_timing (ms per step) for profiling."""
        raise NotImplementedError

    # -- internals ------------------------------------------------------------

    def _confirmed(self, id_of: dict[int, int], now: float) -> dict[int, int]:
        """Keep only IDs whose track has existed for confirm_track_seconds (time, not frames)."""
        for tid in id_of.values():
            self._seen.setdefault(tid, [now, now])[1] = now
        for tid in [t for t, (_, last) in self._seen.items() if now - last > self._forget_after]:
            del self._seen[tid]
        return {i: tid for i, tid in id_of.items() if now - self._seen[tid][0] >= self.confirm_seconds}


class UltralyticsDetector(TrackingDetector):
    """YOLO26-seg (or any Ultralytics segmentation .pt) through Ultralytics."""

    def __init__(self, cfg: Any, fps: float, model: str | None = None) -> None:
        super().__init__(cfg, fps)
        det = cfg.detector
        self.name = model or str(det.model)
        self.imgsz = _imgsz(det.imgsz)
        self.int8 = bool(det.get("int8", False))
        if self.int8 and self.device.startswith("intel:npu"):
            # Measured: the INT8 export on the NPU found nobody in the far-away test section and
            # lowered close-up scores; missed people would be shown uncensored. FP is exact there.
            raise ValueError("detector.int8 with intel:npu loses detections; use int8: false on the NPU")
        self.model = YOLO(str(_weights_for(self.name, self.device, self.imgsz, self.int8)), task="segment")
        self._letterbox_shape: dict[tuple[int, int], tuple[int, int]] = {}

    def _infer(self, frame: Frame, conf: float) -> tuple[np.ndarray, np.ndarray, MaskGeometry]:
        t0 = time.perf_counter()
        result = self.model.predict(
            frame.image, classes=[PERSON], conf=conf, imgsz=self.imgsz,
            device=self.device, quantize=16 if self.half else None, verbose=False,
            predictor=_LowResMaskPredictor,
        )[0]
        t1 = time.perf_counter()
        people = result.boxes.data.cpu().numpy().astype(np.float32)
        in_h, in_w = self._model_input_shape(frame)
        if result.masks is not None:
            masks = result.masks.data.to("cpu").numpy().astype(np.uint8)   # soft, 0..255
        else:
            masks = np.zeros((0, in_h // 4, in_w // 4), dtype=np.uint8)  # prototype grid = input / 4
        geometry = _mask_geometry(frame.image.shape[:2], (in_h, in_w), masks.shape[1:])
        sp = result.speed
        self._infer_timing = {
            "model_pre": sp["preprocess"], "model_infer": sp["inference"], "model_post": sp["postprocess"],
            "ultralytics_other": 1000 * (t1 - t0) - sp["preprocess"] - sp["inference"] - sp["postprocess"],
            "to_arrays": 1000 * (time.perf_counter() - t1),
        }
        return people, masks, geometry

    def _model_input_shape(self, frame: Frame) -> tuple[int, int]:
        """(h, w) of the letterboxed model input for this frame size (computed once per size)."""
        key = frame.image.shape[:2]
        if key not in self._letterbox_shape:
            self._letterbox_shape[key] = self.model.predictor.pre_transform([frame.image])[0].shape[:2]
        return self._letterbox_shape[key]


def _mask_geometry(frame_shape: tuple[int, int], input_shape: tuple[int, int],
                   mask_shape: tuple[int, int]) -> MaskGeometry:
    """Frame -> mask pixels. The letterbox step uses the same arithmetic as Ultralytics'
    LetterBox(center=True) (scale to fit, pad evenly); masks are a further 1/4 of that."""
    h, w = frame_shape
    in_h, in_w = input_shape
    r = min(in_h / h, in_w / w)
    pad_x = round((in_w - round(w * r)) / 2 - 0.1)
    pad_y = round((in_h - round(h * r)) / 2 - 0.1)
    k = mask_shape[1] / in_w  # mask grid per input pixel (1/4)
    return MaskGeometry(scale=r * k, pad_x=pad_x * k, pad_y=pad_y * k, frame_shape=(h, w))


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


def _weights_for(model: str, device: str, imgsz: int | tuple[int, int], int8: bool = False) -> Path:
    """models/<name>.pt (downloaded by Ultralytics if missing). For intel:* devices the
    .pt is exported once to OpenVINO at this imgsz and that folder is used instead.
    int8: quantize to 8-bit during export (needs nncf), calibrated on Ultralytics'
    coco128-seg sample images (downloaded on first use)."""
    MODELS_DIR.mkdir(exist_ok=True)
    pt = MODELS_DIR / model
    if not device.startswith("intel"):
        return pt
    size = f"{imgsz[0]}x{imgsz[1]}" if isinstance(imgsz, tuple) else str(imgsz)
    precision = "_int8" if int8 else ""
    ov_dir = MODELS_DIR / f"{pt.stem}_{size}{precision}_openvino_model"  # suffix is how Ultralytics spots the format
    if not (ov_dir / "metadata.yaml").is_file():
        kwargs = {"quantize": 8, "data": "coco128-seg.yaml"} if int8 else {}
        exported = Path(YOLO(str(pt)).export(format="openvino", imgsz=list(imgsz) if isinstance(imgsz, tuple) else imgsz,
                                             dynamic=False, **kwargs))
        if ov_dir.exists():
            shutil.rmtree(ov_dir)
        exported.rename(ov_dir)
    return ov_dir
