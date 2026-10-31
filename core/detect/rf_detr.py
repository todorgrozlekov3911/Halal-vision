"""RF-DETR-Seg (Roboflow, Apache 2.0) behind core.contracts.Detector, as an alternative to YOLO26-seg.

    detector.model: rf-detr-seg-nano | -small | -medium | -large | -xlarge | -2xlarge

Same tracker, confirm delay and Detections as YOLO26 (TrackingDetector, detector.py);
only the model differs. Each size has a fixed square input (nano 312, small 384, ...);
detector.imgsz does not apply.

Where it runs:
    intel:npu / intel:gpu / intel:cpu   OpenVINO (a device this process cannot see falls back to
                                        the GPU, see AUTO_OK and openvino_device). The PyTorch model is converted once to
                                        models/<name>_<res>_openvino_model/ (fixed 1x3xRxR input,
                                        which the NPU needs). Measured on the Core Ultra 7 155U,
                                        nano: NPU 60-100 ms (varied between runs), GPU ~140 ms,
                                        CPU ~425 ms per frame; YOLO26n on the NPU is ~14 ms
    cuda / cuda:N / cpu                 PyTorch (traced), FP16 when detector.half

The frame is stretched to the square input, like RF-DETR's own predict(), not letterboxed:
no input is wasted on padding. Its masks (a quarter of the input, e.g. 78x78 for nano)
come out stretched the same way, so each is widened back to the frame's shape
(e.g. 78x139 for 16:9) and the mask geometry is a plain scale, no padding.

Masks are soft like YOLO26's: per-pixel sigmoid certainty as uint8 0..255, zero outside the
person's box, and a person whose mask is empty after that is dropped (as on the YOLO side).
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.contracts import MASK_CUT, Frame, MaskGeometry
from core.detect.detector import MODELS_DIR, TrackingDetector

# name -> rfdetr class; weights files and input sizes come from the class's config
SIZES = {
    "rf-detr-seg-nano": "RFDETRSegNano",
    "rf-detr-seg-small": "RFDETRSegSmall",
    "rf-detr-seg-medium": "RFDETRSegMedium",
    "rf-detr-seg-large": "RFDETRSegLarge",
    "rf-detr-seg-xlarge": "RFDETRSegXLarge",
    "rf-detr-seg-2xlarge": "RFDETRSeg2XLarge",
}
PERSON = 1                   # RF-DETR keeps COCO's category ids: 1 = person (logit 0 is unused)
OUT_CLASS = 0.0              # class column handed to the tracker (it only sees people)
MEANS = np.array([0.485, 0.456, 0.406], np.float32)   # ImageNet, as RFDETR.means / .stds
STDS = np.array([0.229, 0.224, 0.225], np.float32)


class RFDETRDetector(TrackingDetector):
    # AUTO runs the first frames on the CPU while the GPU compiles, then switches; with this model the
    # switch breaks the mask output ("Cannot find tensor for port ... f32[1,100,78,78]", OpenVINO
    # 2026.4, after 3-4 frames). So without the NPU it goes straight to the GPU.
    AUTO_OK = False

    def __init__(self, cfg: Any, fps: float, model: str | None = None) -> None:
        super().__init__(cfg, fps)
        self.name = model or str(cfg.detector.model)
        if self.name not in SIZES:
            raise ValueError(f"unknown RF-DETR model {self.name!r}; choose one of {', '.join(SIZES)}")
        import rfdetr
        import rfdetr.config
        self._cls = getattr(rfdetr, SIZES[self.name])
        model_cfg = getattr(rfdetr.config, f"{SIZES[self.name]}Config")()
        self.resolution = int(model_cfg.resolution)
        self._weights = MODELS_DIR / str(model_cfg.pretrain_weights)
        if self.device.startswith("intel"):
            self._run = self._openvino(self.ov_device)   # no NPU: the GPU (not AUTO, see AUTO_OK)
        else:
            self._run = self._torch(self.device)

    def _infer(self, frame: Frame, conf: float) -> tuple[np.ndarray, np.ndarray, MaskGeometry]:
        t0 = time.perf_counter()
        x = preprocess(frame.image, self.resolution)
        t1 = time.perf_counter()
        boxes, logits, mask_logits = self._run(x)
        t2 = time.perf_counter()
        out = people_from_outputs(boxes[0], logits[0], mask_logits[0], frame.image.shape[:2], conf)
        self._infer_timing = {"model_pre": 1000 * (t1 - t0), "model_infer": 1000 * (t2 - t1),
                              "model_post": 1000 * (time.perf_counter() - t2)}
        return out

    # -- runtimes: x (1, 3, R, R) float32 -> (boxes, logits, mask logits) as numpy ------------

    def _model(self, device: str):
        """The rfdetr wrapper with this size's COCO weights (downloaded into models/ once)."""
        if not self._weights.is_file():
            from rfdetr.assets.model_weights import ModelWeights, _download_file
            asset = ModelWeights.from_filename(self._weights.name)
            if asset is None:
                raise FileNotFoundError(f"no download known for {self._weights.name}")
            MODELS_DIR.mkdir(exist_ok=True)
            _download_file(url=asset.url, filename=str(self._weights), expected_md5=asset.md5_hash)
        return self._cls(pretrain_weights=str(self._weights), device=device)

    def _torch(self, device: str):
        import torch
        dtype = torch.float16 if self.half and device.startswith("cuda") else torch.float32
        m = self._model(device.split(":")[0])          # rfdetr takes "cuda" / "cpu", not "cuda:0"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")              # tracing warnings, expected
            m.optimize_for_inference(compile=True, dtype=dtype)
        net, dev = m.model.inference_model, m.model.device

        def run(x: np.ndarray) -> list[np.ndarray]:
            with torch.no_grad():
                out = net(torch.from_numpy(x).to(dev, dtype))
            return [o.float().cpu().numpy() for o in out]
        return run

    def _openvino(self, ov_device: str):
        import openvino as ov
        xml = MODELS_DIR / f"{self.name}_{self.resolution}_openvino_model" / "model.xml"
        if not xml.is_file():
            self._export_openvino(xml)
        compiled = ov.Core().compile_model(str(xml), ov_device, {"PERFORMANCE_HINT": "LATENCY"})
        request = compiled.create_infer_request()
        outputs = compiled.outputs                       # boxes, logits, masks: the PyTorch order

        def run(x: np.ndarray) -> list[np.ndarray]:
            result = request.infer({0: x})
            return [result[o] for o in outputs]
        return run

    def _export_openvino(self, xml: Path) -> None:
        import openvino as ov
        import torch
        m = self._model("cpu")
        m.optimize_for_inference(compile=False)          # the export-mode forward: plain tensors out
        r = self.resolution
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            ov_model = ov.convert_model(m.model.inference_model, example_input=torch.zeros(1, 3, r, r))
        ov_model.reshape({ov_model.inputs[0]: ov.PartialShape([1, 3, r, r])})   # static: the NPU needs it
        xml.parent.mkdir(parents=True, exist_ok=True)
        ov.save_model(ov_model, str(xml), compress_to_fp16=False)


# -- pre / post processing (pure functions) -------------------------------------------

def preprocess(image_bgr: np.ndarray, resolution: int) -> np.ndarray:
    """BGR frame -> (1, 3, R, R) float32, stretched to the square input, ImageNet-normalized.

    Shrinking is antialiased (INTER_AREA) like RF-DETR's own resize, but from 2R, reached
    with a cheap linear resize first: 1.5 ms instead of 12.5 ms for a 1080p frame."""
    if image_bgr.shape[0] > 2 * resolution and image_bgr.shape[1] > 2 * resolution:
        image_bgr = cv2.resize(image_bgr, (2 * resolution, 2 * resolution), interpolation=cv2.INTER_LINEAR)
    small = cv2.resize(image_bgr, (resolution, resolution), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) * (1 / 255)
    return np.ascontiguousarray(((rgb - MEANS) / STDS).transpose(2, 0, 1)[None])


def people_from_outputs(boxes: np.ndarray, logits: np.ndarray, mask_logits: np.ndarray,
                        frame_shape: tuple[int, int], conf: float,
                        ) -> tuple[np.ndarray, np.ndarray, MaskGeometry]:
    """One image's raw outputs -> TrackingDetector._infer's (people, masks, geometry).

    boxes (Q, 4) cx, cy, w, h relative to the frame; logits (Q, classes); mask_logits (Q, m, m),
    the whole frame stretched to a square. People are sorted by score, highest first."""
    h, w = frame_shape
    m = mask_logits.shape[1]
    scale = m / h                                        # mask rows per frame row; columns follow
    geometry = MaskGeometry(scale=scale, pad_x=0.0, pad_y=0.0, frame_shape=(h, w))
    fx = w * scale / mask_logits.shape[2]                # widen: square grid -> the frame's shape

    score = _sigmoid(logits[:, PERSON])
    idx = np.nonzero(score >= conf)[0]
    idx = idx[np.argsort(-score[idx], kind="stable")]
    cx, cy, bw, bh = boxes[idx].T.astype(np.float32)
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1) * np.float32([w, h, w, h])
    xyxy = np.clip(xyxy, 0, [w, h, w, h]).astype(np.float32)

    rows, masks = [], []
    for i, box in zip(idx, xyxy):
        wide = cv2.resize(mask_logits[i].astype(np.float32), None, fx=fx, fy=1.0, interpolation=cv2.INTER_LINEAR)
        soft = np.round(_sigmoid(wide) * 255).astype(np.uint8)
        x1, y1, x2, y2 = box * scale                     # box in mask pixels: zero outside it (like YOLO's crop)
        r, c = np.arange(soft.shape[0])[:, None], np.arange(soft.shape[1])[None, :]
        soft[~((r >= y1) & (r < y2) & (c >= x1) & (c < x2))] = 0
        if soft.max() < MASK_CUT:                        # empty mask: dropped, as on the YOLO side
            continue
        rows.append([*box, score[i], OUT_CLASS])
        masks.append(soft)

    mask_shape = (m, round(mask_logits.shape[2] * fx))
    people = np.array(rows, np.float32).reshape(-1, 6)
    stack = np.stack(masks) if masks else np.zeros((0, *mask_shape), np.uint8)
    return people, stack, geometry


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))
