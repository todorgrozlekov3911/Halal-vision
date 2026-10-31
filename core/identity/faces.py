"""Face analysis (T16): InsightFace buffalo_l / buffalo_s -> FaceObservation.

    faces = FaceAnalyzer(cfg)                                  # identity.face_model picks the pack
    found = faces.analyze(frame.image, frame.ts, frame.idx, region=(x1, y1, x2, y2))

Per face: box, 5 landmarks, an approximate head pose, detector score, a quality score, the
L2-normalised fingerprint (embedding), p_female and age. Everything in full-frame pixels, also when
the model ran on a region (a person's box). This module only describes faces: which person a face
belongs to is T17, what the evidence means is T18-T21.

What InsightFace does not give directly, and how it is made here:
  p_female  the gender model's two raw scores (female, male), as a probability; InsightFace itself
            only keeps the larger one (0/1)
  age       the model's raw output x 100, unrounded
  head_pose (pitch, yaw, roll) in degrees from the 5 landmarks, approximate (no 3D model, which
            would cost another network): roll from the eye line, yaw from how far the nose sits off
            the eye midpoint, pitch from how far down the nose sits between eyes and mouth
  quality   0..1 = size x sharpness x pose:
              size       0 at identity.min_face_px tall, 1 from GOOD_FACE_PX
              sharpness  Laplacian variance of the face at 64x64, 1 from SHARP_REF
              pose       1 facing the camera, 0 at MAX_YAW sideways or MAX_PITCH up/down
            starting values; T18/T21 tune what quality is needed for evidence

Faces smaller than identity.min_face_px are skipped before the fingerprint and gender models run
(they would give no evidence anyway, and those models cost time per face).

Models: downloaded once into models/insightface/models/<pack>/ (gitignored). Runs on the CPU
through ONNX Runtime with identity.face_threads threads per model, so it leaves cores for the
detector and the main loop (README §9.1).

License: InsightFace's pretrained models are for non-commercial use only (README §9.2).
"""
from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.contracts import FaceObservation

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "models" / "insightface"          # InsightFace adds models/<pack>/ below this
MODULES = ["detection", "genderage", "recognition"]

GOOD_FACE_PX = 80        # face height giving full size quality (README §9.4: enrollment faces >= 80 px)
SHARP_REF = 150.0        # Laplacian variance at 64x64 giving full sharpness quality
MAX_YAW = 60.0           # degrees: pose quality reaches 0 here
MAX_PITCH = 45.0
FEMALE = 0               # index of the female score in the gender model's output (InsightFace: 1 = male)


class FaceAnalyzer:
    def __init__(self, cfg: Any) -> None:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        ident = cfg.identity
        self.model_name = str(ident.face_model)
        self.min_face_px = float(ident.min_face_px)
        det_size = int(ident.get("face_det_size", 320))
        threads = int(ident.get("face_threads", 2))
        with contextlib.redirect_stdout(io.StringIO()):             # InsightFace prints per model
            self.app = FaceAnalysis(name=self.model_name, root=str(ROOT), allowed_modules=MODULES,
                                    providers=["CPUExecutionProvider"])
            self.app.prepare(ctx_id=-1, det_thresh=float(ident.get("face_det_thresh", 0.5)),
                             det_size=(det_size, det_size))
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        for model in self.app.models.values():                     # same models, limited threads
            model.session = ort.InferenceSession(model.model_file, sess_options=options,
                                                 providers=["CPUExecutionProvider"])
        missing = [m for m in MODULES if m not in self.app.models]
        if missing:
            raise RuntimeError(f"InsightFace pack {self.model_name} lacks {missing}")
        self.detector = self.app.models["detection"]
        self.genderage = self.app.models["genderage"]
        self.recognition = self.app.models["recognition"]

    def analyze(self, image: np.ndarray, ts: float, frame_idx: int,
                region: tuple[float, float, float, float] | None = None) -> list[FaceObservation]:
        """Faces in the image, or only inside region (x1, y1, x2, y2 frame pixels), largest first."""
        ox, oy = 0, 0
        if region is not None:
            h, w = image.shape[:2]
            x1, y1 = max(0, int(region[0])), max(0, int(region[1]))
            x2, y2 = min(w, math.ceil(region[2])), min(h, math.ceil(region[3]))
            if x2 - x1 < self.min_face_px or y2 - y1 < self.min_face_px:
                return []
            image, ox, oy = image[y1:y2, x1:x2], x1, y1
        boxes, kpss = self.detector.detect(image, max_num=0, metric="default")
        found = []
        for box, kps in zip(boxes, kpss if kpss is not None else [None] * len(boxes)):
            if kps is None or box[3] - box[1] < self.min_face_px:
                continue
            found.append(self._describe(image, box, kps, (ox, oy), ts, frame_idx))
        return sorted(found, key=lambda f: -(f.bbox[3] - f.bbox[1]))

    # -- one face ---------------------------------------------------------------------

    def _describe(self, image: np.ndarray, box: np.ndarray, kps: np.ndarray, offset: tuple[int, int],
                  ts: float, frame_idx: int) -> FaceObservation:
        from insightface.app.common import Face

        face = Face(bbox=box[:4], kps=kps, det_score=float(box[4]))
        self.recognition.get(image, face)
        embedding = face.embedding / max(1e-9, float(np.linalg.norm(face.embedding)))
        p_female, age = self._gender_age(image, box[:4])
        pose = head_pose(kps)
        quality = (size_quality(float(box[3] - box[1]), self.min_face_px)
                   * sharpness_quality(image, box[:4]) * pose_quality(pose))
        shift = np.array(offset, np.float32)
        return FaceObservation(
            bbox=(box[:4].astype(np.float32) + np.tile(shift, 2)),
            landmarks=(kps.astype(np.float32) + shift),
            head_pose=pose, det_score=float(box[4]), quality=quality,
            embedding=embedding.astype(np.float32), embedding_model=self.model_name,
            p_female=p_female, age=age, ts=ts, frame_idx=frame_idx,
        )

    def _gender_age(self, image: np.ndarray, box: np.ndarray) -> tuple[float, float]:
        """InsightFace's Attribute.get (0.7.3) preprocessing, keeping the raw scores."""
        from insightface.utils import face_align

        m = self.genderage
        w, h = box[2] - box[0], box[3] - box[1]
        center = (box[2] + box[0]) / 2, (box[3] + box[1]) / 2
        aimg, _ = face_align.transform(image, center, m.input_size[0], m.input_size[0] / (max(w, h) * 1.5), 0)
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / m.input_std, tuple(aimg.shape[:2][::-1]),
                                     (m.input_mean,) * 3, swapRB=True)
        pred = m.session.run(m.output_names, {m.input_name: blob})[0][0]
        return gender_probability(pred[:2]), float(pred[2]) * 100.0


# -- pieces with no model, tested on their own ------------------------------------------

def gender_probability(scores: np.ndarray) -> float:
    """P(female) from the gender model's two raw scores (softmax; argmax agrees with InsightFace)."""
    z = np.asarray(scores, np.float64)
    e = np.exp(z - z.max())
    return float(e[FEMALE] / e.sum())


def head_pose(kps: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) degrees from 5 landmarks (left eye, right eye, nose, left mouth, right mouth).
    Approximate: 0 = facing the camera upright; yaw > 0 nose towards image right, pitch > 0 looking down."""
    le, re, nose, lm, rm = (np.asarray(p, np.float64) for p in kps)
    roll = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))
    c, s = math.cos(math.radians(-roll)), math.sin(math.radians(-roll))
    rot = np.array([[c, -s], [s, c]])
    eye_mid = (le + re) / 2
    le, re, nose, lm, rm = (rot @ (p - eye_mid) for p in (le, re, nose, lm, rm))   # eyes level, at 0
    iod = max(1e-6, re[0] - le[0])
    yaw = math.degrees(math.asin(float(np.clip(nose[0] / (0.5 * iod), -1, 1))))   # nose at an eye ~ profile
    mouth_y = (lm[1] + rm[1]) / 2
    ratio = nose[1] / mouth_y if mouth_y > 1e-6 else 0.5    # ~0.55 facing the camera
    pitch = float(np.clip((ratio - 0.55) / 0.25, -1, 1)) * MAX_PITCH
    return (round(pitch, 1), round(yaw, 1), round(roll, 1))


def size_quality(face_h: float, min_face_px: float) -> float:
    return float(np.clip((face_h - min_face_px) / max(1e-6, GOOD_FACE_PX - min_face_px), 0, 1))


def sharpness_quality(image: np.ndarray, box: np.ndarray) -> float:
    h, w = image.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, math.ceil(box[2])), min(h, math.ceil(box[3]))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0.0
    gray = cv2.cvtColor(cv2.resize(image[y1:y2, x1:x2], (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return float(np.clip(cv2.Laplacian(gray, cv2.CV_64F).var() / SHARP_REF, 0, 1))


def pose_quality(pose: tuple[float, float, float]) -> float:
    pitch, yaw, _ = pose
    return float(np.clip(1 - (yaw / MAX_YAW) ** 2 - (pitch / MAX_PITCH) ** 2, 0, 1))
