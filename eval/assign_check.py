"""T17 on real video: are faces given to the right person? Checked against the teacher's masks.

    python -m eval.assign_check clips/berlin_people_crossing.mp4 intel

For every person the live pipeline tracks, on every label frame: find faces in their head region
(core/identity/faces.py), assign them (core/identity/assign.py), and check each assigned face
against the teacher's full-resolution masks: the teacher person under the face must be the one
whose box matches the live owner's box. Prints how many faces were assigned or rejected (and why)
and every wrong assignment. "Overlapping people never get each other's face" = WRONG 0.
"""
import collections, sys, cv2, numpy as np
from pathlib import Path
from core.config_table import load_config
from core.contracts import Frame
from core.identity.assign import FaceAssigner, head_region
from core.identity.faces import FaceAnalyzer
from eval import score

clip = Path(sys.argv[1]); machine = sys.argv[2]
labels = score.load_labels(clip.stem); fps = labels["fps"]
truth = {f["idx"]: f for f in labels["frames"] if f["label"]}
cfg = load_config(machine); stride = round(fps / float(cfg.expected_fps))
cache = score.ModelCache.load_or_build(cfg, clip, stride)
det = score.ReplayDetector(cfg, fps / stride, cache)
analyzer, assigner = FaceAnalyzer(cfg), FaceAssigner(cfg)

def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = iw * ih; return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)

reasons = collections.Counter(); wrong, right, unsure = [], 0, 0
cap = cv2.VideoCapture(str(clip)); pos = 0
for idx in sorted(cache.rows):
    while pos < idx: cap.grab(); pos += 1
    img = cap.read()[1]; pos += 1
    dets = det.detect(Frame(img, idx / fps, idx, "x"))
    if idx not in truth: continue
    inst = cv2.imread(str(score.LABELS / clip.stem / "masks" / f"{idx:06d}.png"), cv2.IMREAD_UNCHANGED)
    tp = [p for p in truth[idx]["people"] if "id" in p]
    seen = {}
    for t in dets.tracks:
        if t.track_id is None: continue
        faces = analyzer.analyze(img, idx / fps, idx, region=head_region(t, assigner.head_zone, img.shape[:2]))
        for a in assigner.explain(faces, dets):
            key = tuple(np.round(a.face.bbox).astype(int))
            if key in seen and seen[key] == "assigned": continue   # the same face found from a neighbour's region
            seen[key] = a.reason
            reasons[a.reason] += 1
            if a.reason != "assigned": continue
            x1, y1, x2, y2 = a.face.bbox.astype(int)
            ids, counts = np.unique(inst[max(0,y1):y2, max(0,x1):x2], return_counts=True)
            ids, counts = ids[ids > 0], counts[ids > 0]
            owner = next(tt for tt in dets.tracks if tt.track_id == a.face.track_id)
            p_owner = max(tp, key=lambda p: iou(owner.bbox, p["box"]), default=None)
            if not len(ids) or p_owner is None or iou(owner.bbox, p_owner["box"]) < 0.3:
                unsure += 1; continue
            p_face = int(ids[counts.argmax()])
            if p_face == p_owner["id"]: right += 1
            else: wrong.append((round(idx / fps, 2), a.face.track_id, [round(float(v)) for v in a.face.bbox]))
print(f"{machine}: faces looked at {sum(reasons.values())}: {dict(reasons)}")
print(f"assigned faces checked against the teacher: right {right}, WRONG {len(wrong)}, can't tell {unsure}")
for w in wrong[:20]: print("   wrong:", w)
