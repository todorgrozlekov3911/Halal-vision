# Halal-Vision

Real-time video censoring based on the viewer's personal modesty settings. The first feature: censor people the viewer should not see. For a male viewer, that means women who are not on his family whitelist (wife, mother, sisters, …). The output is the live view in head-mounted display glasses or a video file (movie mode). Later stages add alcohol and romantic scenes.

> **Status:** planning for a 48-hour hackathon. This document is the shared source of truth. It records **what we decided, why, and what's still open**, so both team members (and the LLMs we brainstorm with) can merge ideas against it. Numbers marked *(estimate)* or *(starting value)* must be measured or tuned.

---

## Table of contents
1. [Scope](#1-scope)
2. [Core safety principles](#2-core-safety-principles)
3. [Hardware](#3-hardware)
4. [Architecture](#4-architecture)
5. [Interfaces (contracts)](#5-interfaces-contracts)
6. [Component: video input and output](#6-component-video-input-and-output)
7. [Component: detection, segmentation and tracking](#7-component-detection-segmentation-and-tracking)
8. [Component: rendering (blur and niqab)](#8-component-rendering-blur-and-niqab)
9. [Component: identity (gender, whitelist, policy)](#9-component-identity-gender-whitelist-policy)
10. [Configuration and environments](#10-configuration-and-environments)
11. [Latency and motion sickness](#11-latency-and-motion-sickness)
12. [Team split and timeline](#12-team-split-and-timeline)
13. [Testing, evaluation and metrics](#13-testing-evaluation-and-metrics)
14. [Risks and gotchas](#14-risks-and-gotchas)
15. [Later stages](#15-later-stages)
16. [Open questions](#16-open-questions)
17. [Decision log](#17-decision-log)
18. [Glossary](#18-glossary)

---

## 1. Scope

### MVP (the hackathon)
- Input: live camera (webcam, head-mounted USB camera, phone via scrcpy) or a video file.
- For each person: detect them, segment them (pixel mask), and track them over time.
- Decide per person whether to censor, based on **gender** and a **family whitelist** recognized by face.
- Render the censoring as **blur / pixelate / solid / niqab** (dark fill with an eye slit).
- Output to display glasses (fullscreen on a second monitor) and/or a file with the original audio.
- An enrollment UI for whitelisted family members.

### Out of scope for the hackathon (planned)
- Alcohol detection (stretch goal if time allows: YOLO-World).
- Romantic-scene detection.
- Running on the headset or glasses themselves, and reprojection.

---

## 2. Core safety principles

These rules shape every component. **No change should break them.**

1. **Show only censored frames.** No raw frame can reach the display. This is enforced by types: outputs accept only `CensoredFrame`, which only the renderer creates.
2. **Fail closed.** Any error, missing data or doubt leads to censoring.
   - A processing error shows the last censored frame, or black. Never the raw frame.
   - **New people start censored.** The policy can only *uncensor* people, never censor them late.
   - A detection without a track ID is still censored.
   - A missing or broken mask means the whole box is censored.
3. **Every uncensor decision needs confidence. Every censor decision is the default.**
4. **Prefer false alarms to misses.** Detection thresholds are low on purpose. A blurred coat rack is fine; an uncensored person is a leak.
5. **The eye slit is the only hole in the censoring.** It opens only when the eyes are confirmed in **the current frame**.
6. **Privacy.** Everything is processed on the device. Strangers' face fingerprints live only in memory, only while their track exists, and are never saved or logged. Only enrolled family members (with their consent) are stored.

---

## 3. Hardware

### 3.1 Display type: only video passthrough can censor
| Display type | Examples | Can it censor? |
|---|---|---|
| **Optical see-through** (clear lenses, display adds light) | Meta Ray-Ban Display, XREAL in AR mode, most "smart glasses" | **No.** The display can't block light. Black is transparent, so an overlay is a faint ghost over a fully visible person. **This can't be fixed in software.** |
| **Video passthrough** (you only see camera frames) | Quest 3, Apple Vision Pro, display glasses with lenses fully dimmed | **Yes.** What you see is exactly what the pipeline outputs. |

### 3.2 Hackathon demo setup
| Setup | Effort | Works? |
|---|---|---|
| **Webcam mounted on display glasses (XREAL/Viture), lenses dimmed to max, laptop shows the censored feed fullscreen** | Low | Yes (you still see reality at the edges of your vision). **Main demo.** |
| Quest 3 + Passthrough Camera API | High (Unity, camera access, inference on the headset) | Yes. **Roadmap, not the hackathon.** |
| Optical see-through glasses | Any | No |
| Movie mode (file → censored file) | Low | Yes. **Most reliable demo; always have one ready.** |

Pitch message: the product targets video-passthrough hardware, and we explain why.

### 3.3 Our laptops
| | NVIDIA laptop (teammate) | Intel laptop (Core Ultra 7 155U) |
|---|---|---|
| Hardware | NVIDIA GPU | 12 cores / 14 threads, 15 W chip; Intel Arc integrated GPU; Meteor Lake **NPU** (`/dev/accel0`); 30 GB RAM |
| Model runtime | PyTorch/CUDA or TensorRT FP16 | **OpenVINO** (integrated GPU, NPU or CPU) |
| Role | **Main live-demo machine** (most speed headroom) | **Weak-hardware check** and backup demo machine |
| Target | ≥ 30 fps | ≥ 15 fps (**the minimum: changes that break this don't get merged**) |

Pitch point: "runs at X fps on a 15 W laptop chip with an NPU" suggests it can run on wearable chips later.

### 3.4 Intel laptop: which processor runs what
| Work | Runs on |
|---|---|
| YOLO26-seg (every frame, heaviest) | Integrated GPU or NPU, **whichever the benchmark shows is faster** |
| EyeLocator (tiny face detector, every frame) | CPU |
| Identity worker (rare, background) | CPU, limited to 2–4 threads |
| Capture, mask resizing, rendering | CPU (OpenCV) |

### 3.5 Future hardware (undecided)
See [Open questions](#16-open-questions). Candidates: headset with its own chip (Quest-class), a phone or compute unit tethered to display glasses, or a mini-PC tethered to display glasses. All of them have an **NPU** but no big GPU. **ONNX is our common export format**, so we can move to any vendor's NPU toolkit.

---

## 4. Architecture

```
CameraSource ─┐                                                              ┌─→ DisplaySink (glasses / dev window)
              ├─→ newest-frame grabber → Frame ─→ [ PROCESS, fail-safe ] ─→ CensoredFrame ─┤
FileSource  ──┘   (live / realtime file)          │                                        └─→ FileSink (movie export / recording)
                  (offline file: no grabber,      │
                   every frame processed)         ▼
                                         ┌───────────────────────────────────────────────┐
                                         │ 1. YOLO26-seg track → tracks (A)              │
                                         │ 2. identity.update(frame, tracks) (B)         │──► worker thread (B):
                                         │    returns stored decisions instantly         │    face detection + gender +
                                         │ 3. EyeLocator on censored heads (A)           │    face fingerprint, CLIP fallback,
                                         │ 4. build mask → subtract eye slits → render(A)│    whitelist match → decisions{}
                                         └───────────────────────────────────────────────┘
```

Main loop (one sequence, no extra stages in between):
```python
last_safe = black_frame()
while True:
    frame = grabber.latest()                         # newest frame only
    try:
        tracks    = detector.track(frame)            # A
        decisions = identity.update(frame, tracks)   # B, never blocks (except sync mode)
        eyes      = eye_locator.find(frame, tracks, decisions)   # A
        out       = renderer.render(frame, tracks, decisions, eyes)  # A → CensoredFrame
        last_safe = out
    except Exception:
        log_error()
        out = last_safe                              # never the raw frame
    for sink in sinks:
        sink.write(out)
```

- **No separate watchdog.** If processing hangs, the display keeps the last censored image (safe). If it throws an error, the `try/except` shows the last safe frame. Optional for the product later: fade to black with a warning if the shown image is older than about 500 ms.
- **The source is irrelevant after the grabber.** Everything downstream sees only `Frame`.

### Repository layout
```
core/                      the one Python package; run with `python -m core.app --machine intel`
  app.py                   entry point (main loop lands here in T09)
  config_table.py          config loader (T03)
  contracts.py             shared types and interfaces (T02)
  video/                   sources.py, grabber.py, sinks.py                  (A)
  detect/                  detector.py, masks.py                             (A)
  render/                  renderer.py, eyes.py, niqab.py                    (A)
  identity/                module.py, face.py, gender.py, whitelist.py,
                           policy.py                                         (B)
system_configurations/     shared.yaml, intel.yaml, nvidia.yaml (+ trackers/ when tuned)
requirements/              base.txt, intel.txt, nvidia.txt, lock-<machine>.txt
setup.sh                   creates .venv for this machine
whitelist/                 ← gitignored (family photos)
clips/, crops/             ← gitignored (test footage)
```
Add folders (e.g. `core/eval/`, `tests/`) when the first task needs them. The video
package is `video/`, not `io/`: a folder named `io` shadows Python's own `io` module
whenever a script is run from inside `core/`.
**Never commit family photos, face fingerprints or test footage of real people.**

---

## 5. Interfaces (contracts)

Agree on these at hour 0, in `contracts.py`.

```python
@dataclass
class Frame:
    image: np.ndarray        # BGR uint8, full resolution
    ts: float                # capture time (time.perf_counter), used to measure latency
    idx: int
    source_id: str

@dataclass
class CensoredFrame:         # ONLY the renderer creates these; outputs accept only these
    image: np.ndarray
    ts: float
    idx: int

@dataclass
class Track:                 # A → B, one per person per frame
    track_id: int | None     # None = detected but not tracked yet → always censored
    bbox: np.ndarray         # x1, y1, x2, y2 in frame coordinates
    mask: np.ndarray         # low-resolution mask in model space (see §7.3)
    crop: np.ndarray         # person crop with ~10% padding, BGR
    is_new: bool

class TrackState(Enum):
    UNKNOWN = ...            # censored (default)
    CENSORED = ...           # confidently the filtered gender, not whitelisted
    ALLOWED_GENDER = ...     # confidently the other gender
    ALLOWED_WHITELIST = ...  # confirmed whitelisted family member

@dataclass
class Decision:              # B → A
    track_id: int
    censor: bool             # the only field the renderer needs
    state: TrackState        # for the debug overlay and the fade-out
    confidence: float

class IdentityModule:        # owned by B
    def update(self, frame: Frame, tracks: list[Track]) -> dict[int, Decision]:
        """Called by A EVERY frame. Must return a Decision for every track_id.
        Anything not yet decided → censor=True. Never blocks unless sync=True."""
```

**Color format:** OpenCV, YOLO and InsightFace use **BGR**. **CLIP/MobileCLIP need RGB**, so B converts at that call.

**Landmarks:** B's `Decision` does **not** carry face landmarks. A's `EyeLocator` finds eyes per frame (see §8.3).

---

## 6. Component: video input and output
**Owner: A** (B owns test data and enrollment capture)

### 6.1 Sources (two classes)
| Class | Covers | Notes |
|---|---|---|
| `CameraSource` | Any Linux video device `/dev/videoN`: laptop webcam, USB webcam on the glasses, **the phone via scrcpy**, and later a glasses camera that acts as a standard USB camera | `cv2.VideoCapture(i, cv2.CAP_V4L2)`, request **MJPG** format, 1280×720 at 60 fps if supported. On the Intel laptop consider 960×540 or 640×480 |
| `FileSource` | Any video file OpenCV or ffmpeg can open | Modes: **realtime** (keeps the video's timing, uses the grabber, drops frames like a camera) and **offline** (every frame, nothing dropped, no grabber) |

Phone as a camera: `scrcpy --video-source=camera --v4l2-sink=/dev/videoN` (Android 12+, needs v4l2loopback). **Use USB, not Wi-Fi.** Network streams (RTSP, IP-camera apps) are deliberately left out because of their buffering lag.

### 6.2 Newest-frame grabber
`CAP_PROP_BUFFERSIZE=1` isn't honored by every backend, so a background thread reads continuously and keeps only the latest frame:
```python
class LatestFrameGrabber(threading.Thread):
    def run(self):
        while self.running:
            ok, img = self.cap.read()
            if ok:
                with self.lock:
                    self.latest = Frame(img, time.perf_counter(), self.idx, self.source_id)
                    self.idx += 1
```
If processing is slower than the camera, older frames are dropped and never shown. **Every frame that is shown was censored. Not every camera frame is processed.**

### 6.3 Three ways to test with files
| Mode | Drops frames? | When |
|---|---|---|
| File, **offline** | No, every frame | Movie export, accuracy evaluation (repeatable) |
| File, **realtime** | Yes, like a camera | **Daily development**, testing latency and live behavior |
| File via **v4l2loopback** (fake camera device) | Yes, real camera code path | **Once at the start, and again after any change to the camera code** |

```bash
sudo modprobe v4l2loopback video_nr=10 card_label=fakecam exclusive_caps=1
ffmpeg -re -stream_loop -1 -i clips/wife_enters.mp4 -f v4l2 -pix_fmt yuyv422 /dev/video10
# then: CameraSource("/dev/video10")
```

### 6.4 Outputs (two classes)
| Class | Covers |
|---|---|
| `DisplaySink(monitor, fullscreen, debug_overlay)` | Glasses (fullscreen on the second monitor) **and** the dev window. Same code, different config. **The config must refuse `debug_overlay` raw views when the output is the glasses.** |
| `FileSink(path, audio_from=None)` | Movie export (ffmpeg pipe, H.264, original audio copied back in with `-map 0:v -map 1:a -c copy`) **and** recording the backup demo video |

Both outputs can run at once (show on the glasses and record at the same time). The debug overlay draws track IDs, state, `p_female`, best match score, fps and latency.

### 6.5 Latency measurement
Each `Frame` carries its capture timestamp, and the lag is computed when it's displayed. For the slides, measure real camera-to-screen latency: film a stopwatch app with the camera, photograph the stopwatch and the display together, and subtract.

---

## 7. Component: detection, segmentation and tracking
**Owner: A**

### 7.1 Model: YOLO26n-seg
Why YOLO26 rather than YOLO11: it outputs final detections directly (no NMS step, so lower and more predictable latency), it's optimized for CPU and edge devices, it's simpler to export, it has segmentation models, and it uses the same Ultralytics API (switching is one line). Both are AGPL-3.0.

```python
model = YOLO("yolo26n-seg.pt")   # or an exported .engine / OpenVINO folder, chosen by config
r = model.track(frame.image, persist=True, classes=[0], conf=0.15,
                imgsz=640, tracker="trackers/custom_bytetrack.yaml", verbose=False)[0]
```
| Argument | Meaning |
|---|---|
| `persist=True` | Keep tracker memory between calls. **Without it, IDs never stay the same.** |
| `classes=[0]` | Keep only "person". **It doesn't speed up the model**; it only filters the output. |
| `conf=0.15` | Low on purpose (prefer false alarms to misses) |
| `imgsz` | Model working size. NVIDIA: 640. Intel: about 416 *(choose by benchmark)*. Smaller = faster but misses distant people |
| `half=True` | NVIDIA only (FP16). Not useful on CPU |

What drives speed: model size (n < s < m < l < x), `imgsz`, precision (FP16/INT8), the number of people (one mask each), and hardware. The number of classes doesn't matter.

### 7.2 Intel laptop: OpenVINO benchmark (A's first task, decided by hour 3)
```python
model.export(format="openvino", imgsz=416)             # FP32/FP16
model.export(format="openvino", imgsz=416, int8=True)  # INT8
```
Measure **CPU vs integrated GPU vs NPU** × **imgsz 320 / 416 / 640** × **FP16 vs INT8**, for speed and mask quality. Recent Ultralytics versions accept `device="intel:gpu"` / `"intel:npu"`; if yours doesn't, call OpenVINO directly. NPU: fixed input size, slow first compile, so turn on OpenVINO's model cache.

### 7.3 What `r` contains, and full-resolution masks
The model works on a **letterboxed** image: a 1280×720 frame is shrunk to 640×360 and padded with gray bars to 640×384.

| Field | Content | Coordinates |
|---|---|---|
| `r.boxes.xyxy` | Boxes `(N, 4)` | Camera frame ✔ |
| `r.boxes.conf` | Confidence per person | |
| `r.boxes.id` | Track IDs, or `None` if not tracked yet | |
| `r.masks.data` | Masks `(N, 384, 640)`, a torch tensor | **Model space (letterboxed)** ✘ |
| `r.masks.xy` | Outline polygons | Camera frame ✔ |
| `r.masks` | **`None` when nobody is detected, so always check for it** | |

A **full-resolution mask** is the same size as the camera frame, so `mask[y, x]` lines up with frame pixel `(x, y)`. The renderer needs that for `np.where(mask[..., None], blurred, frame)`.

**Chosen approach: merge the censored masks at low resolution, remove the letterbox bars, resize once.** It's cheap (one resize regardless of how many people) and keeps every piece of a mask. Rejected: `retina_masks=True` (slow, cost grows per person) and polygons with `fillPoly` (can drop or wrongly join pieces when a person is split by something in front of them).

```python
def censor_mask_fullres(r, censor_flags, frame_shape, kernel):
    h0, w0 = frame_shape[:2]
    if r.masks is None or not any(censor_flags):
        return np.zeros((h0, w0), bool)
    m = r.masks.data[censor_flags].amax(dim=0)              # merge censored masks
    mh, mw = m.shape
    gain = min(mh / h0, mw / w0)
    pad_x, pad_y = (mw - w0 * gain) / 2, (mh - h0 * gain) / 2
    m = m[int(pad_y):int(mh - pad_y), int(pad_x):int(mw - pad_x)]   # remove the letterbox bars
    m = (m.float().cpu().numpy() * 255).astype(np.uint8)
    m = cv2.resize(m, (w0, h0), interpolation=cv2.INTER_LINEAR)
    m = cv2.dilate(m, kernel)                                # grow edges
    return m > 127
```
Print `r.masks.data.shape` once to confirm the layout on your setup. Rounding can leave the mask off by a pixel; dilation covers it.

### 7.4 Tracking
- **ByteTrack or BoT-SORT: pick one** in the tracker settings file. BoT-SORT = ByteTrack + camera-movement correction + optional appearance matching (`with_reid`), and it's slower. **Start with ByteTrack, then compare BoT-SORT on head-mounted footage.**
- How it works: a Kalman filter **predicts** each person's next box from their position and speed; new detections are **matched** to predictions by box overlap. ByteTrack's trick: confident detections are matched first, then **low-confidence** ones against the people still unmatched, which keeps half-hidden people tracked.
- **Lost tracks:** an unmatched track is marked "lost" and its predicted position keeps moving. A detection near that spot within `track_buffer` frames gets the old ID back. After that the track is deleted, and if the person shows up again they get a **new ID**. `track_buffer` counts **frames**, so the time depends on fps.
- ByteTrack uses **only position and movement.** It fails when someone leaves and comes back, when someone changes direction while hidden, and on fast head turns (use BoT-SORT's camera-movement correction for those). **No tracker remembers people long-term.** B's face recognition does that.
- **Danger: an ID taken over by the wrong person** after an occlusion or crossing. That's why B re-checks after gaps and crossings (§9.6).

Tracker settings (copy Ultralytics' `bytetrack.yaml` into `trackers/`):
- `track_buffer`: 30–60 *(starting value)*
- `new_track_thresh`: about 0.25, so new people get a track quickly
- **Check whether detections that haven't become tracks yet are returned at all.** Censor every detection without an ID.

### 7.5 Two separate "memories"
| | Controls | Owner |
|---|---|---|
| `track_buffer` | How long an **ID** is remembered after a person is lost | Tracker setting (A) |
| Mask **hold** (~0.5 s) | How long we keep **drawing the censor** after a person is lost | A's renderer |

---

## 8. Component: rendering (blur and niqab)
**Owner: A**

### 8.1 Basics
- **Dilation:** grow the mask by about 3–5% of the person's height (or start with one fixed kernel). It covers hair, coarse edges and misalignment.
- **Hold:** keep each track's last low-resolution mask and its speed. When the track goes missing, keep adding the mask, shifted along its last direction of movement, into the merged mask for about 0.5 s.
- **Blur:** blur the whole frame once (shrink, blur, enlarge again is cheap), then `np.where(mask, blurred, frame)`. **Make it strong**, so it's unrecognizable and not just soft.
- **Modes:** `blur | pixelate | solid | niqab`.
- **Transitions:** **censoring starts instantly** (never faded in). **Uncensoring may fade out** over about 200 ms (smooth when the wife is recognized).

### 8.2 Niqab mode: the eye slit
Build the full cover mask, then **subtract an eye slit** for each censored person whose eyes are found **in this frame**, then fill the remaining mask with dark "fabric".
```
censored masks (merged) → resize → DILATE = full cover
                                           − eye slit per person (this frame only)
                                           = niqab mask → fill with dark fabric
```
```python
def draw_eye_slit(slit_mask, kps, w_factor=2.2, h_factor=0.45):   # starting values, tune by eye
    le, re = kps[0], kps[1]                     # left eye, right eye (frame coordinates)
    d = np.linalg.norm(re - le)                 # distance between the eyes
    if d < 8:                                   # too far away → no slit
        return
    cx, cy = (le + re) / 2
    angle = np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0]))   # head tilt
    axes = (int(w_factor * d / 2), int(h_factor * d / 2))
    cv2.ellipse(slit_mask, (int(cx), int(cy)), axes, angle, 0, 360, 255, -1)

niqab = cover & (slits == 0)                    # subtract AFTER dilating
out[niqab] = FABRIC_COLOR
```
- The slit scales with distance and rotates with head tilt. In side profile it shrinks automatically. In a back view no face is found, so the person is fully covered (correct).
- **Overlap rule:** skip the slit if the head region overlaps another censored person's box. Otherwise the slit could cut a hole in the other person's cover.
- Polish later: a dark color instead of pure black, some noise or fabric texture, a softened slit edge, a darker band around the slit.

### 8.3 EyeLocator (per frame)
Eye positions must come from the **current frame**. B's checks are too infrequent, and old landmarks would misplace the slit.
- For each censored person: take the **head region** (top ~35% of the box, padded) and run **SCRFD-500M** (small InsightFace detector) at 160–224 px.
- **Cap:** only the 3–4 largest (nearest) censored people get slits; everyone else is fully covered.
- Smooth the eye positions slightly over time.
- **No face found this frame means no slit.** At most 1–2 frames of hold, if the slit blinking on and off looks bad.

---

## 9. Component: identity (gender, whitelist, policy)
**Owner: B**

### 9.1 Runs in a background thread, not a process
A full check takes roughly 20–80 ms *(estimate; longer on CPU)*. If the main loop waited, video would stutter whenever someone walks in. `identity.update()` returns stored decisions immediately, queues work for a worker thread and returns `censor=True` for anything not yet decided.

Why a thread, not a process: easy sharing, one model setup, easy debugging. ONNX Runtime, OpenVINO and PyTorch release Python's GIL during inference. Switch to a process only if measurements show the worker slowing the main loop.

```python
class IdentityModule:
    def __init__(self, sync=False):
        self.decisions, self.lock = {}, threading.Lock()
        self.jobs = queue.Queue(maxsize=8)
        self.sync = sync
        if not sync:
            threading.Thread(target=self._worker, daemon=True).start()

    def update(self, frame, tracks):
        out = {}
        for t in tracks:
            if self._needs_check(t):                 # new / unsure / re-check rule (§9.6)
                job = (t.track_id, t.crop.copy(), frame.idx)   # COPY the crop
                if self.sync:
                    self._run_check(*job)            # offline: wait (repeatable results)
                else:
                    try: self.jobs.put_nowait(job)
                    except queue.Full: pass          # try again next frame
            with self.lock:
                out[t.track_id] = self.decisions.get(t.track_id, Decision.censor(t.track_id))
        return out
```
- **A calls `update()` every frame; B decides internally what to check.** New track → queue a check. Confident → reuse the stored decision. Unsure → retry every N frames. Whitelisted → re-confirm periodically. Track gone → forget it after the tracker's buffer expires.
- Queue order: new people before re-checks. Skip jobs for tracks that are gone. Mark tracks "check pending" so they aren't queued twice.
- **`sync=True` for offline mode** (movie export, evaluation), so results don't depend on timing.
- Limit threads: `intra_op_num_threads=2–4` (ONNX Runtime), `torch.set_num_threads(...)`.
- On the NVIDIA laptop, if identity models on the GPU cause frame-time spikes, run them on the CPU.

### 9.2 Face analysis (InsightFace)
```python
app = FaceAnalysis(name="buffalo_l",            # NVIDIA config; Intel config: "buffalo_s"
                   allowed_modules=["detection", "genderage", "recognition"])
app.prepare(ctx_id=0, det_size=(320, 320))
faces = app.get(person_crop)                    # .sex, .age, .kps, .embedding, .det_score
```
- `buffalo_s` = SCRFD-500M detector + MobileFaceNet fingerprint model: several times faster on CPU, slightly less accurate. **Check that it includes the gender model**; if not, load the gender model from `buffalo_l`.
- Convert landmarks from **crop coordinates to frame coordinates** if you ever pass them on.
- **License:** InsightFace's pretrained models are non-commercial only. Fine for the hackathon; flag it for any product.

### 9.3 Gender
1. **Face model** when the face is clear: detection confidence above about 0.6 and face at least about 40 px.
2. **Body fallback** otherwise (back view, too far, turned away): zero-shot classification on the person crop. NVIDIA: CLIP/SigLIP. Intel: **MobileCLIP-S0**. Average over several prompts ("a photo of a woman", "a woman seen from behind", …). Compute the prompt text embeddings **once at startup**.
3. Output **`p_female`**, not a hard label. The policy maps it to a decision depending on the viewer's gender.
4. **Combine evidence over time per track** (add up log-odds across checks); decide only once confidence passes a threshold.
5. **Unequal errors:** a woman classified as a man = **leak (push close to 0)**. A man classified as a woman = an unnecessary blur (annoying but harmless). Set thresholds accordingly.

### 9.4 Enrollment
- Gradio app (webcam + upload). **10–20 photos per person:** straight on, ±30° turns, looking up and down, indoor, outdoor and dim light, **with and without hijab, with and without glasses**, and some taken with the **actual demo camera in the demo room**.
- Quality check per photo: exactly one face, detection confidence above 0.7, face at least 80 px wide, not blurry (`cv2.Laplacian(gray).var()` threshold). Reject with a reason so the person can retake it.
- Storage (gitignored):
```
whitelist/<person>/
  photos/*.jpg
  meta.json                   # name, relationship, enrollment date
  embeddings.buffalo_l.npy    # NVIDIA config
  embeddings.buffalo_s.npy    # Intel config
```
**Fingerprints from different models can't be compared.** Keep the photos and compute fingerprints per model, cached under that model's name.

### 9.5 Whitelist matching
- Cosine similarity between the face fingerprint and every enrolled photo's fingerprint; average the best few per person.
- **Only judge good faces:** detection confidence above 0.6, face at least about 50 px, roughly facing the camera (nose roughly centered between the eyes). A poor face = "no evidence" (stays censored), **not** "not the wife".
- **Accept threshold (high):** zero strangers on the test clips may pass. A stranger matched as the wife is a leak.
- **Reject threshold (low):** a clear "not this person", used to catch ID swaps.
- **Tune separately for `buffalo_l` and `buffalo_s`.** The thresholds don't transfer.

### 9.6 Policy: per-track state machine
```
                ┌── confident other gender ───────────► ALLOWED_GENDER    (not censored)
UNKNOWN ────────┤
(censored)      ├── whitelist match in 2–3 checks ────► ALLOWED_WHITELIST (not censored)
                └── confident filtered gender ────────► CENSORED

Any ALLOWED state ── re-check fails / ID swap suspected ──► UNKNOWN (censored)
```
- Only UNKNOWN → ALLOWED needs confidence. Everything else falls back to censored.
- The whitelist needs **2–3 checks in agreement**, not a single lucky frame.
- **Re-check rules for ALLOWED states** (B can detect gaps and overlaps itself, since it sees every frame's track list):
  - Whitelisted people: re-check about every 1 s.
  - After a **gap of more than about 5 frames**: back to UNKNOWN (censored) until re-confirmed.
  - After the track **crossed or overlapped** another person (when ID swaps happen): re-check immediately.
  - A clear "not this person" → UNKNOWN right away.
- CENSORED decisions never need re-checking.
- **Privacy:** a stranger's fingerprint exists only in memory, only while their track exists.

---

## 10. Configuration and environments

### 10.1 Same code, one config per machine (no `if nvidia:` in the code)
```yaml
# configs/config.nvidia.yaml               # configs/config.intel.yaml
detector:                                  detector:
  weights: yolo26n-seg.engine                weights: yolo26n-seg_openvino_model/
  device: cuda:0                             device: intel:gpu        # or intel:npu (benchmark)
  imgsz: 640                                 imgsz: 416
identity:                                  identity:
  onnx_providers: [CUDAExecutionProvider]    onnx_providers: [OpenVINOExecutionProvider, CPUExecutionProvider]
  face_pack: buffalo_l                       face_pack: buffalo_s
  body_model: clip                           body_model: mobileclip-s0
```
Shared settings:
```yaml
source: { type: camera, uri: /dev/video0, width: 1280, height: 720, fps: 60 }   # or type: file, mode: realtime|offline
sinks:  [ { type: display, monitor: 1, fullscreen: true, debug_overlay: false } ]
viewer:
  gender: male                     # sets which gender gets censored
  whitelist: [wife, mother, sister_amina]
  censor_children: true            # OPEN QUESTION (§16)
render_mode: niqab                 # blur | pixelate | solid | niqab
```

### 10.2 Installs
```
requirements-common.txt   # ultralytics, insightface, opencv-python, gradio, numpy, pyyaml, …
requirements-nvidia.txt   # -r common + torch (CUDA build), onnxruntime-gpu, tensorrt
requirements-intel.txt    # -r common + torch (CPU build), openvino, onnxruntime-openvino
```
`onnxruntime` and `onnxruntime-gpu` can't be installed together. **Set up both environments in hours 0–2.**

### 10.3 Intel laptop setup
- Plug in power and set performance mode: `powerprofilesctl set performance`. Test that the demo still holds up **after 10 minutes** (the chip slows down when hot).
- Install Intel's **compute runtime** (OpenCL/Level Zero) for the integrated GPU and Intel's **user-space NPU driver**. The kernel driver is already loaded. Test both in the first hour.

---

## 11. Latency and motion sickness

**Cause:** the inner ear senses a head turn right away, but the image catches up 50–120 ms later *(estimate)*, so the world seems to "swim".

| Fix | Hackathon? |
|---|---|
| **Cut latency:** no camera buffering (newest-frame grabber), 60–120 fps camera, smaller `imgsz`, FP16/INT8, no queues between stages | **Yes, about 1 hour.** It also raises fps |
| **Steady frame timing:** stutter is worse than a constant delay | Yes |
| **True-size view:** match the camera's field of view to the display's; camera close to the eyes | Partly |
| **Reprojection:** warp the image by the head rotation (from the IMU) since capture, just before showing it. The censor is baked in, so it moves with the person. Capture a wider field of view than you display, to have margin for the warp | **Roadmap** |
| **Stereo** (two cameras) | Roadmap |
| **Use the headset's built-in passthrough** (Quest) and draw opaque censor shapes on top. The headset handles comfort; masks then trail moving people (needs larger masks for fast movers and motion prediction), and new people may be uncovered for a few frames | Roadmap (product) |

Demos are short (nobody gets sick in 2 minutes). Measure and report latency, and present reprojection as the roadmap.

---

## 12. Team split and timeline

### 12.1 Split: "pixels" (A) and "identity" (B)
**Person A: where people are, how they're censored, how frames are shown**
- Video input and output, newest-frame grabber, the "show only censored frames" rule, outputs, latency measurement
- YOLO26-seg + tracking, tracker tuning, the OpenVINO/TensorRT benchmark
- Mask stability (dilation, hold, no flicker), rendering modes, EyeLocator, niqab, fade-out
- Movie mode; the display-glasses demo; stretch: alcohol detection via YOLO-World

**Person B: who each person is, whether to censor them**
- Face analysis, gender (face + body fallback), evidence over time
- Enrollment UI, whitelist storage and matching, threshold tuning
- Identity worker thread, state machine, re-check rules, viewer profile
- Test clips, crop dumps, labels, evaluation tool and metrics

**Avoiding blocking:** A starts with a stub (`censor=True` for everyone, giving "blur everyone"). At hour 1, A dumps person crops from a test video, so B works on crops and photos without waiting. Each person can demo their half alone if a merge slips.

### 12.2 Timeline
| Hours | Person A | Person B |
|---|---|---|
| 0–2 | Repo, both environments, video input, dump crops | Environment, enrollment photos (with consent), record and label test clips |
| 0–3 | **Hardware benchmark → decide imgsz, precision and device** | |
| 2–10 | Segmentation, tracking, "blur everyone" on the stub, show only censored frames | Face analysis and gender (face + body fallback), measured on crops |
| **10** | **Merge 1: only women blurred** | |
| 10–20 | Mask hold, fade-out, EyeLocator + niqab, debug overlay, latency fixes | Enrollment UI, whitelist matching, state machine, re-check rules, worker thread |
| **20** | **Merge 2: full MVP** | |
| 20–26 | Sleep | Sleep |
| 26–34 | Glasses demo, TensorRT/OpenVINO tuning, movie mode | Threshold tuning, hard cases, settings UI |
| 34–40 | Alcohol detection (stretch) | Evaluation numbers and charts |
| 40–48 | **Together:** feature freeze, backup demo video, slides (including the passthrough explanation and the reprojection roadmap), rehearse | |

At every merge, run the same test clip on **both** laptops and write down fps and latency.

---

## 13. Testing, evaluation and metrics

### 13.1 Test data (B)
- **Clips** (phone, 720p): wife entering the view, women walking past, back views, side views, a partial person at the frame edge, a group, people crossing, low light; with and without hijab. Plus a few public movie clips for movie mode.
- **Crop dump:** `crops/{clip}/{frame}_{track}.jpg` + `tracks.json` (boxes, IDs).
- **Labels per track, not per frame:** one thumbnail per track ID → CSV with `clip, track_id, gender, identity`. Takes minutes.

### 13.2 Metrics
| Metric | Target |
|---|---|
| **Uncensored frames shown** (a person who should be censored is visible) | **0** |
| Strangers matched as a whitelisted person | **0** |
| Women classified as men (per track; reported separately for back views) | Close to 0 |
| Time until a whitelisted person is uncensored | About 1 s or less |
| fps / latency on the Intel laptop | ≥ 15 fps, < 120 ms *(target)* |
| fps / latency on the NVIDIA laptop | ≥ 30 fps |

### 13.3 Done when

**Video input and output**
- [ ] Webcam ≥ 30 fps through the stub (NVIDIA), latency shown on screen
- [ ] Phone works as a source via scrcpy
- [ ] An error in processing shows black or the last censored frame, never raw video
- [ ] An offline file export includes audio
- [ ] Fullscreen on the second monitor works (test with any external monitor)
- [ ] The fake-camera (v4l2loopback) check passes

**Merge 1 (hour 10)**
- [ ] Masks cover hair and edges; IDs stay the same through a short occlusion
- [ ] A detection without a track ID is still censored
- [ ] Gender per-track results on labeled crops, women classified as men close to 0
- [ ] Only women blurred on test clips in realtime file mode

**Merge 2 (hour 20)**
- [ ] The wife is uncensored within about 1 s of showing her face, with and without hijab
- [ ] Zero strangers matched as the wife on all clips
- [ ] A woman who takes over the wife's ID after a crossing is censored again within a few frames
- [ ] No flicker on short overlaps or occlusions
- [ ] The eye slit follows the eyes, scales and tilts; back views are fully covered; the slit closes within 1–2 frames when the face is lost; overlapping censored people never show a hole in each other
- [ ] EyeLocator costs about 3 ms per person at most; the cap works
- [ ] Intel ≥ 15 fps, NVIDIA ≥ 30 fps on the same clip

---

## 14. Risks and gotchas

| Risk | Mitigation |
|---|---|
| Woman classified as a man (side view, no visible face, long-haired men / short-haired women for the body model) | Fail closed, body fallback, evidence over time, strict thresholds |
| Tracker gives the wrong person the wife's ID | Re-check after gaps and crossings, reject threshold, strict accept threshold |
| Wife not recognized side-on | She stays censored until facing the camera (acceptable); varied enrollment photos |
| Look-alikes (cousin, sister-in-law) | Strict accept threshold |
| Tracks lost on fast head turns | BoT-SORT camera-movement correction |
| Flicker | Dilation, hold, fade-out only when uncensoring |
| Low fps on the Intel laptop | OpenVINO on integrated GPU/NPU, INT8, imgsz 416, lower capture resolution, cover the whole box for small or unsure detections |
| The 15 W chip slows down when hot | Power plugged in, performance mode, 10-minute test |
| Demo-day failure | Pre-recorded backup video; movie-mode clips rendered in advance (movie mode runs slower than real time) |
| **OpenCV package conflict** | Only `opencv-python`, never also `opencv-python-headless` (`imshow` breaks) |
| **Wayland** | `QT_QPA_PLATFORM=xcb` or an X11 session |
| Webcam ignores the fps or resolution you set | `v4l2-ctl --list-formats-ext -d /dev/video0`; MJPG |
| Fullscreen opens on the wrong monitor | Hardcode the monitor's position in the config |
| `insightface` install | Needs `build-essential` and `python3-dev` (it compiles); pin versions |
| `onnxruntime` vs `onnxruntime-gpu`, CUDA version mismatches | Separate requirements files per machine, pin versions early |
| Intel integrated GPU / NPU drivers on Linux | Install and test in the first hour |
| Licensing | Ultralytics YOLO: AGPL-3.0 (commercial use needs an enterprise license). InsightFace pretrained models: non-commercial only |

---

## 15. Later stages

All censor types should be **plugins with the same `detect → decide → render` interface**.
- **Alcohol:** open-vocabulary detection ("beer bottle", "wine glass", …): YOLO-World (real-time) or OWLv2 (more accurate, slower). Blur the object's box or mask. Stretch goal for hours 34–40.
- **Romantic scenes:** classify the whole scene (CLIP or a small vision-language model) on sampled frames, then skip or blur the scene. Movies can also use subtitles and audio. Much easier offline than live.

---

## 16. Open questions

1. **Children:** should children be censored? The face model gives an age estimate. It's a config setting (`censor_children`), and the team needs to decide the default.
2. **Viewer = woman:** what exactly gets censored? The policy supports it (`viewer.gender`), but the rules need defining.
3. **Future hardware:** headset with its own chip vs a phone or compute unit + display glasses vs a mini-PC + display glasses. Decide after the hackathon; it implies different software stacks.
4. **Intel laptop:** integrated GPU or NPU for YOLO? Decided by the hour 0–3 benchmark.
5. **Tracker:** ByteTrack or BoT-SORT? Decided by testing on head-mounted footage.
6. **Who is A and who is B,** and which laptop each person develops on (A must test on the Intel laptop at every merge).
7. Do we have access to display glasses (XREAL/Viture) for the demo?
8. Thresholds (gender confidence, whitelist accept/reject, dilation size, slit size) are all *starting values* to be tuned.

---

## 17. Decision log

Decisions made during planning, including ones we reversed, so nobody re-litigates them by accident.

| # | Decision | Reason | Replaced |
|---|---|---|---|
| 1 | **YOLO26n-seg** | No NMS step (lower, predictable latency), faster on CPU and edge devices, simpler export, same API | YOLO11n-seg (no real reason for it) |
| 2 | **Split "pixels" (A) vs "identity" (B)** | Both halves are computer vision; video input and output is only about 2 hours of work | A = all computer vision, B = input/output and UI (unbalanced) |
| 3 | **Video passthrough only**; demo with dimmed display glasses + webcam | Optical see-through can't block light | — |
| 4 | **Show only censored frames**, enforced by types | "Censor immediately" | — |
| 5 | **New people start censored; the policy only uncensors** | Delay in identity checks must not leak | — |
| 6 | **Two source classes** (Camera, File) | scrcpy makes the phone a normal camera | A separate `PhoneSource` |
| 7 | **Daily testing with realtime `FileSource`; v4l2loopback check once and after changes to the camera code** | Repeatable and simple; the camera path still gets tested | — |
| 8 | **`try/except` fail-safe, no watchdog** | Sequential loop: a hang already keeps the last censored frame | A separate watchdog component |
| 9 | **Two output classes** (Display, File); debug view is a flag | Four classes were really two with different configs | Glasses / File / Debug / Record outputs |
| 10 | **A calls `identity.update()` every frame; B decides what to check** | Check scheduling is identity logic | A decides when to call B |
| 11 | **Masks: merge at low resolution, remove bars, resize once** | Cheap, keeps every piece | Polygons + `fillPoly` |
| 12 | **EyeLocator owned by A, runs every frame**; landmarks removed from `Decision` | Old landmarks misplace the slit | B returning landmarks |
| 13 | **Re-check ALLOWED tracks after gaps (>5 frames) and crossings** | The tracker can give the wrong person an ID | — |
| 14 | **Identity in a background thread, `sync=True` offline** | No stutter; repeatable evaluation | Separate process |
| 15 | **Hardware-agnostic code, one config per machine; OpenVINO on Intel, CUDA/TensorRT on NVIDIA** | Two different laptops; future hardware unknown | CUDA-only plan |
| 16 | **Intel laptop = minimum performance target; NVIDIA = demo machine** | Keeps the design viable for wearable chips | — |
| 17 | **`buffalo_s` + MobileCLIP-S0 on Intel; `buffalo_l` + CLIP on NVIDIA**; fingerprints and thresholds per model | Speed on CPU; fingerprints from different models can't be compared | — |
| 18 | **ONNX as the common export format** | Every NPU vendor's toolkit can import it | — |

---

## 18. Glossary

| Term | Meaning |
|---|---|
| **Bounding box** | Rectangle around a person: `(x1, y1, x2, y2)` |
| **Segmentation mask** | Per-pixel yes/no grid: does this pixel belong to the person? |
| **Full-resolution mask** | A mask the same size as the camera frame, aligned pixel for pixel |
| **Letterboxing** | Resizing an image to the model's size while keeping its shape, padding with gray bars |
| **Dilation** | Growing a mask outward by a few pixels (`cv2.dilate`) |
| **Track / track ID** | One person followed across frames; the tracker gives them a stable ID |
| **Kalman filter** | Predicts each track's next position from its position and speed |
| **`track_buffer`** | How many frames a lost track is remembered before its ID is dropped |
| **Hold** | How long we keep drawing a censor after its person disappears |
| **ByteTrack / BoT-SORT** | Two alternative trackers; BoT-SORT adds camera-movement correction and appearance matching |
| **Landmarks (kps)** | Key face points: 2 eyes, nose, 2 mouth corners (InsightFace gives 5) |
| **Embedding / face fingerprint** | A 512-number vector describing a face; similar faces → similar vectors |
| **Cosine similarity** | How alike two fingerprints are (−1 to 1; higher = more alike) |
| **Log-odds** | A way to add up evidence from several checks into one confidence |
| **Zero-shot (CLIP)** | Classifying an image by comparing it to text prompts, without training |
| **Fail closed** | When in doubt, censor |
| **Optical see-through / video passthrough** | Seeing reality through clear lenses vs seeing only camera frames |
| **Reprojection** | Warping the image by the latest head rotation just before showing it, to hide latency |
| **NPU** | Chip dedicated to running neural networks efficiently |
| **OpenVINO** | Intel's toolkit for running models on Intel CPUs, integrated GPUs and NPUs |
| **TensorRT** | NVIDIA's toolkit for fast model inference on NVIDIA GPUs |
| **FP16 / INT8** | Lower-precision numbers (16-bit float / 8-bit integer): faster, slightly less accurate |
| **GIL** | Python's global lock; model libraries release it while they run, so threads still work |
| **v4l2loopback** | Linux module that creates a fake camera device |
| **scrcpy** | Tool that mirrors an Android phone (including its camera) to the computer |
