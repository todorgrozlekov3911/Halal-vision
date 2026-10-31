Halal-Vision: merged design and execution plan
Oct 31, 2026 · @Toto
Purpose
This plan merges the Halal-Vision README with a second design discussion into one design and one task roadmap. The README stays the reference for details it already covers (code snippets, setup, metrics); this doc records where the merge changed or added something, and defines every task with its prerequisites.
The goal is unchanged from the README: real-time video censoring based on the viewer's modesty settings. The first feature censors people a male viewer should not see. The horizon now reaches past the 48-hour hackathon, so the roadmap has phases: an MVP first, then hardening and product work.
There is no fixed A/B team split anymore. Work is organized as tasks; anyone can pick up a task once its prerequisites are done.
Merged decisions
These decisions extend the README's decision log (#1–#18). Where one replaces a README decision, the last column says which.
#
Decision
Reason
Changes README
19
Horizon goes past the hackathon; the roadmap has phases
The design should grow into a product
Timeline (§12) becomes phase 1 of the roadmap
20
Censoring unit = the whole person (segmentation mask)
Safer than faces; already designed
Confirms §7
21
Censor style for women = niqab, always
Product rule
—
22
Men are not censored for now; the policy keeps a general "conceal" action so this can change later
Design around the domain, not one rule
—
23
Viewer = male only, for now
Scope
Closes §16 Q2 for now
24
Children: a person confidently estimated at 10 years or younger is not censored; everyone else is treated as an adult
Product rule; uncertain age fails closed
Closes §16 Q1
25
Gender is its own component: estimators produce evidence, a decider combines it through a list of rules. No decision on body-only evidence by default
Room for more logic later at no frame-time cost
Refines §9.3
26
Actions are general: pass, conceal (style), decorate (asset). Only pass and conceal(niqab) are built now
Keeps asset overlays possible without a redesign
Decision.censor: bool becomes an action
27
Three threads (grabber, main loop, identity worker), plus a writer thread whenever a file is written
A slow encoder must not stall the main loop
Adds to §4
28
Writer queue: blocking in offline export, drop-oldest when recording live
No lost frames offline; no frozen display live
—
29
Crossings: when tracks overlap, both are censored immediately and stay censored until re-verified
Closes the ID-swap leak window
Tightens §9.6
30
All temporal rules are written in seconds and checked against Frame.ts; offline mode uses video time
Same behavior at 15 and 30 fps
Replaces frame counts in §7.4, §9.6
31
Faces are assigned to people through the segmentation mask; an ambiguous face gives no evidence and no slit
Boxes overlap; masks follow the body
New
32
YOLO runs through Ultralytics behind our own Detector interface
Fast start; swappable backend later
Refines #1
33
The eye slit is an end-goal task; until then the niqab is full cover
Full cover is the safe default
Moves §8.2–§8.3 to a later phase
34
Both machines run Linux; one config per machine for now, automatic hardware selection later
Matches the README's setup
Adds a later task to #15
Domain model and policy
Every tracked person starts concealed; rules can only lift that, and only with enough confidence. The model below is written around the domain (subjects, evidence, actions), not around one rule, so new viewer profiles or censor types add rules instead of code paths.
Concepts
Concept
What it is
Produced by
Subject
One tracked person: track ID, box, mask, timestamps (first seen, last seen)
Detector + tracker
Face observation
A face assigned to a subject: box, 5 landmarks, quality, embedding
Identity worker (face analysis + mask assignment)
Evidence
A typed claim about a subject with a confidence and a source: gender (p_female), age, identity match
Estimators (gender, age, whitelist matcher)
Viewer profile
Who is watching and their settings: gender (male only for now), whitelist, conceal style
Config
Rule
Condition on evidence → action, with the confidence it needs
Policy config
Action
pass, conceal(style) or decorate(asset)
Policy
Decision
The current action for a subject, its state and confidence
Identity component (owner of decision state)
Rules for the current viewer profile (male)
Rules are checked in order; the first rule whose evidence is confident enough wins. If none fires, the default applies.
Order
Condition
Action
Evidence required
1
Overlapping another subject, or re-verification pending after a crossing or gap
conceal(niqab)
None: safety override
2
Confirmed whitelisted family member
pass
2–3 agreeing face matches above the accept threshold
3
Confidently male
pass
Face-based gender evidence; body-only evidence is not enough by default
4
Confidently 10 years or younger
pass
Face-based age evidence, confidence above threshold
—
Default (no rule fired)
conceal(niqab)
—
Uncertain age counts as adult, and uncertain gender counts as female. Both fall through to the default.
Gender component
Gender is split into two parts. Estimators (face model, later a body model) each produce evidence with a source and a quality score. A decider combines evidence over time (log-odds per track) and applies a list of decision rules; the default rule set requires face evidence. Adding logic later means adding a rule or an estimator, both running in the identity worker, so the frame loop does not slow down.
Contract changes
Decision.censor: bool becomes an action. The renderer only reads the action.
class ActionKind(Enum):
    PASS = "pass"
    CONCEAL = "conceal"      # style: niqab | blur | pixelate | solid
    DECORATE = "decorate"    # asset overlay; later phase

@dataclass
class Action:
    kind: ActionKind
    style: str | None = None
    asset: str | None = None

@dataclass
class Decision:
    track_id: int
    action: Action
    state: TrackState
    confidence: float
Detections without a track ID never get a Decision; the renderer conceals them directly.
Architecture
The system runs as three threads plus a writer thread whenever a file is written; only the renderer creates frames that reach a sink. The main loop never waits for identity: it reads the latest decisions and queues jobs, and the worker answers later.
Frames flow down the main loop; identity jobs go right to the worker, and decisions come back through the decision store, which the identity component owns.
Thread
Runs
Talks to others through
Grabber
Source reads; keeps only the newest frame (live and realtime-file modes only)
Newest-frame slot
Main loop
Detect + track, identity.update, mask pipeline, render, display, fail-safe
Slot in; job queue out; decision store in; writer queue out
Identity worker
Face analysis, face-to-mask assignment, estimators, policy decider
Priority job queue in; decision store out (under a lock)
Writer
ffmpeg encoding, audio remux
Writer queue: blocking offline, drop-oldest live
In offline mode there is no grabber: the main loop reads every frame, and identity runs in sync mode so results do not depend on timing.
The crossing and gap rules run inside identity.update on every frame, not in the worker, because they need each frame's track list and must act in the same frame.
Safety invariants
The README's core principles (§2) stay in force; the merge adds four. No task may break any of them, and each has a test in the evaluation harness.
1. Only censored frames are shown. Sinks accept only CensoredFrame, which only the renderer creates. An error shows the last censored frame or black.
2. Fail closed. New subjects, ID-less detections, broken masks and undecided subjects are concealed.
3. Only lifting a conceal needs confidence. Concealing is the default and is instant; uncensoring may fade out.
4. Downgrade fast, upgrade slow. A failed re-check or a suspected ID swap conceals at once; passing needs several agreeing checks.
5. Crossings conceal (new). Overlapping subjects are both concealed until each is re-verified.
6. Ambiguous faces give nothing (new). A face not clearly owned by one subject's mask gives no identity evidence, no gender evidence and no eye slit.
7. Body-only evidence never uncensors by default (new). Only face-based evidence can fire a pass rule unless a later rule explicitly allows otherwise.
8. Time, not frames (new). Every temporal safety rule is measured in seconds on Frame.ts, so it behaves the same on every machine and in offline mode.
9. Privacy. Stranger embeddings live in memory only, for the life of their track; only enrolled family members are stored.
Time-based rules and starting values
Every "how long" rule is set in seconds in the shared config and compared against Frame.ts. In live mode Frame.ts is the capture clock; in offline mode it is the frame's position in the video (index / video_fps), so a 0.5 s hold means 0.5 s of video even when export runs slower than real time.
Setting
Starting value
Meaning
mask_hold_s
0.5 s
Keep drawing a lost subject's mask (shifted along its motion)
gap_reverify_s
0.15 s
A passed subject unseen for longer goes back to concealed until re-verified
track_buffer_s
1.5 s
How long the tracker keeps a lost ID
recheck_whitelist_s
1.0 s
Re-confirm a whitelisted subject
recheck_gender_s
2.0 s
Re-confirm a gender-passed subject
retry_unsure_s
0.3 s
Retry a subject whose evidence is still below threshold
fade_out_s
0.2 s
Fade when lifting a conceal; concealing is always instant
overlap_iou
0.1
Box overlap that counts as a crossing (not a time, listed for completeness)
All values are starting points to tune in the evaluation task. The tracker counts frames internally, so track_buffer_s is converted once at startup: frames = seconds × expected fps of that machine's config.
Roadmap
Forty tasks in five phases, with three gates; a task can start as soon as its prerequisites are done. The critical path runs T01 → T02 → T08 → T10 → T17 → T22 → T25 → T26; everything else can run beside it. Phases 1–4 are the hackathon; phase 5 is after it.
Each gate must pass before the next phase's gated tasks start; tasks without a gate prerequisite can start earlier.
Phase 1: Foundations
ID
Task
Prerequisites
Done when
T01
Environments on both Linux laptops, pinned per-machine requirements
—
Both import every library; GPU / iGPU / NPU visible
T02
Contracts in contracts.py: Frame, CensoredFrame, Track, FaceObservation, Evidence, Action, Decision, TrackState; Detector, IdentityModule, Renderer, Source, Sink interfaces
T01
Agreed and committed; stubs compile
T03
Config system: shared + per-machine YAML, settings in seconds, command-line overrides
T01
Both machine configs load; an override changes one value
T04
Record test clips (README §13.1 scenarios), with consent of everyone filmed
—
Clips for every scenario on disk, gitignored
Phase 2: Conceal-everyone pipeline
ID
Task
Prerequisites
Done when
T05
Sources: CameraSource, FileSource (realtime / offline); Frame.ts from clock live, from video time offline
T02, T03
Webcam and file both produce Frames with correct timestamps
T06
Newest-frame grabber
T05
Slow processing drops frames, latency stays bounded
T07
Sinks: DisplaySink, FileSink with writer thread (blocking offline, drop-oldest live), audio remux
T02
Export keeps audio; a slow encoder never freezes the display
T08
Detector interface + Ultralytics YOLO26-seg adapter: tracking, ID-less detections, track_buffer_s conversion, ByteTrack vs BoT-SORT choice
T02, T03
Tracks with stable IDs on test clips; ID-less detections returned
T09
Main loop with fail-safe and stub identity (conceal all)
T06, T07, T08
An injected error never shows a raw frame
T10
Mask pipeline: merge at low resolution, remove letterbox, resize once, dilate
T08
Masks line up with people at full resolution
T11
Renderer: conceal styles (niqab full cover, blur, pixelate, solid)
T10
Every style renders; niqab covers hair and edges
T12
Mask hold and instant conceal / fade-out, time-based
T11
No flicker on short occlusions at 15 and 30 fps alike
T13
Debug overlay, fps and latency measurement
T09
Overlay shows IDs, state, fps, latency; refused on the glasses output
T14
Hardware benchmark (Intel CPU / iGPU / NPU × imgsz × precision; NVIDIA TensorRT FP16) → per-machine configs
T08, T13
Device and imgsz chosen per machine and written into configs
Gate M1: everyone is concealed live on both laptops at ≥ 30 fps (NVIDIA) and ≥ 15 fps (Intel), and the fail-safe test passes.
Phase 3: Identity and policy
ID
Task
Prerequisites
Done when
T15
Identity component skeleton: worker thread, priority queue (new before re-checks), pending flags, sync mode, dead-track cleanup
T02
Returns conceal for all; never blocks the main loop
T16
Face analysis wrapper (InsightFace buffalo_l / buffalo_s) → face observations
T01
Box, landmarks, quality, embedding, gender, age per face
T17
Face-to-subject assignment through the mask, ambiguity rule, minimum face size
T10, T16
Overlapping people never get each other's face
T18
Gender component: face estimator, log-odds accumulation, decider with a rule list (face evidence required by default)
T16
p_female per track; adding a rule needs no other change
T19
Age estimator + child rule (confidently ≤ 10 → pass)
T16, T18
Uncertain age falls through to conceal
T20
Enrollment tool with quality checks, per-model embedding storage
T16
Rejected photos give a reason; embeddings cached per model
T21
Whitelist matcher with accept / reject thresholds per model
T20
Zero strangers accepted on test clips
T22
Policy engine and state machine: ordered rules, votes, re-check timers
T15, T17, T18, T19, T21
Rule order from the policy section holds on test clips
T23
Crossing and gap rules
T22
A swapped ID is concealed in the same frame the overlap starts
Gate M2: on all test clips, only non-whitelisted females over 10 are concealed (niqab), with zero leaked frames.
Phase 4: Evaluation and demo
ID
Task
Prerequisites
Done when
T24
Crop dump and per-track labeling tool
T04, T08
CSV labels per track for every clip
T25
Evaluation harness: offline sync mode, README §13.2 metrics, one test per safety invariant
T22, T24
One command prints all metrics for a clip set
T26
Threshold tuning per model and machine
T25
Metrics meet README §13.2 targets
T27
Fake-camera (v4l2loopback) and phone (scrcpy) source checks
T06
Both run through the real camera path
T28
Movie mode end to end
T07, T22
A clip exports with censoring and original audio
T29
10-minute thermal run on the Intel laptop
T14, Gate M2
Still ≥ 15 fps after 10 minutes
T30
Backup demo video and slides
T28, T29
Recorded and rehearsed
Gate M3: hackathon demo ready, live and from the backup video.
Phase 5: End goals and product
ID
Task
Prerequisites
Done when
T31
Per-frame face locator on head regions (SCRFD-500M), using T17's assignment
T17, Gate M2
Landmarks every frame for concealed subjects, ≤ 3 ms each
T32
Eye slit in the niqab
T31
Slit follows, scales, tilts; closes when the face is lost
T33
Automatic hardware detection, benchmark and cached profile
T03, T14
A new machine picks its profile without a hand-written config
T34
Adaptive quality controller
T13, T33
Over-budget runs recover by lowering quality, with hysteresis
T35
ONNX Runtime / OpenVINO detector backend behind the same interface
T08, T14
Swappable by config; faster where profiling showed overhead
T36
Two-pass offline runner: gap filling, backward extension, retroactive identity
T25, T28
Fewer misses than single pass on the evaluation set
T37
Decorate action: asset overlays anchored on landmarks
T31
Any asset can be attached to a rule
T38
Opt-in body-evidence rules
T18, T25
Enabled only if evaluation shows zero added leaks
T39
More censor domains as plugins (alcohol, scenes)
Gate M2
Same detect → decide → render interface
T40
More viewer profiles (female viewer, conceal for men)
T22
New profile = new rules, no code change
Caveats and open items
These are known weak spots of the merged design; each is owned by the task named, so none is forgotten.
Caveat
Effect
Handled in
Masks are low resolution; small faces cover few mask pixels and hair falls partly outside the mask
Face-to-subject assignment is unreliable for distant faces
T17: minimum face size, dilated mask, ambiguous → no evidence
Body-only evidence is off by default
Men seen from behind or far away stay concealed (false alarms, never leaks)
T38 may add a body rule once evaluation proves it safe
Age estimates are unreliable for children
Some young girls stay concealed; the error is always toward concealing
T19 threshold tuned in T26
Crossing rule conceals both subjects
More blur in crowds, brief blur on family members walking past others
Accepted; T26 tunes overlap_iou
Ultralytics adds Python pre/post-processing around the model
A few ms per frame (estimate); matters most on the Intel laptop
T35 swaps the backend if profiling (T14) shows it matters
TensorRT engines are tied to GPU, driver and TensorRT version
Rebuild per machine, minutes each
T14
Tracker counts frames internally
track_buffer_s must be converted per machine
T08
Licenses: Ultralytics AGPL-3.0, InsightFace models non-commercial
Fine for the hackathon; blocks a commercial product as is
Product phase decision
Still open, to be decided by the task named:
[ ] ByteTrack or BoT-SORT (T08, on head-mounted and handheld footage)
[ ] Intel laptop: integrated GPU or NPU for YOLO, and imgsz (T14)
[ ] All thresholds: gender, age, whitelist accept/reject, dilation size (T26)
[ ] Future target hardware (headset, phone + glasses, mini-PC): after phase 2