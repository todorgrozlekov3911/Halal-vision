"""Side-by-side player: a clip played as if it were a live camera, censored next to it (T10/T11 check).

    python -m core.player --machine intel
    python -m core.player --machine intel clips/berlin_people_crossing.mp4

Left: the input. Right: the output, processed live. Each processed frame is the one
"on air" at that moment, and frames the pipeline is too slow for are skipped, exactly
like a camera. Seeking resets the tracker and renderer, like switching cameras.

Keys: Space play / pause, Left / Right 5 s back / forward, q or Esc quit.

Detector: radio buttons switch the person model live (YOLO26n, YOLO26s, RF-DETR-Seg
nano), to compare them on the same clip. Each loads the first time it is picked (RF-DETR:
download + OpenVINO conversion on Intel, about a minute) and is then kept, so switching
back is instant. The status line shows which one is running and its time per frame.

Dev tool: "No censor" shows people uncensored. Identity is the conceal-everyone stub
until T15 / T22, so everyone the detector finds is covered.
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, ttk

import cv2
from PIL import Image, ImageTk

from core.config_table import ConfigError, available_machines, load_config
from core.contracts import Action, ActionKind, ConcealAllIdentity, Detections, Frame
from core.detect.detector import make_detector
from core.render.renderer import STYLES, StyleRenderer

CLIPS = Path(__file__).resolve().parents[1] / "clips"
SEEK_STEP_S = 5.0
NONE = "none"
MODES = (NONE, *STYLES)
MODE_LABELS = {NONE: "No censor", "niqab": "Niqab", "blur": "Blur", "pixelate": "Pixelate", "solid": "Solid"}
DETECTORS = {                    # detector.model value -> button label
    "yolo26n-seg.pt": "YOLO26n",
    "yolo26s-seg.pt": "YOLO26s",
    "rf-detr-seg-nano": "RF-DETR-Seg N",
}
VIDEO_TYPES = [("Video", "*.mp4 *.mkv *.mov *.avi *.webm"), ("All files", "*")]
START_VIDEO_AREA = (1100, 340)   # px at start: two 16:9 panels of about 546x307; resize the window for more
PANEL_TITLE_PX = 26              # the "Input" / "Output" label above each video


# -- the clip, read like a camera ----------------------------------------------------

class Playback:
    """A video file read like a camera (the frame on air now, missed ones skipped), plus pause and seek."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise OSError(f"OpenCV cannot open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.source_id = f"file:{path.name}"
        self.playing = False
        self.at_end = False
        self.skipped = 0
        self.next_idx = 0                  # the frame cap.read() returns next
        self._anchor_idx = 0               # frame on air at _anchor_t
        self._anchor_t = time.perf_counter()

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps

    def on_air(self) -> int:
        if not self.playing:
            return self._anchor_idx
        return self._anchor_idx + int((time.perf_counter() - self._anchor_t) * self.fps)

    def play(self) -> None:
        if not self.playing:
            if self.at_end:                             # at the end: start over
                self.seek(0)
            self._anchor_t = time.perf_counter()
            self.playing = True

    def pause(self) -> None:
        if self.playing:
            self.playing = False
            self._anchor_idx = max(0, self.next_idx - 1)  # stay on the frame last shown

    def seek(self, idx: int) -> None:
        idx = max(0, min(int(idx), self.frame_count - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        self.next_idx = self._anchor_idx = idx
        self.at_end = False
        self._anchor_t = time.perf_counter()

    def read_on_air(self) -> tuple[str, Frame | None]:
        """("frame", f) | ("wait", None) when the next frame is not due yet | ("end", None)."""
        target = self.on_air()
        if target < self.next_idx:
            return "wait", None
        while self.next_idx < target:                   # too slow: skip what was missed, like a camera
            if not self.cap.grab():
                self.at_end = True
                return "end", None
            self.next_idx += 1
            self.skipped += 1
        return self._read()

    def read_current(self) -> tuple[str, Frame | None]:
        """Paused: the frame at the current position."""
        if self.next_idx != self._anchor_idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._anchor_idx)
            self.next_idx = self._anchor_idx
        return self._read()

    def seconds_until_next(self) -> float:
        due = self._anchor_t + (self.next_idx - self._anchor_idx) / self.fps
        return due - time.perf_counter()

    def _read(self) -> tuple[str, Frame | None]:
        ok, image = self.cap.read()                     # a fresh array per frame (rule 3)
        if not ok:
            self.at_end = True
            return "end", None
        idx, self.next_idx = self.next_idx, self.next_idx + 1
        return "frame", Frame(image=image, ts=time.perf_counter(), idx=idx, source_id=self.source_id)

    def close(self) -> None:
        self.cap.release()


# -- processing thread ---------------------------------------------------------------

@dataclass
class Shown:
    left: Image.Image | None
    right: Image.Image
    mode: str
    model: str
    position_s: float
    duration_s: float
    playing: bool
    status: str


class Worker(threading.Thread):
    """Reads, detects, renders and prepares both panels; the Tk thread only displays them."""

    def __init__(self, cfg) -> None:
        super().__init__(name="player-worker", daemon=True)
        self.cfg = cfg
        self.commands: queue.Queue[tuple] = queue.Queue()
        self.mode = str(cfg.policy.conceal_style)       # changed by the radio buttons
        self.show_input = True                           # changed by the checkbox
        self.model = str(cfg.detector.model)             # the detector running now
        self.model_request = self.model                  # set by the radio buttons; loaded by this thread
        self.labels = {**DETECTORS, self.model: DETECTORS.get(self.model, self.model)}
        self.loading = ""                                # label of the detector being loaded, if any
        self._detectors: dict = {}                       # model -> loaded detector, kept for quick switching
        self.panel_size = (640, 360)                     # max (w, h) per panel, set by the UI
        self.message = "loading the detector..."
        self.error = ""
        self._latest: Shown | None = None
        self._lock = threading.Lock()
        self.playback: Playback | None = None
        self._cached: tuple[Frame, Detections | None] | None = None   # last frame, for re-rendering while paused
        self._last_mode = self.mode
        self._fps = 0.0
        self._t_last = 0.0
        self._det_ms = self._render_ms = 0.0
        self._people = 0

    def send(self, *command) -> None:
        self.commands.put(command)

    def take(self) -> Shown | None:
        with self._lock:
            shown, self._latest = self._latest, None
            return shown

    # -- loop -------------------------------------------------------------------

    def run(self) -> None:
        try:
            self.detector = self._load_detector(self.model)
            self.renderer = StyleRenderer(self.cfg)
            self.message = "open a video"
        except Exception as exc:                          # noqa: BLE001 - shown in the window
            self.error = f"detector failed to load: {exc}"
            return
        while True:
            try:
                if not self._handle_commands():
                    return
                self._step()
            except Exception as exc:                      # noqa: BLE001 - keep the player alive
                self.error = f"{type(exc).__name__}: {exc}"
                if self.playback:
                    self.playback.pause()

    def _load_detector(self, model: str):
        if model not in self._detectors:
            self.loading = self.labels.get(model, model)
            try:
                self._detectors[model] = make_detector(self.cfg, fps=float(self.cfg.expected_fps), model=model)
            finally:
                self.loading = ""
        return self._detectors[model]

    def _switch_detector(self) -> None:
        model = self.model_request
        try:
            detector = self._load_detector(model)
        except Exception as exc:                          # noqa: BLE001 - shown; the old detector stays
            self.error = f"{self.labels.get(model, model)} failed to load: {type(exc).__name__}: {exc}"
            self.model_request = self.model               # the window puts the radio button back
            return
        self.detector, self.model = detector, model
        frame = self._cached[0] if self._cached else None
        self._reset_pipeline()                            # fresh tracks: IDs from one model mean nothing to another
        if frame is not None and self.playback and not self.playback.playing:
            self._show("frame", frame)                    # paused: show the new model on the same frame

    def _handle_commands(self) -> bool:
        idle = self.playback is None or (not self.playback.playing and self.mode == self._last_mode)
        idle = idle and self.model_request == self.model
        try:
            command = self.commands.get(timeout=0.05) if idle else self.commands.get_nowait()
        except queue.Empty:
            return True
        while True:
            name, *args = command
            if name == "quit":
                if self.playback:
                    self.playback.close()
                return False
            self.error = ""
            if name == "open":
                self._open(Path(args[0]))
            elif self.playback is not None:
                pb = self.playback
                if name == "toggle":
                    if pb.playing:
                        pb.pause()
                        self._publish_again()
                    else:
                        pb.play()
                elif name == "redraw":
                    if not pb.playing and self._cached:
                        self._show("frame", *self._cached)
                elif name in ("seek", "seek_rel"):
                    seconds = args[0] if name == "seek" else pb.on_air() / pb.fps + args[0]
                    self._reset_pipeline()
                    pb.seek(round(seconds * pb.fps))
                    if not pb.playing:
                        self._show(*pb.read_current())
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                return True

    def _step(self) -> None:
        if self.model_request != self.model:
            self._switch_detector()
        pb = self.playback
        if pb is None:
            return
        if pb.playing:
            status, frame = pb.read_on_air()
            if status == "wait":
                time.sleep(max(0.0, min(pb.seconds_until_next(), 0.01)))
                return
            if status == "end":
                pb.pause()
                self._publish_again()
                return
            self._show(status, frame)
        elif self.mode != self._last_mode and self._cached:   # paused, style changed: redo the shown frame
            frame, dets = self._cached
            self._show("frame", frame, dets)

    def _open(self, path: Path) -> None:
        if self.playback:
            self.playback.close()
        self.playback = Playback(path)
        self._reset_pipeline()
        self.playback.skipped = 0
        self._show(*self.playback.read_current())
        self.playback.play()

    def _reset_pipeline(self) -> None:
        self.detector.reset()
        self.renderer.reset()
        self._cached = None
        self._t_last = 0.0

    # -- one frame -------------------------------------------------------------

    def _show(self, status: str, frame: Frame | None, dets: Detections | None = None) -> None:
        if status != "frame" or frame is None:
            return
        mode = self.mode
        if mode != NONE and self._last_mode == NONE and dets is None:
            self.detector.reset()                      # the tracker saw nothing while uncensored
        self._last_mode = mode

        t0 = time.perf_counter()
        if mode == NONE:
            out, dets = frame.image, None
            self._det_ms = self._render_ms = 0.0
            self._people = 0
        else:
            if dets is None:
                dets = self.detector.detect(frame)
                self._det_ms = 1000 * (time.perf_counter() - t0)
            t1 = time.perf_counter()
            self.renderer.set_default_style(mode)      # people without a decision (ID-less) follow the buttons too
            identity = ConcealAllIdentity(Action(ActionKind.CONCEAL, style=mode))
            out = self.renderer.render(frame, dets, identity.update(frame, dets)).image
            self._render_ms = 1000 * (time.perf_counter() - t1)
            self._people = len(dets.tracks)
        self._cached = (frame, dets)

        now = time.perf_counter()
        if self._t_last and self.playback and self.playback.playing:
            rate = 1.0 / max(now - self._t_last, 1e-6)
            self._fps = rate if not self._fps else 0.9 * self._fps + 0.1 * rate
        self._t_last = now

        max_w, max_h = self.panel_size
        left = _panel_image(frame.image, max_w, max_h) if self.show_input else None
        self._publish(left, _panel_image(out, max_w, max_h), frame.idx)

    def _publish(self, left: Image.Image | None, right: Image.Image, idx: int) -> None:
        pb = self.playback
        on = getattr(self.detector, "runs_on", "")
        status = (f"{self.labels.get(self.model, self.model)} on {on}   {self._fps:4.1f} fps processed of {pb.fps:.0f}   skipped {pb.skipped}   "
                  f"detect {self._det_ms:5.1f} ms   mask + render {self._render_ms:4.1f} ms   "
                  f"people {self._people}")
        shown = Shown(left, right, self._last_mode, self.model, idx / pb.fps, pb.duration, pb.playing, status)
        with self._lock:
            self._latest = shown
        self._last_shown = shown

    def _publish_again(self) -> None:
        """Re-send the last panels with the play state changed (play button, time)."""
        last = getattr(self, "_last_shown", None)
        if last and self.playback:
            last.playing = self.playback.playing
            with self._lock:
                self._latest = last


def _panel_image(image, max_w: int, max_h: int) -> Image.Image:
    h, w = image.shape[:2]
    k = min(max_w / w, max_h / h, 1.0)
    if k < 1.0:
        image = cv2.resize(image, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# -- window ------------------------------------------------------------------------

class PlayerWindow:
    def __init__(self, root: tk.Tk, worker: Worker, machine: str) -> None:
        self.root, self.worker = root, worker
        root.title(f"Halal-Vision player ({machine})")
        root.configure(padx=8, pady=8)
        self._dragging = False
        self._duration = 0.0
        self._photos: dict[str, ImageTk.PhotoImage] = {}

        top = ttk.Frame(root)
        top.pack(fill="x")
        ttk.Button(top, text="Open video...", command=self.open_dialog).pack(side="left")
        self.file_label = ttk.Label(top, text="no video", width=34)
        self.file_label.pack(side="left", padx=(8, 16))
        ttk.Label(top, text="Output:").pack(side="left")
        self.mode_var = tk.StringVar(value=worker.mode)
        for mode in MODES:
            ttk.Radiobutton(top, text=MODE_LABELS[mode], value=mode, variable=self.mode_var,
                            command=self.on_mode).pack(side="left", padx=4)
        self.show_input_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Show input video", variable=self.show_input_var,
                        command=self.on_show_input).pack(side="right")

        models = ttk.Frame(root)
        models.pack(fill="x", pady=(6, 0))
        ttk.Label(models, text="Detector:").pack(side="left")
        self.model_var = tk.StringVar(value=worker.model)
        for model, label in worker.labels.items():
            ttk.Radiobutton(models, text=label, value=model, variable=self.model_var,
                            command=self.on_model).pack(side="left", padx=4)

        # The video area takes whatever size the window has (resize or maximize the window to
        # make the videos bigger); the images never push the window bigger themselves.
        videos = ttk.Frame(root, width=START_VIDEO_AREA[0], height=START_VIDEO_AREA[1])
        videos.pack(fill="both", expand=True, pady=8)
        videos.grid_propagate(False)
        videos.columnconfigure(0, weight=1, uniform="panels")
        videos.columnconfigure(1, weight=1, uniform="panels")
        videos.rowconfigure(0, weight=1)
        videos.bind("<Configure>", lambda e: self._schedule_resize())
        self.videos = videos
        self._resize_job: str | None = None
        self.left_box = ttk.Frame(videos)
        self.left_box.grid(row=0, column=0, padx=(0, 4), sticky="n")
        ttk.Label(self.left_box, text="Input").pack(anchor="w")
        self.left_img = ttk.Label(self.left_box)
        self.left_img.pack()
        right_box = ttk.Frame(videos)
        right_box.grid(row=0, column=1, padx=(4, 0), sticky="n")
        self.right_box = right_box
        self.right_title = ttk.Label(right_box, text="Output")
        self.right_title.pack(anchor="w")
        self.right_img = ttk.Label(right_box)
        self.right_img.pack()

        controls = ttk.Frame(root)
        controls.pack(fill="x")
        ttk.Button(controls, text=f"<< {SEEK_STEP_S:.0f} s",
                   command=lambda: worker.send("seek_rel", -SEEK_STEP_S)).pack(side="left")
        self.play_button = ttk.Button(controls, text="Play", width=7, command=lambda: worker.send("toggle"))
        self.play_button.pack(side="left", padx=4)
        ttk.Button(controls, text=f"{SEEK_STEP_S:.0f} s >>",
                   command=lambda: worker.send("seek_rel", SEEK_STEP_S)).pack(side="left")
        self.time_label = ttk.Label(controls, text="0:00 / 0:00", width=13, anchor="e")
        self.time_label.pack(side="right")
        self.scale = ttk.Scale(controls, from_=0, to=1, orient="horizontal")
        self.scale.pack(side="left", fill="x", expand=True, padx=8)
        self.scale.bind("<ButtonPress-1>", self.on_scrub)
        self.scale.bind("<B1-Motion>", self.on_scrub)
        self.scale.bind("<ButtonRelease-1>", self.on_scrub_release)

        self.status = ttk.Label(root, text=worker.message, anchor="w")
        self.status.pack(fill="x", pady=(6, 0))

        root.bind("<space>", lambda e: worker.send("toggle"))
        root.bind("<Left>", lambda e: worker.send("seek_rel", -SEEK_STEP_S))
        root.bind("<Right>", lambda e: worker.send("seek_rel", SEEK_STEP_S))
        for key in ("q", "<Escape>"):
            root.bind(key, lambda e: self.quit())
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self._update_panel_size()
        root.after(15, self.poll)

    # -- user actions -------------------------------------------------------------

    def open_dialog(self) -> None:
        path = filedialog.askopenfilename(parent=self.root, initialdir=CLIPS if CLIPS.is_dir() else Path.cwd(),
                                          filetypes=VIDEO_TYPES)
        if path:
            self.open(Path(path))

    def open(self, path: Path) -> None:
        self.file_label.configure(text=path.name)
        self.worker.send("open", str(path))

    def on_mode(self) -> None:
        self.worker.mode = self.mode_var.get()

    def on_model(self) -> None:
        self.worker.model_request = self.model_var.get()

    def on_show_input(self) -> None:
        show = self.show_input_var.get()
        self.worker.show_input = show
        if show:
            self.right_box.grid(row=0, column=1, columnspan=1, padx=(4, 0))
            self.left_box.grid()
        else:
            self.left_box.grid_remove()
            self.left_img.configure(image="")
            self.right_box.grid(row=0, column=0, columnspan=2, padx=0)
        self._update_panel_size()

    def on_scrub(self, event) -> str:
        self._dragging = True
        fraction = min(max(event.x / max(self.scale.winfo_width(), 1), 0.0), 1.0)
        self.scale.set(fraction * self._duration)
        return "break"

    def on_scrub_release(self, event) -> str:
        self.on_scrub(event)
        self._dragging = False
        self.worker.send("seek", float(self.scale.get()))
        return "break"

    def quit(self) -> None:
        self.worker.send("quit")
        self.root.destroy()

    # -- display ------------------------------------------------------------------

    def _schedule_resize(self) -> None:
        """Window resizing sends a burst of events: act once it settles."""
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(120, self._update_panel_size)

    def _update_panel_size(self) -> None:
        self._resize_job = None
        w, h = self.videos.winfo_width(), self.videos.winfo_height()
        if w <= 1 or h <= 1:                      # not drawn yet
            w, h = START_VIDEO_AREA
        h -= PANEL_TITLE_PX
        size = ((w - 8) // 2, h) if self.show_input_var.get() else (w, h)
        size = (max(size[0], 160), max(size[1], 90))
        if size != self.worker.panel_size:
            self.worker.panel_size = size
            self.worker.send("redraw")            # paused: show the current frame at the new size

    def poll(self) -> None:
        shown = self.worker.take()
        if shown:
            if shown.left is not None and self.show_input_var.get():
                self._set_image("left", self.left_img, shown.left)
            self._set_image("right", self.right_img, shown.right)
            title = ("Output: NOT CENSORED" if shown.mode == NONE
                     else f"Output: {MODE_LABELS[shown.mode].lower()}, {self.worker.labels.get(shown.model, shown.model)} "
                          "(everyone detected is covered: identity stub)")
            self.right_title.configure(text=title)
            self.play_button.configure(text="Pause" if shown.playing else "Play")
            self._duration = shown.duration_s
            self.scale.configure(to=max(shown.duration_s, 0.001))
            if not self._dragging:
                self.scale.set(shown.position_s)
            self.time_label.configure(text=f"{_clock(shown.position_s)} / {_clock(shown.duration_s)}")
            fallback = "unavailable" in shown.status          # e.g. NPU configured, but no render group
            self.status.configure(text=shown.status, foreground="#b35900" if fallback else "")
        if self.model_var.get() != self.worker.model_request:
            self.model_var.set(self.worker.model_request)   # a detector failed to load: back to the running one
        if self.worker.error:
            self.status.configure(text=self.worker.error, foreground="red")
        elif self.worker.loading:
            self.status.configure(text=f"loading {self.worker.loading}... (the first time: download, and on "
                                       "Intel conversion for OpenVINO; can take a minute)", foreground="")
        elif not self.worker.playback:
            self.status.configure(text=self.worker.message)
        if not self.worker.is_alive() and not self.worker.error:
            return
        self.root.after(15, self.poll)

    def _set_image(self, key: str, label: ttk.Label, image: Image.Image) -> None:
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo)
        self._photos[key] = photo                  # Tk shows nothing if the Python object is collected


def _clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="player", description=__doc__.split("\n")[0])
    p.add_argument("clip", type=Path, nargs="?", help="video to open at start (or use Open video...)")
    p.add_argument("--machine", required=True, choices=available_machines())
    p.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY.PATH=VALUE")
    args = p.parse_args(argv)
    try:
        cfg = load_config(args.machine, args.sets)
    except ConfigError as exc:
        p.exit(2, f"config error: {exc}\n")
    if args.clip and not args.clip.is_file():
        p.exit(2, f"no such file: {args.clip}\n")

    worker = Worker(cfg)
    worker.start()
    root = tk.Tk()
    window = PlayerWindow(root, worker, args.machine)
    if args.clip:
        window.open(args.clip)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
