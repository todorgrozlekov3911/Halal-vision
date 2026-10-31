"""Detection viewer: play a clip with each person's outline, box and track ID drawn on.

    python -m core.detect.detect_check clips/berlin_people_crossing.mp4 --machine intel
    python -m core.detect.detect_check clips/berlin_people_crossing.mp4 --machine intel --mode realtime
    python -m core.detect.detect_check clips/berlin_people_crossing.mp4 --machine intel \\
        --set detector.tracker=botsort --out out/detect_botsort.mp4

Colored outline + "#7 0.63"  tracked person, ID 7, detector confidence 0.63
Red outline + "no ID 0.31"   detected but not (yet) tracked: concealed downstream

Dev tool only, like io_check: shows people uncensored, under a red banner.
"""
from __future__ import annotations

import argparse
import colorsys
import sys
import time
from pathlib import Path

import cv2

from core.config_table import ConfigError, available_machines, load_config
from core.contracts import CensoredFrame, Detections
from core.detect.detector import UltralyticsDetector
from core.detect.masks import outlines_in_frame
from core.video.io_check import draw_header
from core.video.pacing import Pacer
from core.video.sinks import DisplaySink, FileSink
from core.video.sources import MODES, FileSource

NO_ID = (0, 0, 255)


def color_of(track_id: int) -> tuple[int, int, int]:
    """A stable, bright BGR color per ID (golden-ratio hue steps)."""
    r, g, b = colorsys.hsv_to_rgb(track_id * 0.618033988749895 % 1.0, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_detections(img, dets: Detections) -> None:
    s = img.shape[0] / 1080
    for t in dets.tracks:
        color = NO_ID if t.track_id is None else color_of(t.track_id)
        cv2.drawContours(img, outlines_in_frame(dets.mask_of(t), dets.geometry), -1, color, max(2, int(3 * s)))
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        label = f"no ID {t.score:.2f}" if t.track_id is None else f"#{t.track_id} {t.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, max(1, int(2 * s)))
        cv2.rectangle(img, (x1, y1 - th - int(10 * s)), (x1 + tw + int(8 * s), y1), color, -1)
        cv2.putText(img, label, (x1 + int(4 * s), y1 - int(5 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s,
                    (0, 0, 0), max(1, int(2 * s)), cv2.LINE_AA)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="detect_check", description=__doc__.split("\n")[0])
    p.add_argument("clip", type=Path)
    p.add_argument("--machine", required=True, choices=available_machines())
    p.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY.PATH=VALUE")
    p.add_argument("--mode", choices=MODES, default="offline")
    p.add_argument("--out", type=Path, help="also write an H.264 file")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--max-seconds", type=float, default=0, help="stop after this much wall-clock time")
    p.add_argument("--unpaced", action="store_true", help="realtime mode: process as fast as possible "
                   "instead of at expected_fps from the config")
    args = p.parse_args(argv)

    try:
        cfg = load_config(args.machine, args.sets)
    except ConfigError as exc:
        p.exit(2, f"config error: {exc}\n")

    source = FileSource(args.clip, mode=args.mode)
    # The tracker counts processed frames: every frame offline, about expected_fps live.
    processed_fps = float(cfg.expected_fps) if source.is_live else source.fps
    detector = UltralyticsDetector(cfg, fps=processed_fps)
    pacer = Pacer(processed_fps) if source.is_live and not args.unpaced else None

    sinks, display = [], None
    if not args.no_display:
        display = DisplaySink(window=f"detect_check - {args.clip.name} - {cfg.detector.tracker}")
        sinks.append(display)
    if args.out:
        offline = args.mode == "offline"
        sinks.append(FileSink(args.out, fps=source.fps, audio_from=args.clip if offline else None, block=offline))
    if not sinks:
        print("no output: timing only")

    d = cfg.detector
    print(f"{args.clip.name} | {d.model} imgsz={d.imgsz} on {d.device} | tracker={d.tracker} "
          f"track_buffer={detector.track_buffer_frames} frames ({cfg.timing.track_buffer_seconds} s at {processed_fps:.1f} fps)")

    ids_seen: set[int] = set()
    life: dict[int, list[float]] = {}  # track_id -> [first ts, last ts]
    ts_first = ts_last = 0.0
    det_ms = loop_fps = 0.0
    n, t_prev = 0, time.perf_counter()
    t_start = t_prev
    prof: dict[str, list[float]] = {}  # step -> ms per frame

    def lap(step: str, since: float) -> float:
        now = time.perf_counter()
        prof.setdefault(step, []).append(1000 * (now - since))
        return now

    try:
        while True:
            if pacer:
                pacer.wait()
            t_read = time.perf_counter()
            frame = source.read()
            if frame is None:
                break
            lap("read (decode + skip)", t_read)
            ts_first = frame.ts if n == 0 else ts_first
            ts_last = frame.ts
            t0 = time.perf_counter()
            dets = detector.detect(frame)
            t1 = lap("detect (total)", t0)
            for step, ms in detector.last_timing.items():
                prof.setdefault(f"  detect: {step}", []).append(ms)
            det_ms = 1000 * (t1 - t0) if n == 0 else 0.9 * det_ms + 100 * (t1 - t0)
            loop_fps = 0.9 * loop_fps + 0.1 / max(t1 - t_prev, 1e-6) if n else 0.0
            t_prev = t1
            ids_now = [t.track_id for t in dets.tracks if t.track_id is not None]
            ids_seen.update(ids_now)
            for tid in ids_now:
                life.setdefault(tid, [frame.ts, frame.ts])[1] = frame.ts

            t_draw = time.perf_counter()
            img = frame.image.copy()
            draw_detections(img, dets)
            draw_header(img, "DETECTION VIEW - NOT CENSORED", [
                f"frame {frame.idx}   ts {frame.ts:.2f} s   skipped {source.skipped}",
                f"{cfg.detector.tracker}   detect {det_ms:5.1f} ms   loop {loop_fps:4.1f} fps",
                f"people {len(dets.tracks)} (no ID {len(dets.tracks) - len(ids_now)})   IDs used so far {len(ids_seen)}",
            ])
            out = CensoredFrame(image=img, ts=frame.ts, idx=frame.idx)
            t_show = lap("draw (copy + outlines + text)", t_draw)
            for sink in sinks:
                sink.write(out)
            t_end = lap("output (window / file)", t_show)
            prof.setdefault("whole loop", []).append(1000 * (t_end - t_read))
            n += 1
            if ((display and display.quit_requested) or (args.max_frames and n >= args.max_frames)
                    or (args.max_seconds and time.perf_counter() - t_start >= args.max_seconds)):
                break
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        for sink in sinks:
            sink.close()

    took = time.perf_counter() - t_start
    if display and display.quit_requested:
        stop = f"window: {display.quit_reason}"
    elif source.end_reason:
        stop = f"source: {source.end_reason}"
    else:
        stop = "frame/time limit or Ctrl+C"
    print(f"stopped because: {stop}")
    paced = f", paced at {processed_fps:.0f} fps, {pacer.late} frames over budget" if pacer else ""
    print(f"{n} frames in {took:.1f} s ({n / took:.1f} fps), skipped {source.skipped}{paced}")
    if life:
        spans = sorted(last - first for first, last in life.values())
        video_s = max(ts_last - ts_first, 1e-6)
        print(f"IDs: {len(spans)} total, {len(spans) / video_s * 10:.1f} per 10 s of video | "
              f"median lifetime {spans[len(spans) // 2]:.2f} s | under 0.5 s: {sum(x < 0.5 for x in spans)} | "
              f"over 5 s: {sum(x >= 5 for x in spans)}")
    if prof:
        print("time per frame (median ms, after the first 5 frames):")
        for step, v in prof.items():
            v = sorted(v[5:] or v)
            print(f"  {step:34s} {v[len(v) // 2]:6.1f}")
    if args.out:
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
