"""I/O check: play a clip through FileSource -> DisplaySink / FileSink, with no detection.

    python -m core.video.io_check clips/berlin_people_crossing.mp4
    python -m core.video.io_check clips/berlin_people_crossing.mp4 --mode realtime
    python -m core.video.io_check clips/berlin_people_crossing.mp4 --out out/io_check.mp4 --no-display

Dev tool only. It is the one place outside the renderer that creates a CensoredFrame,
and it does so without censoring anything: every frame carries a red "NOT CENSORED"
banner. Remove it once the real pipeline (T09) can do the same job.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from core.contracts import CensoredFrame, Frame
from core.video.sinks import DisplaySink, FileSink
from core.video.sources import MODES, FileSource


def annotate(frame: Frame, lines: list[str]) -> CensoredFrame:
    img = frame.image.copy()  # Frame.image is read-only; draw on a copy
    h, w = img.shape[:2]
    s = h / 1080
    cv2.rectangle(img, (0, 0), (w, int(60 * s)), (0, 0, 200), -1)
    cv2.putText(img, "I/O CHECK - NOT CENSORED", (int(20 * s), int(44 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4 * s, (255, 255, 255), max(1, int(3 * s)), cv2.LINE_AA)
    for i, text in enumerate(lines):
        y = int((110 + 45 * i) * s)
        org = (int(20 * s), y)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1 * s, (0, 0, 0), max(1, int(6 * s)), cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1 * s, (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    return CensoredFrame(image=img, ts=frame.ts, idx=frame.idx)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="io_check", description=__doc__.split("\n")[0])
    p.add_argument("clip", type=Path)
    p.add_argument("--mode", choices=MODES, default="offline")
    p.add_argument("--out", type=Path, help="also write an H.264 file (original audio kept in offline mode)")
    p.add_argument("--no-display", action="store_true", help="no window, e.g. over SSH")
    p.add_argument("--max-frames", type=int, default=0, help="stop after this many frames (0 = whole clip)")
    args = p.parse_args(argv)

    source = FileSource(args.clip, mode=args.mode)
    sinks = []
    display = None
    if not args.no_display:
        display = DisplaySink(window=f"io_check - {args.clip.name}")
        sinks.append(display)
    if args.out:
        offline = args.mode == "offline"
        sinks.append(FileSink(args.out, fps=source.fps, audio_from=args.clip if offline else None, block=offline))
    if not sinks:
        p.error("nothing to output: drop --no-display or add --out")

    h, w = source.frame_shape
    print(f"{args.clip.name}: {w}x{h} @ {source.fps:.2f} fps, ~{source.frame_count} frames, mode={args.mode}")

    t_start = time.perf_counter()
    t_prev, loop_fps, n = t_start, 0.0, 0
    try:
        while (frame := source.read()) is not None:
            now = time.perf_counter()
            dt, t_prev = now - t_prev, now
            loop_fps = 0.9 * loop_fps + 0.1 * (1 / dt) if dt > 0 and n else loop_fps
            lines = [
                f"frame {frame.idx}   ts {frame.ts:.3f} s   ({'capture clock' if source.is_live else 'video time'})",
                f"loop {loop_fps:5.1f} fps   video {source.fps:.2f} fps   skipped {source.skipped}",
            ]
            out = annotate(frame, lines)
            for sink in sinks:
                sink.write(out)
            n += 1
            if (display and display.quit_requested) or (args.max_frames and n >= args.max_frames):
                break
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        for sink in sinks:
            sink.close()

    took = time.perf_counter() - t_start
    print(f"shown {n} frames in {took:.1f} s ({n / took:.1f} fps), skipped {source.skipped}")
    if args.out:
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
