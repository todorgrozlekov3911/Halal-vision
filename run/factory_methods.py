"""Factories (T09): parse CLI args + config into a composed Runner.

One registry per dimension that genuinely varies (sources, sinks); a plain
build_* function where only one variant exists (pacer, identity, fail-safe —
promote to a registry the day a second variant appears). Adding a sink or
source = write one builder function + one @register_* line; build_runner
never changes.

CLI says WHAT to run, config says HOW this machine runs, probed facts
(source.is_live, frame_shape, CUDA presence) say what is actually true.
build_runner returns a Runner or raises ConfigError before any frame flows.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core.config_table import CONFIG_DIR, ConfigError, ConfigTable
from core.contracts import ConcealAllIdentity, IdentityModule, Renderer, Sink, Source
from core.detect.detector import UltralyticsDetector
from core.eval.dump import DecisionDump
from core.render.overlay import DebugOverlay
from core.render.renderer import StyleRenderer
from core.video.grabber import Grabber
from core.video.meters import LoopMeters
from core.video.pacing import Pacer
from core.video.sinks import DisplaySink, FileSink
from core.video.sources import FileSource
from run.runner import FailSafe, Runner

log = logging.getLogger("halal_vision.build")


@dataclass(frozen=True)
class BuildContext:
    """Everything a builder may need, parsed/probed exactly once."""

    cfg: ConfigTable
    args: argparse.Namespace
    source: Source
    live: bool            # the pivot: source.is_live drives the whole wiring table


# -- sources -----------------------------------------------------------------

SOURCES: dict[str, Callable[[argparse.Namespace], Source]] = {}


def register_source(kind: str):
    def deco(fn: Callable[[argparse.Namespace], Source]):
        SOURCES[kind] = fn
        return fn
    return deco


@register_source("file")
def _file_source(args: argparse.Namespace) -> Source:
    return FileSource(args.clip, mode=args.mode)


# later: @register_source("camera") (CameraSource, T05 rest),
# @register_source("phone") (scrcpy, T27)


def build_source(args: argparse.Namespace) -> Source:
    if not args.clip:
        raise ConfigError("no input given (clip path)")
    return SOURCES["file"](args)


# -- sinks -------------------------------------------------------------------

SINKS: dict[str, Callable[[BuildContext], Sink | None]] = {}


def register_sink(name: str):
    def deco(fn: Callable[[BuildContext], Sink | None]):
        SINKS[name] = fn
        return fn
    return deco


@register_sink("display")
def _display_sink(ctx: BuildContext) -> Sink | None:
    if ctx.args.no_display:
        return None
    sink = DisplaySink(window=f"halal-vision - {Path(ctx.args.clip).name}",
                       fullscreen=ctx.args.fullscreen)
    sink.essential = True      # live: losing the display aborts the run
    return sink


@register_sink("recording")
def _recording_sink(ctx: BuildContext) -> Sink | None:
    if not ctx.args.out:
        return None
    sink = FileSink(ctx.args.out, fps=ctx.source.fps,
                    audio_from=None if ctx.live else ctx.args.clip,
                    block=not ctx.live)
    sink.essential = False     # live: losing the recording is logged, run continues
    return sink


# later: @register_sink("virtual-camera") (v4l2loopback, T27)


def build_sinks(ctx: BuildContext) -> list[Sink]:
    return [sink for build in SINKS.values() if (sink := build(ctx)) is not None]


# -- single-variant builders ---------------------------------------------------

class _DirectFrames:
    """Offline frame supply: every frame, no thread; Grabber-shaped interface."""

    def __init__(self, source: Source) -> None:
        self.source = source

    def get(self, timeout: float | None = None):
        return self.source.read()

    def close(self) -> None:
        self.source.close()


def build_frames(ctx: BuildContext):
    return Grabber(ctx.source) if ctx.live else _DirectFrames(ctx.source)


def build_pacer(ctx: BuildContext) -> Pacer | None:
    return Pacer(float(ctx.cfg.expected_fps)) if ctx.live else None


def build_detector(ctx: BuildContext) -> UltralyticsDetector:
    fps = float(ctx.cfg.expected_fps) if ctx.live else ctx.source.fps
    return UltralyticsDetector(ctx.cfg, fps=fps)


def build_identity(ctx: BuildContext) -> IdentityModule:
    return ConcealAllIdentity(sync=not ctx.live)     # stub until T15+


def build_renderer(ctx: BuildContext) -> Renderer:
    return StyleRenderer(ctx.cfg)


def build_meters(ctx: BuildContext) -> LoopMeters:
    return LoopMeters(live=ctx.live)     # always on: a subtraction per frame, and T14 needs it


def build_overlay(ctx: BuildContext) -> DebugOverlay | None:
    return DebugOverlay() if ctx.args.debug_overlay else None


def build_dump(ctx: BuildContext) -> DecisionDump | None:
    return DecisionDump(ctx.args.dump) if ctx.args.dump else None


def build_failsafe(ctx: BuildContext) -> FailSafe:
    fs = ctx.cfg["fail_state"]
    image = fs.get("pipeline_break_image")
    if image:
        # relative paths are relative to system_configurations/, not the CWD
        path = Path(image) if Path(image).is_absolute() else CONFIG_DIR / image
        if path.is_file():
            image = str(path)
        else:
            log.warning("fail_state.pipeline_break_image %s not found, card will be black", path)
            image = None
    return FailSafe(ctx.source.frame_shape,
                    hold_seconds=float(fs["hold_before_break_image"]),
                    exit_seconds=float(fs["abort_after"]),
                    image_path=image)


# -- validation + aggregation --------------------------------------------------

def _openvino_sees(wanted: str, available: list[str]) -> bool:
    """OpenVINO lists devices as NPU / GPU / GPU.0 / CPU; intel:gpu matches GPU and GPU.0."""
    return any(d == wanted or d.startswith(f"{wanted}.") for d in available)


def _validate(ctx: BuildContext, sinks: list[Sink]) -> None:
    """Cross-cutting checks: raise now, not at frame 1."""
    if not sinks:
        raise ConfigError("nothing to output: drop --no-display or add --out")
    if ctx.args.debug_overlay and ctx.args.fullscreen:
        raise ConfigError("the debug overlay is refused on the glasses output (fullscreen "
                          "display): internal state never reaches the glasses (README §6.4)")
    device = str(ctx.cfg.detector.device)
    if device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise ConfigError(f"config wants detector.device={device} but CUDA is not "
                              "available on this machine (driver? wrong --machine?)")
    elif device.startswith("intel"):
        # The device is a benchmark/config decision: a device this process cannot see is a
        # configuration error, never a silent fall-back to something slower.
        import openvino
        wanted = device.partition(":")[2].upper()
        available = list(openvino.Core().available_devices)
        if wanted and not _openvino_sees(wanted, available):
            hint = ("NPU driver loaded (/dev/accel0)? user in the render group?"
                    if wanted == "NPU" else "driver?")
            raise ConfigError(
                f"config wants detector.device={device} but OpenVINO sees only {available} "
                f"({hint}); pick an available device (e.g. --set detector.device=intel:gpu) "
                f"or fix the driver — this does not fall back silently")


def build_runner(cfg: ConfigTable, args: argparse.Namespace) -> Runner:
    source = build_source(args)
    try:
        ctx = BuildContext(cfg=cfg, args=args, source=source, live=source.is_live)
        sinks = build_sinks(ctx)
        _validate(ctx, sinks)
        return Runner(
            frames=build_frames(ctx),
            detector=build_detector(ctx),
            identity=build_identity(ctx),
            renderer=build_renderer(ctx),
            sinks=sinks,
            failsafe=build_failsafe(ctx),
            pacer=build_pacer(ctx),
            meters=build_meters(ctx),
            overlay=build_overlay(ctx),
            dump=build_dump(ctx),
            fatal_sink_errors=not ctx.live,
            max_frames=args.max_frames,
            max_seconds=args.max_seconds,
        )
    except BaseException:
        source.close()      # a half-built runner must not leak the capture
        raise
