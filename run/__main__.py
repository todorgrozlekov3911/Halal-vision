"""Entry point:  python -m run <clip> --machine nvidia [--mode realtime] [--out f.mp4]

Loads config, composes a Runner through the factories, runs it, prints RunStats.
Exit codes: 0 ok, 1 pipeline failure, 2 bad config/arguments.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from core.config_table import ConfigError, available_machines, load_config
from core.video.sources import MODES
from run.factory_methods import build_runner
from run.runner import PipelineFailure


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="run", description=__doc__.split("\n")[0])
    p.add_argument("clip", type=Path, help="input video (CameraSource comes later)")
    p.add_argument("--machine", required=True, choices=available_machines())
    p.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY.PATH=VALUE")
    p.add_argument("--mode", choices=MODES, default="offline")
    p.add_argument("--out", type=Path, help="also record to an H.264 file")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--fullscreen", action="store_true",
                   help="fullscreen display (the glasses output)")
    p.add_argument("--debug-overlay", action="store_true",
                   help="draw IDs, state, rule, fps and latency (dev window only; refused with --fullscreen)")
    p.add_argument("--dump", type=Path,
                   help="write per-frame decisions as JSON Lines, for evaluation")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--max-seconds", type=float, default=0.0)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        cfg = load_config(args.machine, args.sets)
        runner = build_runner(cfg, args)
    except ConfigError as exc:
        p.exit(2, f"config error: {exc}\n")

    try:
        stats = runner.run()
    except PipelineFailure as exc:
        print(f"pipeline failure: {exc}", file=sys.stderr)
        return 1
    print(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
