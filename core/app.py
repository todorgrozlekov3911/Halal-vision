
from __future__ import annotations

import argparse
import sys

import yaml

from .config_table import ConfigError, available_machines, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="halal-vision")

    #1. Default launch of the configuration shared + override based on hardware
    parser.add_argument(
        "--machine",
        required=True,
        choices=available_machines(),
        help="which per-machine yaml to layer over shared.yaml",
    )
    
    #2. --set functionality for run-time overriding values in the configuration docs without chancing the static version of the existing files
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help="override one setting (repeatable), e.g. --set detector.imgsz=320",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.machine, args.sets)
    except ConfigError as exc:
        parser.exit(2, f"config error: {exc}\n")

    yaml.safe_dump(cfg.as_dict(), sys.stdout, sort_keys=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())