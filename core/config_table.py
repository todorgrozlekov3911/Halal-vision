"""Config system (T03).

Layers, later wins:
    system-configurations/shared.yaml  <  <machine>.yaml  <  --set key.path=value

load_config() returns a ConfigTable: a read-only, dict-like object holding the
merged result. Components read from it; nothing writes to it after startup.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterator

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "system-configurations"
SHARED = "shared"


ALLOWED_VALUES = {
    "detector.tracker": {"bytetrack", "botsort"},
    "policy.conceal_style": {"niqab", "blur", "pixelate", "solid"},
    "policy.viewer_profile": {"male"},
    "identity.face_model": {"buffalo_l", "buffalo_s"},
}


class ConfigError(Exception):
    ...


class ConfigTable(Mapping):
    """Read-only, dict-like view of the merged configuration.

    Nested values are reachable three ways:
        cfg["detector"]["imgsz"], cfg["detector.imgsz"], cfg.detector.imgsz
    """

    def __init__(self, data: dict[str, Any]):
        object.__setattr__(self, "_data", data)

    def __getitem__(self, key: str) -> Any:
        node: Any = self._data
        for part in key.split("."):
            if not (isinstance(node, dict) and part in node):
                raise KeyError(key)
            node = node[part]
        return ConfigTable(node) if isinstance(node, dict) else node

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("ConfigTable is read-only after load")

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"ConfigTable({self._data!r})"

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)


def available_machines(config_dir: Path = CONFIG_DIR) -> list[str]: #returns all supported machine architectures yamls | expludes the shared configuration
    return sorted(p.stem for p in config_dir.glob("*.yaml") if p.stem != SHARED)


def load_config(
    machine: str,
    sets: tuple[str, ...] | list[str] = (),
    config_dir: Path = CONFIG_DIR,
) -> ConfigTable:
    data = _load_yaml_basic(config_dir / f"{SHARED}.yaml")
    data = _hardware_override_merge(data, _load_yaml_basic(config_dir / f"{machine}.yaml"))
    for expr in sets:
        _apply_set(data, expr)
    _check_allowed(data)
    return ConfigTable(data)


def _load_yaml_basic(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    loaded = yaml.safe_load(path.read_text()) or {}
    if not loaded:
        raise ConfigError(f"{path} - trying to parse the project with an empty yaml in that path")
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a mapping at top level")
    return loaded


def _hardware_override_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _hardware_override_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_set(data: dict, expr: str) -> None:
    key, sep, raw = expr.partition("=")
    if not sep or not key:
        raise ConfigError(f"--set wants KEY.PATH=VALUE, got {expr!r}")
    node = data
    *parents, leaf = key.split(".")
    for part in parents:
        if not isinstance(node.get(part), dict):
            raise ConfigError(f"--set {key}: no section {part!r} in the config")
        node = node[part]
    if leaf not in node:
        raise ConfigError(f"--set {key}: unknown setting (typo?)")
    try:
        node[leaf] = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"--set {key}: cannot parse value {raw!r}: {exc}") from None


def _check_allowed(data: dict) -> None:
    table = ConfigTable(data)
    for key, allowed in ALLOWED_VALUES.items():
        try:
            value = table[key]
        except KeyError:
            continue
        if value not in allowed:
            raise ConfigError(f"{key}={value!r} is not one of for now allowed configurations, try with: {sorted(allowed)}")