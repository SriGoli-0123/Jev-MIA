"""YAML experiment configs with an optional ``base:`` key for small overrides (used by smoke configs)."""
import copy
from pathlib import Path

import yaml


def deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path}: config must be a mapping")
    base = config.pop("base", None)
    if base is not None:
        config = deep_merge(load_config(path.parent / base), config)
    return config


def dump_config(config, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
