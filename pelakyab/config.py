"""Configuration loader.

Reads ``config.yaml`` into nested, attribute-accessible objects so the rest of
the code can do ``cfg.detection.plate_model`` instead of dict gymnastics.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

# Project root = the folder that contains this package's parent.
ROOT = Path(__file__).resolve().parent.parent


class Config(SimpleNamespace):
    """Attribute-accessible config node with dict-style fallback."""

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def as_dict(self) -> dict:
        out: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            out[k] = v.as_dict() if isinstance(v, Config) else v
        return out


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config(**{k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _resolve_paths(cfg: "Config") -> None:
    """Make model/data/font paths absolute relative to the project ROOT."""
    def absolutize(p: str | None) -> str | None:
        if not p:
            return p
        path = Path(p)
        if path.is_absolute():
            return str(path)
        return str((ROOT / path).resolve())

    cfg.detection.plate_model = absolutize(cfg.detection.plate_model)
    cfg.detection.char_model = absolutize(cfg.detection.char_model)
    cfg.storage.json_path = absolutize(cfg.storage.json_path)
    cfg.storage.images_dir = absolutize(cfg.storage.images_dir)
    # persian_font may legitimately be an absolute system path (Windows Fonts).
    cfg.gui.persian_font = absolutize(cfg.gui.persian_font)


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load config.yaml (defaults to ROOT/config.yaml)."""
    cfg_path = Path(path) if path else (ROOT / "config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _wrap(raw)
    _resolve_paths(cfg)
    return cfg
