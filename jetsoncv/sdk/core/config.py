"""Simple configuration utilities for the SDK."""
from typing import Dict, Any


DEFAULTS: Dict[str, Any] = {
    "frame_width": 360,
    "behavior": "HOVER",
}


def get_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = DEFAULTS.copy()
    if overrides:
        cfg.update(overrides)
    return cfg
