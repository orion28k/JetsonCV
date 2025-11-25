"""Tello video input wrapper.

This module provides a minimal TelloVideoInput class that manages connection and
streaming. It is intentionally lightweight so apps can use it without pulling
advanced dependencies.
"""
from typing import Optional
import time
import threading

try:
    from djitellopy import tello as djitello
except Exception:  # pragma: no cover - djitellopy optional
    djitello = None

import numpy as np


class TelloVideoInput:
    """Simple wrapper around `djitellopy.Tello` for frame retrieval.

    Methods:
        connect()
        start_stream()
        stop_stream()
        read_frame() -> np.ndarray | None
        read_state() -> dict | None
    """

    def __init__(self, tello_obj=None):
        self._tello = tello_obj
        self._started = False

    def connect(self):
        if djitello is None and self._tello is None:
            raise RuntimeError("djitellopy not installed; Tello support unavailable")

        if self._tello is None:
            self._tello = djitello.Tello()
        # connect but do not take off
        self._tello.connect()

    def start_stream(self):
        if self._tello is None:
            raise RuntimeError("Tello not connected")
        try:
            self._tello.streamon()
            self._started = True
        except Exception as e:
            # Some environments do not support streaming; propagate a controlled error
            self._started = False
            raise RuntimeError(f"Stream start failed: {e}")

    def stop_stream(self):
        if self._tello is not None:
            try:
                self._tello.streamoff()
            except Exception:
                pass
        self._started = False

    def read_frame(self) -> Optional[np.ndarray]:
        """Return the most recent frame from the Tello or None."""
        if self._tello is None or not self._started:
            return None
        try:
            frame = self._tello.get_frame_read().frame
            return frame
        except Exception:
            return None

    def read_state(self) -> Optional[dict]:
        if self._tello is None:
            return None
        try:
            # djitellopy exposes basic telemetry via .get_battery(), but more
            # advanced state may be available depending on version.
            return {"battery": self._tello.get_battery()}
        except Exception:
            return None

    def close(self):
        self.stop_stream()
        try:
            if self._tello is not None:
                self._tello.end()
        except Exception:
            pass
