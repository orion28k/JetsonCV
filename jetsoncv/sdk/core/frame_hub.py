"""FrameHub: lightweight frame distribution hub."""
from typing import Optional
import threading

import numpy as np


class FrameHub:
    """A simple thread-safe container for the latest frame.

    Modules can call `update_frame(frame)` to publish and `get_frame()` to read.
    Optionally supports a trivial frame skip policy if consumers need to.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._timestamp: Optional[float] = None

    def update_frame(self, frame: np.ndarray, timestamp: float = None) -> None:
        with self._lock:
            # store a reference — callers should copy if they need persistence
            self._frame = frame
            self._timestamp = timestamp

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame

    def get_timestamp(self):
        with self._lock:
            return self._timestamp
