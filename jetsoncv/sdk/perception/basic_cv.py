"""Lightweight BasicCV perception utilities using only OpenCV and numpy.

This module is intentionally lightweight and always available. It implements a
simple motion magnitude detector and a collision-warning heuristic.
"""
from typing import Optional
import numpy as np
import cv2

from ..core.types import BasicCvOutput


class BasicCV:
    """Basic CV processing: frame differencing and motion magnitude.

    Usage:
        bc = BasicCV()
        output = bc.update(frame)
    """

    def __init__(self):
        self._prev_gray = None

    def update(self, frame: np.ndarray) -> BasicCvOutput:
        if frame is None:
            return BasicCvOutput()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out = BasicCvOutput()

        if self._prev_gray is None:
            self._prev_gray = gray
            return out

        # Handle frame size mismatch (e.g., if resolution changes)
        if gray.shape != self._prev_gray.shape:
            self._prev_gray = gray
            return out

        # compute frame difference and a simple motion magnitude metric
        diff = cv2.absdiff(gray, self._prev_gray)
        mag = float(np.mean(diff))
        out.motion_magnitude = mag
        out.collision_warning = mag > 25.0  # heuristic threshold

        self._prev_gray = gray
        return out
