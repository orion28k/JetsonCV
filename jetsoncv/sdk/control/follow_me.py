"""Follow-me controller: converts detections to motion commands.

This controller implements a PID-based centering behavior based on bounding box
center and area. It is lightweight and does not require YOLO at import time;
it accepts detections produced by any detector.
"""
from typing import List, Optional, Tuple
from ..core.types import Detection, Command
import numpy as np


class FollowMeController:
    """Generate a Command to center the drone on the target detection.

    Simple behavior:
        - compute center error horizontally and produce yaw_rate
        - use bbox area to estimate distance and produce vx (forward/back)
        - use vertical center to adjust vz (up/down)
    """

    def __init__(
        self,
        frame_width: int = 360,
        frame_height: int = 240,
        target_area_min: float = 6200,
        target_area_max: float = 6800,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.target_area_min = target_area_min
        self.target_area_max = target_area_max

        # PID gains
        self.pid_yaw = [0.4, 0.4, 0.0]
        self.pid_forward = [0.02, 0.0, 0.0]

        self._error_yaw = 0.0
        self._error_forward = 0.0

    def compute(self, detections: List[Detection]) -> Command:
        """Return a Command based on the largest detection."""
        if not detections:
            return Command()

        # pick highest-confidence detection
        best = max(detections, key=lambda d: d.confidence)
        x1, y1, x2, y2 = best.bbox

        # Compute center and area
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        area = (x2 - x1) * (y2 - y1)

        # Yaw control: center horizontally
        error_yaw = cx - (self.frame_width / 2.0)
        yaw_rate = self.pid_yaw[0] * error_yaw + self.pid_yaw[1] * (error_yaw - self._error_yaw)
        yaw_rate = float(np.clip(yaw_rate, -100, 100))

        # Forward/back control: maintain target area
        fb = 0.0
        if self.target_area_min < area < self.target_area_max:
            fb = 0.0
        elif area >= self.target_area_max:
            fb = -20.0
        elif area < self.target_area_min and area > 0:
            fb = 20.0

        # Store errors for next iteration
        self._error_yaw = error_yaw
        self._error_forward = area - self.target_area_min

        return Command(vx=fb, vy=0.0, vz=0.0, yaw_rate=yaw_rate)

    def reset(self):
        """Reset PID state."""
        self._error_yaw = 0.0
        self._error_forward = 0.0

