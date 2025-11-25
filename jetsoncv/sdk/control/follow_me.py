"""Follow-me controller: converts detections to motion commands.

This controller implements a simple PID-like centering behavior based on
bounding box center and area. It is lightweight and does not require YOLO
at import time; it accepts detections produced by any detector.
"""
from typing import List, Optional
from ..core.types import Detection, Command
import numpy as np


class FollowMeController:
    """Generate a Command to center the drone on the target detection.

    Simple behavior:
        - compute center error horizontally and produce yaw_rate
        - use bbox area to estimate distance and produce vx
    """

    def __init__(self, frame_width: int = 360, target_area: float = 20000.0):
        self.frame_width = frame_width
        self.target_area = target_area
        self.p_yaw = 0.003
        self.p_forward = 0.00002

    def compute(self, detections: List[Detection]) -> Command:
        if not detections:
            return Command()

        # pick highest-confidence detection
        best = max(detections, key=lambda d: d.confidence)
        x1, y1, x2, y2 = best.bbox
        cx = (x1 + x2) / 2.0
        area = (x2 - x1) * (y2 - y1)

        error = cx - (self.frame_width / 2.0)
        yaw_rate = float(np.clip(self.p_yaw * error, -100, 100))
        forward = 0.0
        if area < self.target_area and area > 0:
            forward = float(np.clip(self.p_forward * (self.target_area - area), -50, 50))

        return Command(vx=forward, vy=0.0, vz=0.0, yaw_rate=yaw_rate)
