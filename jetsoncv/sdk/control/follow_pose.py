"""Pose-following controller.

Follows a person's torso/center-of-mass using PID control based on pose keypoints.
"""
from typing import Optional, Tuple
from ..core.types import Command
import numpy as np


class PoseFollowController:
    """Generate a Command to follow a person's pose (torso center).

    Behavior:
        - Compute center error horizontally and produce yaw_rate
        - Estimate distance from torso height and produce vx (forward/back)
        - Use vertical center to adjust vz (up/down)
    """

    def __init__(
        self,
        frame_width: int = 360,
        frame_height: int = 240,
        target_height_ratio: float = 0.2,  # target person height as % of frame
        zone_width_ratio: float = 0.2,  # width of dead zone (center) as ratio of half-frame
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.target_height_ratio = target_height_ratio
        self.zone_width_ratio = zone_width_ratio

        # Define horizontal zones (centered dead zone)
        center = frame_width / 2.0
        half_dead_zone = (frame_width * zone_width_ratio) / 2.0
        self.dead_zone_bounds = (center - half_dead_zone, center + half_dead_zone)

        # Yaw control gains for zone-based system
        # Yaw limits and dynamic scaling
        self.yaw_max = 30  # maximum yaw command magnitude

        # PID gains for vertical control only
        self.pid_vz = [0.3, 0.3, 0.0]

        self._error_vz = 0.0

    def compute(self, torso_center: Optional[Tuple[float, float]], torso_height: Optional[float]) -> Command:
        """Generate command based on torso center and height using 3-zone horizontal control.

        Horizontal zones:
            - Left zone: [0, left_zone_end) -> rotate left
            - Middle zone: [left_zone_end, right_zone_start) -> no rotation
            - Right zone: [right_zone_start, frame_width) -> rotate right

        Args:
            torso_center: (x, y) in pixel coordinates
            torso_height: estimated height of person in pixels
        """
        if torso_center is None:
            return Command()

        x, y = torso_center

        # Yaw control: 3-zone system with proportional scaling
        center_x = float(self.frame_width) / 2.0
        half_width = max(center_x, 1.0)

        if x < self.dead_zone_bounds[0]:
            # Target in left zone: rotate left with magnitude proportional to distance from center
            frac = (center_x - x) / half_width  # 0..1 (0=center, 1=far edge)
            yaw_rate = float(np.clip(-self.yaw_max * frac, -self.yaw_max, 0.0))
        elif x > self.dead_zone_bounds[1]:
            # Target in right zone: rotate right with magnitude proportional to distance from center
            frac = (x - center_x) / half_width
            yaw_rate = float(np.clip(self.yaw_max * frac, 0.0, self.yaw_max))
        else:
            # Target in middle zone: no rotation
            yaw_rate = 0.0

        # Forward/back control: maintain target distance (based on person height)
        fb = 0.0
        if torso_height is not None:
            target_height = self.frame_height * self.target_height_ratio
            height_error = target_height - torso_height  # Inverted: if person is too big (close), move backward
            if abs(height_error) > 10:  # deadband
                fb = float(np.clip(height_error * 0.1, -50, 50))

        # Vertical control: center vertically
        error_vz = y - (self.frame_height / 2.0)
        vz = self.pid_vz[0] * error_vz + self.pid_vz[1] * (error_vz - self._error_vz)
        vz = float(np.clip(vz, -50, 50))

        # Store errors for next iteration
        self._error_vz = error_vz

        return Command(vx=0.0, vy=0.0, vz=0.0, yaw_rate= yaw_rate)

    def reset(self):
        """Reset PID state."""
        self._error_vz = 0.0
