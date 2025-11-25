"""Behavior planner that selects which controller to use based on mode."""
from typing import Optional, List
from ..core.types import Command, Detection, SafetyContext
from .follow_me import FollowMeController


class BehaviorPlanner:
    """Simple behavior manager.

    Modes: HOVER, FOLLOW_TARGET, EMERGENCY_STOP
    """

    def __init__(self, frame_width: int = 360):
        self.mode = "HOVER"
        self._follow = FollowMeController(frame_width=frame_width)

    def set_behavior(self, mode: str):
        self.mode = mode

    def compute_command(self, detections: Optional[List[Detection]] = None, ctx: Optional[SafetyContext] = None) -> Command:
        if self.mode == "FOLLOW_TARGET":
            return self._follow.compute(detections or [])
        if self.mode == "EMERGENCY_STOP":
            return Command()  # hover
        # default HOVER
        return Command()
