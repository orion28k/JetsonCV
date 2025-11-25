"""Safety filter for commands.

This module clamps velocities and can override commands if the basic CV
module reports a collision warning.
"""
from ..core.types import Command, SafetyContext


class SafetyFilter:
    def __init__(self, max_vx=50, max_vy=50, max_vz=50, max_yaw=100):
        self.max_vx = max_vx
        self.max_vy = max_vy
        self.max_vz = max_vz
        self.max_yaw = max_yaw

    def filter(self, cmd: Command, ctx: SafetyContext) -> Command:
        # If collision warning is present, override to hover
        if ctx and ctx.basic_cv and ctx.basic_cv.collision_warning:
            return Command()  # safe hover

        safe = Command(
            vx=float(max(min(cmd.vx, self.max_vx), -self.max_vx)),
            vy=float(max(min(cmd.vy, self.max_vy), -self.max_vy)),
            vz=float(max(min(cmd.vz, self.max_vz), -self.max_vz)),
            yaw_rate=float(max(min(cmd.yaw_rate, self.max_yaw), -self.max_yaw)),
            takeoff=cmd.takeoff,
            land=cmd.land,
        )
        return safe
