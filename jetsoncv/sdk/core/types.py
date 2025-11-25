from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class Pose:
    """Simple pose representation (units are left to the integrator)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass
class Detection:
    """Generic detection result.

    bbox: (x1,y1,x2,y2)
    label: Optional[str]
    confidence: float
    """
    bbox: Tuple[int, int, int, int]
    label: Optional[str]
    confidence: float = 0.0


@dataclass
class BasicCvOutput:
    collision_warning: bool = False
    motion_magnitude: float = 0.0
    notes: Optional[str] = None


@dataclass
class Command:
    """Abstract command used by the planner and low-level controller.

    vx: forward/back velocity (positive forward),
    vy: lateral velocity (positive right),
    vz: vertical velocity (positive up),
    yaw_rate: rotation rate (deg/s or normalized),
    takeoff: whether to trigger takeoff, land: whether to land
    """
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0
    takeoff: bool = False
    land: bool = False


@dataclass
class SafetyContext:
    pose: Optional[Pose] = None
    basic_cv: Optional[BasicCvOutput] = None
    battery: Optional[float] = None


@dataclass
class DroneState:
    battery: Optional[int] = None
    flying: bool = False
    raw: Optional[dict] = None
