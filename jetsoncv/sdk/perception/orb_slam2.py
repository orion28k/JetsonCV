"""ORB-SLAM2 interface stub.

This module provides a placeholder interface so applications depending on a
SLAM provider can still import and run even if ORB-SLAM2 is not set up.
"""
from typing import Optional
from ..core.types import Pose
import warnings


class OrbSlam2Interface:
    """Stub interface for ORB-SLAM2 integration.

    Methods are no-ops and return None until ORB-SLAM2 bindings are added.
    """

    def __init__(self, vocab=None, settings=None):
        warnings.warn("ORB-SLAM2 not available; using stub implementation")

    def update(self, frame) -> None:
        """Process a frame (no-op for stub)."""
        return None

    def get_pose(self) -> Optional[Pose]:
        """Return None for stub implementation."""
        return None
