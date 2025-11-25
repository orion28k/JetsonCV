"""DroneFormer integration stub.

This is a placeholder interface for future DroneFormer integration.
"""
from typing import Optional


class DroneFormerInterface:
    def __init__(self):
        # TODO: Implement integration with a transformer-based navigator
        pass

    def set_instruction(self, text: str) -> None:
        """Set a high-level instruction (no-op for stub)."""
        return None

    def update(self, frame, pose) -> Optional[dict]:
        """Process current observation and return a high-level goal if available."""
        return None
