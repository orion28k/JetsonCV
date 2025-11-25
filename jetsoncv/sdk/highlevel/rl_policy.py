"""RL policy stub for future integration with gym-dji-tello style environments."""
from typing import Any, Optional


class RlPolicyInterface:
    def __init__(self):
        # TODO: Load RL policy when available
        pass

    def update(self, observation: Any) -> Optional[dict]:
        """Return an action/command dictionary or None for no-op."""
        return None
