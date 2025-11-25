"""Low-level controller that converts SDK Commands to Tello calls."""
from ..core.types import Command


class LowLevelController:
    """Translate `Command` into Tello SDK send_rc_control calls.

    This class keeps mapping logic in one place. It does not call Tello
    directly — the app or video input wrapper should call `tello.send_rc_control`.
    """

    @staticmethod
    def command_to_rc(cmd: Command):
        """Return a tuple (left_right, forward_backward, up_down, yaw) usable by `send_rc_control`.

        Note: mapping of vx/vy signs conforms to `djitellopy` convention (forward positive)
        """
        # Map vx (forward) -> forward_backward, vy -> left_right
        fb = int(cmd.vx)
        lr = int(cmd.vy)
        ud = int(cmd.vz)
        yaw = int(cmd.yaw_rate)
        return lr, fb, ud, yaw
