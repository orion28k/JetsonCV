"""Primary demo app (refactored from original main.py).

This app uses the SDK core modules and runs a simple loop that displays the
Tello frames and runs BasicCV. Advanced modules are optional and will be
used only if available.
"""
import time
import cv2

from jetsoncv.sdk.core.video_input import TelloVideoInput
from jetsoncv.sdk.core.frame_hub import FrameHub
from jetsoncv.sdk.perception.basic_cv import BasicCV
from jetsoncv.sdk.control.behavior import BehaviorPlanner
from jetsoncv.sdk.control.safety import SafetyFilter
from jetsoncv.sdk.control.low_level import LowLevelController
from jetsoncv.sdk.core.types import SafetyContext
from jetsoncv.sdk.core.logging_utils import get_logger

logger = get_logger("apps.main")


def run_demo(active_behavior: str = "HOVER"):
    # initialize components
    video = TelloVideoInput()
    try:
        video.connect()
        video.start_stream()
    except Exception as e:
        logger.warning("Tello connection/stream failed: %s", e)

    hub = FrameHub()
    basic = BasicCV()
    planner = BehaviorPlanner(frame_width=360)
    planner.set_behavior("FOLLOW_TARGET" if active_behavior == "FOLLOW" else "HOVER")
    safety = SafetyFilter()
    ll = LowLevelController()

    try:
        while True:
            frame = video.read_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # publish frame
            hub.update_frame(frame)

            # run lightweight perception
            bc_out = basic.update(frame)

            ctx = SafetyContext(basic_cv=bc_out, battery=video.read_state().get("battery") if video.read_state() else None)

            # For demo: no detection modules by default -> planner in HOVER or FOLLOW will be limited
            cmd = planner.compute_command(detections=None, ctx=ctx)
            safe_cmd = safety.filter(cmd, ctx)

            # map and send to Tello
            if video._tello is not None:
                lr, fb, ud, yaw = ll.command_to_rc(safe_cmd)
                try:
                    video._tello.send_rc_control(lr, fb, ud, yaw)
                except Exception:
                    # ignore send errors during demo
                    pass

            # draw a simple status overlay and show
            h, w = frame.shape[:2]
            cv2.putText(frame, f"Mode: {planner.mode}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Motion: {bc_out.motion_magnitude:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("JetsonCV Demo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
    finally:
        logger.info("Shutting down demo")
        cv2.destroyAllWindows()
        try:
            video.stop_stream()
            video.close()
        except Exception:
            pass


if __name__ == "__main__":
    run_demo()
