"""Face detection and follow demo using the Tello drone.

This app demonstrates face detection (Haar cascade) and follow-me control.

Controls:
  [T] Takeoff
  [L] Land
  [F] Toggle FOLLOW_TARGET / HOVER mode
  [ESC/q] Quit
"""
import sys
import os
import time
import cv2

# Ensure jetsoncv module can be imported from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jetsoncv.sdk.core.video_input import TelloVideoInput
from jetsoncv.sdk.core.frame_hub import FrameHub
from jetsoncv.sdk.perception.basic_cv import BasicCV
from jetsoncv.sdk.perception.face_detector import HaarFaceDetector
from jetsoncv.sdk.control.behavior import BehaviorPlanner
from jetsoncv.sdk.control.follow_me import FollowMeController
from jetsoncv.sdk.control.safety import SafetyFilter
from jetsoncv.sdk.control.low_level import LowLevelController
from jetsoncv.sdk.core.types import SafetyContext, Detection
from jetsoncv.sdk.core.logging_utils import get_logger

logger = get_logger("apps.face_follow")


def draw_detections(frame, detections):
    """Draw bounding boxes for detections on frame."""
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = det.label or "unknown"
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def run_face_follow():
    # Initialize components
    video = TelloVideoInput()
    try:
        logger.info("Attempting to connect to Tello...")
        video.connect()
        logger.info("Connected. Starting stream...")
        video.start_stream()
        logger.info("Stream started successfully.")
    except Exception as e:
        logger.error("Tello connection/stream failed: %s. Exiting.", e)
        logger.info("Make sure you are connected to the Tello Wi-Fi network.")
        return

    hub = FrameHub()
    basic = BasicCV()
    face_detector = HaarFaceDetector()
    follow_controller = FollowMeController(frame_width=360, frame_height=240)
    planner = BehaviorPlanner(frame_width=360)
    planner.set_behavior("HOVER")
    safety = SafetyFilter()
    ll = LowLevelController()

    flying = False

    try:
        logger.info("Face follow demo started. Controls:")
        logger.info("  [T] Takeoff")
        logger.info("  [L] Land")
        logger.info("  [F] Toggle FOLLOW_TARGET / HOVER mode")
        logger.info("  [ESC/q] Quit")

        while True:
            frame = video.read_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Publish frame
            hub.update_frame(frame)

            # Run detectors
            bc_out = basic.update(frame)
            detections = face_detector.detect(frame)

            ctx = SafetyContext(basic_cv=bc_out, battery=video.read_state().get("battery") if video.read_state() else None)

            # Compute command based on behavior
            if planner.mode == "FOLLOW_TARGET" and detections:
                cmd = follow_controller.compute(detections)
            else:
                cmd = planner.compute_command(detections=detections, ctx=ctx)

            safe_cmd = safety.filter(cmd, ctx)

            # Send to Tello if flying
            if video._tello is not None and flying:
                lr, fb, ud, yaw = ll.command_to_rc(safe_cmd)
                try:
                    video._tello.send_rc_control(lr, fb, ud, yaw)
                except Exception:
                    pass

            # Build the image that the detector actually saw and display that
            model_input = face_detector.preprocess(frame)  # grayscale image
            display_frame = cv2.cvtColor(model_input, cv2.COLOR_GRAY2BGR)

            # Draw detections on the detector input image
            draw_detections(display_frame, detections)

            # Draw status overlay on the display frame
            h, w = display_frame.shape[:2]
            status = "FLYING" if flying else "GROUNDED"
            cv2.putText(display_frame, f"Status: {status} | Mode: {planner.mode}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Faces: {len(detections)} | Motion: {bc_out.motion_magnitude:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if video.read_state() and video.read_state().get("battery"):
                cv2.putText(display_frame, f"Battery: {video.read_state()['battery']}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("Face Follow Demo", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                logger.info("Quitting...")
                if flying:
                    logger.info("Landing drone...")
                    try:
                        video._tello.land()
                    except Exception:
                        pass
                break
            elif key == ord("t") or key == ord("T"):
                if not flying and video._tello is not None:
                    logger.info("Taking off...")
                    try:
                        video._tello.takeoff()
                        flying = True
                        follow_controller.reset()
                    except Exception as e:
                        logger.error("Takeoff failed: %s", e)
            elif key == ord("l") or key == ord("L"):
                if flying and video._tello is not None:
                    logger.info("Landing...")
                    try:
                        video._tello.land()
                        flying = False
                    except Exception as e:
                        logger.error("Land failed: %s", e)
            elif key == ord("f") or key == ord("F"):
                new_mode = "FOLLOW_TARGET" if planner.mode == "HOVER" else "HOVER"
                planner.set_behavior(new_mode)
                follow_controller.reset()
                logger.info(f"Switched to mode: {new_mode}")
    finally:
        logger.info("Shutting down face follow demo")
        cv2.destroyAllWindows()
        try:
            if flying:
                video._tello.land()
        except Exception:
            pass
        try:
            video.stop_stream()
            video.close()
        except Exception:
            pass


if __name__ == "__main__":
    run_face_follow()
