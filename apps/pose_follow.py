"""Pose detection and follow demo using the Tello drone.

This app demonstrates pose detection (MediaPipe) and follow-me control based on
person's torso position.

Controls:
  [T] Takeoff
  [L] Land
  [F] Toggle FOLLOW_TARGET / HOVER mode
  [ESC/q] Quit
  [-] Lower Max Yaw Speed
  [=] Increase Max Yaw Speed
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
from jetsoncv.sdk.perception.pose_detector import MediaPipePoseDetector
from jetsoncv.sdk.control.behavior import BehaviorPlanner
from jetsoncv.sdk.control.follow_pose import PoseFollowController
from jetsoncv.sdk.control.safety import SafetyFilter
from jetsoncv.sdk.control.low_level import LowLevelController
from jetsoncv.sdk.core.types import SafetyContext
from jetsoncv.sdk.core.logging_utils import get_logger

logger = get_logger("apps.pose_follow")


def run_pose_follow():
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
    
    fh, fw = video.read_frame().shape[:2]

    hub = FrameHub()
    basic = BasicCV()
    pose_detector = MediaPipePoseDetector()
    pose_controller = PoseFollowController(frame_width=fw, frame_height=fh, target_height_ratio=0.3)
    planner = BehaviorPlanner(frame_width=fw)
    planner.set_behavior("HOVER")
    safety = SafetyFilter()
    ll = LowLevelController()

    # Check if drone is already flying (in case of previous force quit)
    flying = False
    # track whether the target was centered last loop to avoid repeated stop commands
    last_centered = False
    last_mode = planner.mode

    try:
        state = video.read_state()
        if state and state.get("flying"):
            logger.warning("Drone appears to be already flying! Setting flying=True.")
            flying = True
    except Exception as e:
        logger.debug("Could not check flight state: %s", e)

    try:
        logger.info("Pose follow demo started. Controls:")
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
            landmarks, pose_conf = pose_detector.detect(frame)

            ctx = SafetyContext(basic_cv=bc_out, battery=video.read_state().get("battery") if video.read_state() else None)

            # Compute torso position and height if pose detected
            torso_center = None
            torso_height = None
            if landmarks is not None:
                torso_x_px, torso_y_px = pose_detector.get_torso_center(landmarks, frame.shape)
                if torso_x_px is not None:
                    torso_center = (torso_x_px, torso_y_px)
                    # Estimate height from shoulder-to-hip distance (in pixels)
                    shoulder_y = (landmarks[11][1] + landmarks[12][1]) / 2.0 * frame.shape[0]
                    hip_y = (landmarks[23][1] + landmarks[24][1]) / 2.0 * frame.shape[0]
                    torso_height = abs(hip_y - shoulder_y)

            # Compute command based on behavior
            if planner.mode == "FOLLOW_TARGET" and landmarks is not None:
                cmd = pose_controller.compute(torso_center, torso_height)
            else:
                cmd = planner.compute_command(detections=None, ctx=ctx)

            safe_cmd = safety.filter(cmd, ctx)

            # Send to Tello if flying
            if video._tello is not None and flying:
                lr, fb, ud, yaw = ll.command_to_rc(safe_cmd)
                # Determine if the target is horizontally centered (small deadband)
                target_centered = False
                if torso_center is not None:
                    frame_h, frame_w = frame.shape[:2]
                    deadband_px = int(frame_w * 0.05)  # 5% of frame width
                    cx = int(torso_center[0])
                    if abs(cx - frame_w // 2) <= deadband_px:
                        target_centered = True

                # If target is centered, send an explicit stop once
                if target_centered and not last_centered:
                    try:
                        video._tello.send_rc_control(0, 0, 0, 0)
                    except Exception:
                        pass
                    last_centered = True
                else:
                    # Normal behavior: send the computed RC command
                    try:
                        video._tello.send_rc_control(lr, fb, ud, yaw)
                    except Exception:
                        pass

                # if target is no longer centered, clear the flag
                if not target_centered:
                    last_centered = False

            # Build the image that the detector actually saw and display that
            model_input = pose_detector.preprocess(frame)  # RGB image
            display_frame = model_input
            
            # Update controller with actual display dimensions on first iteration
            h_display, w_display = display_frame.shape[:2]
            if pose_controller.frame_width != w_display or pose_controller.frame_height != h_display:
                pose_controller.frame_width = w_display
                pose_controller.frame_height = h_display
                # Recompute dead zone bounds with correct dimensions
                center = w_display / 2.0
                half_dead_zone = (w_display * pose_controller.zone_width_ratio) / 2.0
                pose_controller.dead_zone_bounds = (center - half_dead_zone, center + half_dead_zone)

            # Draw pose skeleton on the detector input image
            if landmarks is not None:
                pose_detector.draw_landmarks(display_frame, landmarks)

            # Draw torso center marker if detected
            if torso_center is not None:
                cx, cy = int(torso_center[0]), int(torso_center[1])
                cv2.circle(display_frame, (cx, cy), 8, (0, 0, 255), -1)
                cv2.putText(display_frame, "TARGET", (cx - 30, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw status overlay on the display frame
            h, w = display_frame.shape[:2]
            status = "FLYING" if flying else "GROUNDED"
            cv2.putText(display_frame, f"Status: {status} | Mode: {planner.mode}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            pose_str = f"Pose: detected" if landmarks is not None else "Pose: none"
            cv2.putText(display_frame, f"{pose_str} | Motion: {bc_out.motion_magnitude:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if video.read_state() and video.read_state().get("battery"):
                cv2.putText(display_frame, f"Battery: {video.read_state()['battery']}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(display_frame, f"yaw_max: {pose_controller.yaw_max}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1) 

            # Draw dead zone boundaries (vertical lines)
            try:
                dz = pose_controller.dead_zone_bounds
                # dz[0] is left zone end (left boundary of dead zone)
                # dz[1] is right zone start (right boundary of dead zone)
                # Draw thick vertical lines at dead zone boundaries
                cv2.line(display_frame, (int(dz[0]), 0), (int(dz[0]), h), (0, 255, 0), 3)
                cv2.line(display_frame, (int(dz[1]), 0), (int(dz[1]), h), (0, 255, 0), 3)
            except Exception:
                pass

            cv2.imshow("Pose Follow Demo", display_frame)

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
                        pose_controller.reset()
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
                pose_controller.reset()
                logger.info(f"Switched to mode: {new_mode}")
                # If switching out of FOLLOW mode into HOVER, send an explicit stop
                if new_mode == "HOVER" and video._tello is not None and flying:
                    try:
                        video._tello.send_rc_control(0, 0, 0, 0)
                    except Exception:
                        pass
            elif key == ord("-"):
                pose_controller.yaw_max = pose_controller.yaw_max - 1
            elif key == ord("="):
                pose_controller.yaw_max = pose_controller.yaw_max + 1
    finally:
        logger.info("Shutting down pose follow demo")
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
    run_pose_follow()
