"""Minimal example using only core video input, frame hub, and BasicCV.

This script demonstrates that the SDK can run in a minimal configuration
without YOLO, ORB-SLAM2, or other advanced modules installed.
"""
import time
import cv2

from jetsoncv.sdk.core.video_input import TelloVideoInput
from jetsoncv.sdk.perception.basic_cv import BasicCV
from jetsoncv.sdk.core.logging_utils import get_logger

logger = get_logger("apps.example_basic_cv")


def run():
    video = TelloVideoInput()
    try:
        video.connect()
        video.start_stream()
    except Exception as e:
        logger.warning("Tello stream not available: %s", e)

    basic = BasicCV()

    try:
        while True:
            frame = video.read_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            out = basic.update(frame)
            cv2.putText(frame, f"Motion: {out.motion_magnitude:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if out.collision_warning:
                cv2.putText(frame, "COLLISION WARNING", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Basic CV", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        try:
            video.stop_stream()
            video.close()
        except Exception:
            pass


if __name__ == '__main__':
    run()
