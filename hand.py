import math
import cv2
import pynput

class HTC:
    def __init__(self, screen_size=(1920, 1080)):
        self.mouse = pynput.mouse.Controller()
        self.touching = False
        self._set_screen_size(screen_size)

    def hand_to_cursor(self, img, hand_landmarks):
        """
        Draw a red line between landmark 8 and landmark 4 for each hand, and if the
        distance between those two landmarks is small enough (interpreted as the
        fingertips touching), draw a large circle once at the midpoint between them.

        Args:
            img: BGR image (numpy array) to draw on.
            hand_landmarks: list-like of two elements [left_hand, right_hand], where
                each element is either:
                - a MediaPipe NormalizedLandmarkList for that hand, or
                - None if that hand is not present in the frame.
        """
        if hand_landmarks is None:
            return

        h, w = img.shape[:2]

        # How close lm8 and lm4 must be (in pixels) to be considered "touching".
        # This is relative to the image size so it scales across resolutions.
        touch_threshold = 0.07 * min(w, h)

        # Iterate over left (index 0) and right (index 1) hands
        for idx in (0, 1):
            hand = None
            if len(hand_landmarks) > idx:
                hand = hand_landmarks[idx]

            # Skip if this hand is not present or doesn't look like a landmark list
            if hand is None or not hasattr(hand, "landmark") or len(hand.landmark) <= 8:
                continue

            # Landmark 8: index fingertip, Landmark 4: thumb fingertip
            lm8 = hand.landmark[8]
            lm4 = hand.landmark[4]

            x8 = lm8.x * w
            y8 = lm8.y * h
            x4 = lm4.x * w
            y4 = lm4.y * h

            pt8 = (int(x8), int(y8))
            pt4 = (int(x4), int(y4))

            # Compute the distance between the two landmarks
            dx = x8 - x4
            dy = y8 - y4
            dist = math.hypot(dx, dy)
            
            # Compute midpoint of the two landmarks
            cx = int((x8 + x4) * 0.5)
            cy = int((y8 + y4) * 0.5)

            if dist <= touch_threshold:
                self.touching = True
            else:
                self.touching = False

            if self.touching:
                color = (0, 255, 0)  # Line color green
            else:
                color = (0, 0, 255)  # Line color red

            # Draw the line between lm8 and lm4
            cv2.line(
                img,
                pt8,
                pt4,
                color,
                2,
                cv2.LINE_AA,
            )

            if idx == 1:
                # Use the right hand (index 1) to steer the mouse cursor.
                self.landmark_to_mouse_pos((cx, cy), (h, w))

            # If the distance is small enough, draw a large circle at midpoint
            if self.touching:
                # Large circle radius relative to image size
                radius = int(0.1 * min(w, h))

                cv2.circle(
                    img,
                    (cx, cy),
                    radius,
                    color,
                    3,
                    cv2.LINE_AA,
                )

    def landmark_to_mouse_pos(self, landmark_pt, frame_size):
        """
        Convert a landmark's pixel location within the frame to screen coordinates
        and move the OS cursor to that position.

        Args:
            landmark_pt: Tuple[float, float] representing (x, y) pixels in the frame.
            frame_size: Tuple[int, int] representing (frame_height, frame_width).

        Returns:
            Tuple[int, int]: The new cursor position in screen coordinates, or None
            if inputs are invalid.
        """
        if (
            landmark_pt is None
            or frame_size is None
            or len(frame_size) < 2
            or frame_size[0] == 0
            or frame_size[1] == 0
        ):
            return None

        frame_h, frame_w = frame_size[:2]
        x_px, y_px = landmark_pt

        # Normalize to 0..1 range based on the camera frame.
        norm_x = max(0.0, min(1.0, x_px / frame_w))
        norm_y = max(0.0, min(1.0, y_px / frame_h))

        # Map to absolute screen coordinates and clamp to the desktop bounds.
        screen_x = int(norm_x * self.screen_width)
        screen_y = int(norm_y * self.screen_height)
        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))

        self.mouse.position = (screen_x, screen_y)
        return self.mouse.position

    def _set_screen_size(self, screen_size):
        """
        Normalize and store the active screen dimensions.
        """
        if (
            screen_size is None
            or len(screen_size) < 2
            or screen_size[0] <= 0
            or screen_size[1] <= 0
        ):
            screen_size = (1920, 1080)

        self.screen_width = int(screen_size[0])
        self.screen_height = int(screen_size[1])
