import math
import cv2


def hand_to_cursor(img, hand_landmarks):
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
    
    touching = False

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

        if dist <= touch_threshold:
            touching = True
        else:
            touching = False

        # Determine color: green if touching, red otherwise
        if touching:
            color = (0, 255, 0)  # Green
        else:
            color = (0, 0, 255)  # Red

        # Draw the line between lm8 and lm4
        cv2.line(
            img,
            pt8,
            pt4,
            color,
            2,
            cv2.LINE_AA,
        )

        # If the distance is small enough, draw a large circle at midpoint
        if touching:
            cx = int((x8 + x4) * 0.5)
            cy = int((y8 + y4) * 0.5)

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