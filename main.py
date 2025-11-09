import cv2
import mediapipe as mp
import cv_util as util

draw_hands = True
hand_effect = False

# Initialize MediaPipe Hands via the library
hands = util.init_hands()

# Capture default cameras
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap.release()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to access a webcam on indices 1 or 0.")

while True:
    success, img = cap.read()
    if not success or img is None:
        continue

    # Flip for mirror effect
    img = cv2.flip(img, 1)

    util.process_hands(img, hands, draw=True)

    cv2.imshow("Window", img)
    key = cv2.waitKey(1)

    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
