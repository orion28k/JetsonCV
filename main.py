
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

    if draw_hands:
        mp_drawing = mp.solutions.drawing_utils

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img, # image
                    hand_landmarks, # hand landmarks
                    mp.solutions.hands.HAND_CONNECTIONS, # list of index pairs that define the connections
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3), # customize landmarks
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2) # customize lines
                )

    cv2.imshow("Window", img)
    key = cv2.waitKey(1)

    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
