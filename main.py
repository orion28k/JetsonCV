import cv2
import mediapipe as mp
import cv_util as util
from htc import HTC

# Arguments (Configurable)
detection_mode = "hands"  # one of: "none", "hands", "pose", "face", "holistic"
draw = False

# Create Objects

## Capture default cameras
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap.release()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to access a webcam on indices 1 or 0.")
    
## Determine screen size for cursor mapping
screen_size = util.get_screen_size()

## Create Hand-to-Cursor Object
htc = HTC(cursor_smooth=0.3, scale = 1.75)

# Initialize detection mode
obj = util.init_detection_obj(detection_mode)

while True:
    success, img = cap.read()
    if not success or img is None:
        continue

    # Flip for mirror effect
    img = cv2.flip(img, 1)

    # Holistic Data
    if detection_mode == "holistic":
        holistic_landmarks = util.process_holistic(img, obj, draw)

    # Hand Data
    elif detection_mode == "hands":
        hand_landmarks = util.process_hands(img, obj, draw)

        if hand_landmarks:
            htc.hand_to_cursor(img, hand_landmarks)

    # Body Pose Data
    elif detection_mode == "pose":
        pose_landmarks = util.process_pose(img, obj, draw)

    # Facial Feature Data
    elif detection_mode == "face":
        face_landmarks = util.process_face(img, obj, draw)

    x = 0.1
    if x > 1.0:
        img_resize = cv2.resize(img, (0,0), fx=x, fy=x, interpolation=cv2.INTER_AREA)
    else:
        img_resize = cv2.resize(img, (0,0), fx=x, fy=x, interpolation=cv2.INTER_AREA)

    cv2.imshow("Window", img_resize)
    key = cv2.waitKey(1)

    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()