import cv2
import mediapipe as mp
import cv_util as util
from htc import HTC
from DroneController import main as dronert


# Arguments (Configurable)
detection_mode = "holistic"  # one of: "none", "hands", "pose", "face", "holistic"
draw = True

hand_to_cursor = False

# Create Objects
## Determine screen size
screen_size = util.get_screen_size()
## Create Hand-to-Cursor Object
if hand_to_cursor and detection_mode == "hands":
    htc = HTC(cursor_smooth=0.3, scale = 1.75)
else:
    htc = None
## Initialize detection mode
obj = util.init_detection_obj(detection_mode)
## Create drone controller
drone = dronert.DroneController()

while True:
    # Grab the latest frame from the drone video stream
    frame = drone.frame_read.frame

    if frame is None:
        print("none")
        continue

    # Flip for mirror effect
    img = cv2.flip(frame, 1)
    # -----------------------------------------------------

    # Holistic Data
    if detection_mode == "holistic":
        holistic_landmarks = util.process_holistic(img, obj, draw)

    # Hand Data
    elif detection_mode == "hands":
        hand_landmarks = util.process_hands(img, obj, draw)

    # Body Pose Data
    elif detection_mode == "pose":
        pose_landmarks = util.process_pose(img, obj, draw)

    # Facial Feature Data
    elif detection_mode == "face":
        face_landmarks = util.process_face(img, obj, draw)

    x = 1
    if x > 1.0:
        img_resize = cv2.resize(img, (0,0), fx=x, fy=x, interpolation=cv2.INTER_AREA)
    else:
        img_resize = cv2.resize(img, (0,0), fx=x, fy=x, interpolation=cv2.INTER_AREA)

    cv2.imshow("Window", img_resize)
    key = cv2.waitKey(1)

    if key == 27:  # ESC to exit
        break

drone.end()
cv2.destroyAllWindows()