import cv2
import mediapipe as mp
import cv_util as util
from htc import HTC
from DroneController import controller


# Arguments (Configurable)
detection_mode = "holistic"  # one of: "none", "hands", "pose", "face", "holistic"
draw = True

hand_to_cursor = False

# Create Objects
if hand_to_cursor and detection_mode == "hands":
    htc = HTC(cursor_smooth=0.3, scale = 1.75)
else:
    htc = None
## Initialize detection mode
obj = util.init_detection_obj(detection_mode)
## Create drone controller
drone = controller.DroneController()

while True:
    # Grab the latest frame from the drone video stream
    frame = drone.frame_read.frame
    if frame is None:
        print("none")
        continue
    img = cv2.flip(frame, 1)

    y,x = img.shape[:2]
    cv2.line(img, (int(x/2),0), (int(x/2),y), color=(0,255,0)) #pt1[x,y], pt2[x,y]

    # if landmark is less than x/2: print "left"

    # Holistic Data
    if detection_mode == "holistic":
        holistic_landmarks = util.process_holistic(img, obj, draw)

        if holistic_landmarks and holistic_landmarks.pose_landmarks:
            pose_landmarks = holistic_landmarks.pose_landmarks.landmark
            target_ids = (11, 12, 23, 24)
            coords = []
            for landmark_id in target_ids:
                lm = pose_landmarks[landmark_id]
                coords.append(
                    (landmark_id, int(lm.x * x), int(lm.y * y))
                )
            coord_msg = ", ".join(
                f"id {landmark_id}: ({px}, {py})" for landmark_id, px, py in coords
            )
            print(f"Pose landmarks -> {coord_msg}")

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
