import cv2
import mediapipe as mp
import cv_util as util
import hand

# Arguments (Configurable)
detection_mode = "hands"  # one of: "none", "hands", "pose", "face", "holistic"
draw = False

# Initialize detection mode
if detection_mode == "holistic":
    # Multiple detections requested -> use Holistic
    holistic = util.init_holistic()
    print("[INFO]: Using Mediapipe Holistic detection (hands + pose + face).")
elif detection_mode == "hands":
    hands = util.init_hands()
    print("[INFO]: Using Mediapipe Hand detection.")
elif detection_mode == "pose":
    pose = util.init_pose()
    print("[INFO]: Using Mediapipe Body Pose detection.")
elif detection_mode == "face":
    face = util.init_face()
    print("[INFO]: Using Mediapipe Facial Feature detection.")
else:
    print("[INFO]: Not using Mediapipe detection")

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

    # Holistic Data
    if detection_mode == "holistic":
        holistic_landmarks = util.process_holistic(img, holistic, draw)

    # Hand Data
    elif detection_mode == "hands":
        hand_landmarks = util.process_hands(img, hands, draw)

        if hand_landmarks:
            hand.draw_effect(img, hand_landmarks)

    # Body Pose Data
    elif detection_mode == "pose":
        pose_landmarks = util.process_pose(img, pose, draw)

    # Facial Feature Data
    elif detection_mode == "face":
        face_landmarks = util.process_face(img, face, draw)

    cv2.imshow("Window", img)
    key = cv2.waitKey(1)

    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
