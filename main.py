import cv2
import mediapipe as mp
import cv_util as util

# Arguments
detect_hands = False
detect_pose = False
detect_holistic = True

# Initialize MediaPipe detections via the library
if detect_hands and detect_pose:
    pass
    print("Using Mediapipe Holistic detection.")
elif detect_hands:
    hands = util.init_hands()
    print("Using Mediapipe Hand detection.")
elif detect_pose:
    pose = util.init_pose()
    print("Using Mediapipe Body Pose detection.")
else:
    print("Not using Mediapipe detection")

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

    if detect_hands:
        hand_landmarks = util.process_hands(img, hands, draw=True)
    
    if detect_pose:
        pose_landmarks = util.process_pose(img, pose, draw=True)

    cv2.imshow("Window", img)
    key = cv2.waitKey(1)

    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
