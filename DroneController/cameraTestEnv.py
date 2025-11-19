import cv2
import mediapipe as mp
import time
import camera

cap = cv2.VideoCapture(0)


timeout = 0
pTime = 0
cTime = 0

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

# set_home, move_by, get_pose

while True:
    success, img = cap.read()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    img = camera.framer(img)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == 27 or timeout == 150:  # ESC to exit
        break