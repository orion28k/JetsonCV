import cv2
import mediapipe as mp


def framer(img, res = None, fps = None, doHandDetection = False):
    """
    Optimizes window edits
    :return: image of current frame in loop
    """

    if img is None:
        return  # Skip if no frame

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if res:
        img = cv2.resize(img, res)

    if doHandDetection:
        img = handDetection(img)

    if fps:
        cv2.putText(img, f"FPS: {int(fps)}", (2, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)

    return img