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


def handDetection(img):
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1)
    mpDraw = mp.solutions.drawing_utils

    if hands:
        results = hands.process(img)
        # print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    if id == 7 or 8 or 11 or 12 or 16 or 18 or 19 or 20:
                        # print(id, lm)
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        # print(id, cx, cy)
                    else:
                        break

                    if id == 8:
                        ix, iy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    elif id == 7:
                        imx, imy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    elif id == 12:
                        iix, iiy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    elif id == 11:
                        iimx, iimy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    elif id == 16:
                        iiix, iiiy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    elif id == 15:
                        iiimx, iiimy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    elif id == 20:
                        ivx, ivy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    elif id == 19:
                        ivmx, ivmy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    return img