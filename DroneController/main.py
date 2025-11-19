"""

Augments the DJI TELLO EDU DRONE

Upgrades:
- Keyboard Control
- Live Camera Feed

"""
import os
import time
import camera
import cv2
import mediapipe as mp
import pygame

from coordinate_tello import CoordinateTello


controller = CoordinateTello()
try:
    controller.connect()
except Exception as exc:
    print(f"Failed to connect to Tello: {exc}")
    raise SystemExit(1)

# Initialize pygame window
pygame.init()
win = pygame.display.set_mode((700, 480))
pygame.display.set_caption("Tello Camera")
clock = pygame.time.Clock()

global img
executed = False
try:
    frame_read = controller.get_frame_read()
except NotImplementedError as exc:
    print(f"Video stream unavailable: {exc}")
    controller.end()
    raise SystemExit(1)

timeout = 0
pTime = 0
cTime = 0

doHandDetection = False

movement_history = []
record_start_time = None


def getKey(keyName):
    ans = False
    for eve in pygame.event.get(): pass
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, 'K_{}'.format(keyName))
    #print('K_{}'.format(keyName))

    if keyInput[myKey]:
        ans = True
    pygame.display.update()
    return ans


def getKeyboardInput():
    global controller
    lr, fb, ud, yv = 0, 0, 0, 0

    speed = 50

    if getKey("LEFT"):
        lr = -speed
    elif getKey("RIGHT"):
        lr = speed

    if getKey("w"):
        fb = speed
    elif getKey("s"):
        fb = -speed

    if getKey("UP"):
        ud = speed
    elif getKey("DOWN"):
        ud = -speed

    if getKey("a"):
        yv = -speed
    elif getKey("d"):
        yv = speed

    if getKey("f") and controller.is_flying():
        controller.land()

    if getKey("r") and not controller.is_flying():
        controller.takeoff()

    if getKey("z") and controller.is_flying():
        controller.flip_forward()
        time.sleep(1)

    if getKey("2") and controller.is_flying():
        controller.goto(0,0,2)
        time.sleep(1)

    if getKey("1"):
        pygame.quit()
        controller.streamoff()
        if controller.is_flying():
            controller.land()
        controller.end()
        exit()

    if getKey("p"):
        try:
            path = "Images"
            cv2.imwrite(os.path.join(path, str(time.time()) + ".jpg"), img)
            print(f"Picture Taken: @{time.time()}.jpg")
            time.sleep(0.3)
        except Exception as e:
            print(f"Error taking picture: {e}")

    return [lr, fb, ud, yv]


while True:
    dt_ms = clock.tick(30)  # Limit to 30 FPS and capture elapsed milliseconds
    dt = dt_ms / 1000.0 if dt_ms else 1 / 30
    vals = getKeyboardInput()

    try:
        if controller.is_flying() and vals != [0, 0, 0, 0]:
            yaw_cmd = vals[3] * 2
            controller.send_rc_control(vals[0], vals[1], vals[2], yaw_cmd)
            executed = False
            controller.dead_reckon_rc(vals[0], vals[1], vals[2], yaw_cmd, dt)
        elif not executed:
            controller.send_rc_control(0, 0, 0, 0)
            executed = True
            time.sleep(0.05)
    except Exception as e:
        print(f"Drone command error: {e}")


    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime != pTime else 0
    pTime = cTime

    pose = controller.get_pose()
    img = camera.framer(frame_read.frame, fps = fps)

    # Dsiplay battery life
    cv2.putText(img, f"Battery: {controller.get_battery()}", (2, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
    # Display coordinates and yaw
    cv2.putText(img, f"Coords: x={pose.x:.2f}, y={pose.y:.2f}, z={pose.z:.2f}, yaw={pose.yaw_deg:.1f}",
                (2, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    # Display with pygame
    surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
    win.blit(surf, (0, 0))
    pygame.display.update()

    if timeout == 150:
        break

pygame.quit()
controller.streamoff()
controller.end()
