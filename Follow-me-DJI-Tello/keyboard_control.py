from djitellopy import tello
import cv2
import KeyPressModule as kp

from time import sleep

kp.init()

me = tello.Tello()

me.connect()

print(me.get_battery())

def getKeyboardInput():

    lr, fb, ud, yv = 0, 0, 0, 0

    speed = 40

    if kp.getKey("LEFT"): 
        lr = -speed

    elif kp.getKey("RIGHT"):
        lr = speed

    if kp.getKey("UP"): 
        fb = speed

    elif kp.getKey("DOWN"):
        fb = -speed

    if kp.getKey("w"):
        ud = speed

    elif kp.getKey("s"):
        ud = -speed

    if kp.getKey("a"):
        yv = -speed

    elif kp.getKey("d"):
        yv = speed

    if kp.getKey("q"):
        me.land()
        sleep(3)
        quit()

    if kp.getKey("e"): 
        print("im here")
        me.takeoff()

    return [lr, fb, ud, yv]


me.streamon()

while True:

    vals = getKeyboardInput()

    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    print(me.get_battery())
    sleep(0.05)


    img = me.get_frame_read().frame

    img = cv2.resize(img, (360, 240))

    cv2.imshow("lol", img)

    cv2.waitKey(1)