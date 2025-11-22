from djitellopy import Tello
import cv2
import time
import math

tello = Tello()
tello.connect()
print("Battery Life:" + str(tello.get_battery()))

# -----------------------------------------------
tello.takeoff()

# Example movement: move forward
tello.send_rc_control(0, 50, 0, 0)  # Move forward

tello.land()
# -----------------------------------------------

tello.streamoff()
tello.end()
cv2.destroyAllWindows()