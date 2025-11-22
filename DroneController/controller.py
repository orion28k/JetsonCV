import time
import cv2
from DroneController.coordinate_tello import CoordinateTello

class DroneController:
    def __init__(self):
        self.controller = CoordinateTello()
        try:
            self.controller.connect()
        except Exception as exc:
            print(f"Failed to connect to Tello: {exc}")
            raise SystemExit(1)

        try:
            self.frame_read = self.controller.get_frame_read()
        except NotImplementedError as exc:
            print(f"Video stream unavailable: {exc}")
            self.controller.end()
            raise SystemExit(1)

        self.flying = False
        self.timeout = 0
        self.pTime = 0
        self.cTime = 0
        self.img = self.frame_read

    def togglepropellors(self):
        if not self.flying:
            self.controller.takeoff()
            self.flying = False
        else:
            self.controller.land()
            self.flying = True

    def move(self, velocity = (0,0,0), yaw = 0):
        if self.flying:
            self.controller.send_rc_control(velocity[0],velocity[1],velocity[2],yaw)
        else:
            print("Drone not flying")

    def end(self):
        self.controller.streamoff()
        self.controller.end()

