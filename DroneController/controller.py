import time
from queue import Queue
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
        
        self.command_q = Queue()

    def togglepropellors(self):
        if not self.controller.is_flying:
            self.controller.takeoff()
            time.sleep(3)
        else:
            self.controller.land()
            time.sleep(5)

    def move(self, velocity = (0,0,0), yaw = 0):
        if self.flying:
            self.controller.send_rc_control(velocity[0],velocity[1],velocity[2],yaw)
            time.sleep(5)
        else:
            print("Drone not flying")

    def end(self):
        self.controller.streamoff()
        self.controller.end()

    def rc_worker(self):
        while True:
            vx, vy, vz, yaw = self.command_q.get()
            self.controller.send_rc_control(vx, vy, vz, yaw)
            self.command_q.task_done()

