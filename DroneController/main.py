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

        self.executed = False
        self.timeout = 0
        self.pTime = 0
        self.cTime = 0
        self.img = self.frame_read

    def control(self):
        self.cTime = time.time()
        fps = 1 / (self.cTime - self.pTime) if self.cTime != self.pTime else 0
        self.pTime = self.cTime

        #---------------------------------------------------
        pose = self.controller.get_pose()
        self.framer(self.frame_read.frame, fps = fps)

        # ---------------------------------------------------

    def end(self):
        self.controller.streamoff()
        self.controller.end()

    def framer(self, img, res=None, fps=None):
        """
        Optimizes window edits
        :return: image of current frame in loop
        """

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if res:
            img = cv2.resize(img, res)


        if fps:
            cv2.putText(img, f"FPS: {int(fps)}", (2, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)

        self.img = img
