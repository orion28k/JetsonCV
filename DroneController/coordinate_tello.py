import time
import math
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Any

try:
    from djitellopy import Tello  # Officially, community SDK wrapper for Tello
    _DJITELLOPY_AVAILABLE = True
except Exception:
    _DJITELLOPY_AVAILABLE = False


# ------------------------------ Math helpers ------------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def wrap_deg(angle_deg: float) -> float:
    """Wrap angle to [-180, 180)."""
    a = (angle_deg + 180.0) % 360.0 - 180.0
    return a

def rot_world_to_body(yaw_deg: float, vx: float, vy: float) -> Tuple[float, float]:
    """Rotate a world-frame (x,y) vector into drone body frame using yaw (deg)."""
    th = math.radians(yaw_deg)
    c, s = math.cos(th), math.sin(th)
    # World ENU -> Body: Xb =  c*Xw + s*Yw ; Yb = -s*Xw + c*Yw (yaw about +Z)
    xb =  c * vx + s * vy
    yb = -s * vx + c * vy
    return xb, yb

def rot_body_to_world(yaw_deg: float, vx: float, vy: float) -> Tuple[float, float]:
    """Rotate a body-frame (x,y) vector into world frame."""
    th = math.radians(yaw_deg)
    c, s = math.cos(th), math.sin(th)
    xw =  c * vx - s * vy
    yw =  s * vx + c * vy
    return xw, yw


# ------------------------------ Data structures ------------------------------

@dataclass
class Pose:
    """Local ENU pose, meters and degrees."""
    x: float = 0.0  # East
    y: float = 0.0  # North
    z: float = 0.0  # Up
    yaw_deg: float = 0.0  # +Z right-hand, degrees

@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    i: float = 0.0
    prev_e: float = 0.0

    def step(self, e: float, dt: float, i_limit: float = 1.0) -> float:
        self.i = clamp(self.i + e * dt, -i_limit, i_limit)
        d = (e - self.prev_e) / dt if dt > 1e-6 else 0.0
        self.prev_e = e
        return self.kp * e + self.ki * self.i + self.kd * d


# ------------------------------ Low-level driver abstraction ------------------------------

class BaseTello:
    """
    Minimal interface we need from a Tello-like object.
    You can replace this with a simulator by subclassing and overriding methods.
    """
    def connect(self): ...
    def get_battery(self) -> int: return 100
    def streamon(self): ...
    def streamoff(self): ...
    def takeoff(self): ...
    def land(self): ...
    def end(self): ...
    def get_height(self) -> Optional[int]: return None  # cm
    def get_yaw(self) -> Optional[int]: return None     # deg
    def send_rc_control(self, lr: int, fb: int, ud: int, yaw: int): ...
    def get_frame_read(self) -> Any:
        raise NotImplementedError("Video stream not available for this driver.")
    def flip_forward(self): ...
    def is_flying(self) -> bool: return False
    def stop(self): self.send_rc_control(0, 0, 0, 0)


class DjitelloDriver(BaseTello):
    def __init__(self):
        if not _DJITELLOPY_AVAILABLE:
            raise RuntimeError("djitellopy not installed. `pip install djitellopy`")
        self._t = Tello()

    def connect(self):
        self._t.connect()

    def get_battery(self) -> int:
        return self._t.get_battery()

    def streamon(self):
        try:
            self._t.streamon()
            print("stream on")
        except Exception:
            pass

    def streamoff(self):
        try:
            self._t.streamoff()
        except Exception:
            pass

    def takeoff(self):
        self._t.takeoff()

    def land(self):
        self._t.land()

    def end(self):
        try:
            self._t.end()
        except Exception:
            pass

    def get_height(self) -> Optional[int]:
        # Returns cm; can be None if telemetry not ready
        try:
            return self._t.get_height()
        except Exception:
            return None

    def get_yaw(self) -> Optional[int]:
        try:
            return self._t.get_yaw()
        except Exception:
            return None

    def send_rc_control(self, lr: int, fb: int, ud: int, yaw: int):
        # lr: left/right(+right), fb: forward/back(+forward), ud: up/down(+up), yaw: cw/ccw(+cw) in [-100,100]
        self._t.send_rc_control(lr, fb, ud, yaw)

    def get_frame_read(self) -> Any:
        return self._t.get_frame_read()

    def flip_forward(self):
        self._t.flip_forward()

    def is_flying(self) -> bool:
        try:
            return bool(self._t.is_flying)
        except AttributeError:
            return False


class SimTello(BaseTello):
    """
    Simple simulator driven by commanded velocities.
    Useful for unit tests without hardware.
    """
    def __init__(self):
        self.pose = Pose()
        self._last_cmd = (0,0,0,0)
        self._running = False
        self._thread = None

    def connect(self): pass
    def streamon(self): pass
    def streamoff(self): pass
    def takeoff(self): self._running = True; self._start_integration()
    def land(self): self._running = False
    def end(self): self._running = False

    def _start_integration(self):
        if self._thread: return
        def loop():
            prev = time.time()
            while self._running:
                now = time.time()
                dt = now - prev; prev = now
                lr, fb, ud, yaw = self._last_cmd
                # Map rc [-100,100] to rough m/s (scale chosen conservatively)
                vx_body = fb * 0.01  # m/s forward
                vy_body = lr * 0.01  # m/s right
                vz      = ud * 0.006 # m/s up
                yawrate = yaw * 0.8  # deg/s
                # Integrate
                dxw, dyw = rot_body_to_world(self.pose.yaw_deg, vx_body*dt, vy_body*dt)
                self.pose.x += dxw
                self.pose.y += dyw
                self.pose.z = max(0.0, self.pose.z + vz*dt)
                self.pose.yaw_deg = wrap_deg(self.pose.yaw_deg + yawrate*dt)
                time.sleep(0.02)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def get_height(self) -> Optional[int]:
        return int(self.pose.z * 100)

    def get_yaw(self) -> Optional[int]:
        return int(self.pose.yaw_deg)

    def send_rc_control(self, lr: int, fb: int, ud: int, yaw: int):
        self._last_cmd = (clamp(lr, -100, 100), clamp(fb, -100, 100),
                          clamp(ud, -100, 100), clamp(yaw, -100, 100))

    def get_frame_read(self) -> Any:
        raise NotImplementedError("SimTello does not provide camera frames.")

    def flip_forward(self):
        pass

    def is_flying(self) -> bool:
        return self._running


# ------------------------------ Coordinate system wrapper ------------------------------

class CoordinateTello:
    """
    Provides a local ENU coordinate frame (meters) over a Tello(-EDU) using dead-reckoning
    with optional fusion from telemetry (height, yaw). Control is done via velocity (rc).
    """
    def __init__(self,
                 driver: Optional[BaseTello] = None,
                 vel_limit_xy: float = 0.8,      # m/s
                 vel_limit_z: float  = 0.6,      # m/s
                 yaw_rate_limit: float = 60.0,   # deg/s
                 pose_filter: Optional[Callable[[Pose], Pose]] = None):
        """
        pose_filter: optional callable to refine pose (e.g., AprilTag/Pad correction).
        """
        self.drv = driver if driver else DjitelloDriver()
        self.pose = Pose()
        self.home = Pose()
        self.vel_limit_xy = vel_limit_xy
        self.vel_limit_z = vel_limit_z
        self.yaw_rate_limit = yaw_rate_limit
        self.pose_filter = pose_filter

        # PIDs for x,y,z (world frame tracking translated to body velocities)
        self.pid_x = PID(1.2, 0.0, 0.35)
        self.pid_y = PID(1.2, 0.0, 0.35)
        self.pid_z = PID(1.0, 0.0, 0.25)
        self.pid_yaw = PID(3.0, 0.0, 0.5)

        self._last_time = None
        self._running = False
        self._telemetry_thread = None
        self._is_flying = False
        self._has_height = False
        self._has_yaw = False

    # ----------- Lifecycle -----------

    def connect(self):
        self.drv.connect()
        self.drv.streamon()
        print(f"Battery: {self.drv.get_battery()}%")

    def takeoff(self):
        self.drv.takeoff()
        self._running = True
        self._last_time = time.time()
        # Reset pose at takeoff (origin)
        self.pose = Pose()
        self.home = Pose()
        self._start_telemetry()
        self._is_flying = True
        self._has_height = False
        self._has_yaw = False

    def land(self):
        self._running = False
        self.drv.stop()
        self.drv.land()
        self._is_flying = False

    def end(self):
        self._running = False
        self.drv.stop()
        self.drv.end()
        self._is_flying = False

    # ----------- Telemetry / dead-reckoning -----------

    def _start_telemetry(self):
        if self._telemetry_thread: return
        def loop():
            prev = self._last_time
            while self._running:
                now = time.time()
                dt = now - prev; prev = now
                # Pull what we can from the drone
                yaw = self.drv.get_yaw()
                if yaw is not None:
                    self.pose.yaw_deg = float(wrap_deg(yaw))
                    self._has_yaw = True
                h = self.drv.get_height()
                if h is not None:
                    self.pose.z = max(0.0, h / 100.0)
                    self._has_height = True

                # Optional external filter (e.g., AprilTag -> absolute XY)
                if self.pose_filter:
                    self.pose = self.pose_filter(self.pose)

                # (We rely on command-integration inside goto/move loops for XY.)
                time.sleep(0.05)
        self._telemetry_thread = threading.Thread(target=loop, daemon=True)
        self._telemetry_thread.start()

    # ----------- Public API -----------

    def set_home(self):
        """Reset the world-frame origin to the current pose."""
        self.home = Pose(self.pose.x, self.pose.y, self.pose.z, self.pose.yaw_deg)
        self.pose = Pose()  # shift to zero

    def get_pose(self) -> Pose:
        return Pose(self.pose.x, self.pose.y, self.pose.z, self.pose.yaw_deg)

    def is_flying(self) -> bool:
        return self._is_flying

    def get_battery(self) -> int:
        return self.drv.get_battery()

    def streamoff(self):
        self.drv.streamoff()

    def get_frame_read(self) -> Any:
        return self.drv.get_frame_read()

    def flip_forward(self):
        self.drv.flip_forward()

    def send_rc_control(self, lr: int, fb: int, ud: int, yaw: int):
        self.drv.send_rc_control(lr, fb, ud, yaw)

    def move_by(self, dx: float, dy: float, dz: float, yaw_deg: Optional[float] = None,
                pos_tol: float = 0.1, yaw_tol_deg: float = 3.0, timeout: float = 15.0):
        """Move by a relative delta in meters (world frame). Optionally also set yaw (deg)."""
        target = Pose(self.pose.x + dx, self.pose.y + dy, self.pose.z + dz,
                      self.pose.yaw_deg if yaw_deg is None else yaw_deg)
        return self._goto_impl(target, pos_tol, yaw_tol_deg, timeout)

    def dead_reckon_rc(self, lr: int, fb: int, ud: int, yaw: int, dt: float):
        """Integrate pose estimate from a manual RC command when telemetry is unavailable."""
        if not self._is_flying or dt <= 0.0:
            return

        lr = clamp(lr, -100, 100)
        fb = clamp(fb, -100, 100)
        ud = clamp(ud, -100, 100)
        yaw = clamp(yaw, -100, 100)

        vx_b_mps = fb / 100.0 * 1.0
        vy_b_mps = lr / 100.0 * 1.0
        dx_w, dy_w = rot_body_to_world(self.pose.yaw_deg, vx_b_mps * dt, vy_b_mps * dt)
        self.pose.x += dx_w
        self.pose.y += dy_w

        if not self._has_height:
            self.pose.z = max(0.0, self.pose.z + (ud / 100.0 * 0.6) * dt)

        if not self._has_yaw:
            self.pose.yaw_deg = wrap_deg(self.pose.yaw_deg + (yaw / 100.0 * 80.0) * dt)

    def goto(self, x: float, y: float, z: float, yaw_deg: Optional[float] = None,
             pos_tol: float = 0.1, yaw_tol_deg: float = 3.0, timeout: float = 20.0):
        """Go to an absolute world-frame position (meters)."""
        target = Pose(x, y, z, self.pose.yaw_deg if yaw_deg is None else yaw_deg)
        return self._goto_impl(target, pos_tol, yaw_tol_deg, timeout)

    def follow_path(self, waypoints, pos_tol: float = 0.12, yaw_tol_deg: float = 4.0, seg_timeout: float = 20.0):
        """
        waypoints: list of (x,y,z[,yaw_deg]) tuples in meters and degrees.
        """
        for wp in waypoints:
            if len(wp) == 3:
                ok = self.goto(wp[0], wp[1], wp[2], None, pos_tol, yaw_tol_deg, seg_timeout)
            else:
                ok = self.goto(wp[0], wp[1], wp[2], wp[3], pos_tol, yaw_tol_deg, seg_timeout)
            if not ok:
                return False
            time.sleep(0.3)  # small settle
        return True

    # ----------- Internals: closed-loop goto -----------

    def _goto_impl(self, target: Pose, pos_tol: float, yaw_tol_deg: float, timeout: float) -> bool:
        start_t = time.time()
        # Fresh PIDs
        self.pid_x = PID(1.2, 0.0, 0.35)
        self.pid_y = PID(1.2, 0.0, 0.35)
        self.pid_z = PID(1.0, 0.0, 0.25)
        self.pid_yaw = PID(3.0, 0.0, 0.5)

        # Control loop
        prev = time.time()
        # We integrate XY by tracking *commanded* velocity as proxy (open-loop),
        # but keep Z, yaw from telemetry when available.
        x_cmd_i = 0.0
        y_cmd_i = 0.0

        while time.time() - start_t < timeout:
            now = time.time()
            dt = now - prev
            prev = now
            if dt <= 0: dt = 1e-3

            # Position errors in world frame
            ex = target.x - self.pose.x
            ey = target.y - self.pose.y
            ez = target.z - self.pose.z
            eyaw = wrap_deg((target.yaw_deg if target else self.pose.yaw_deg) - self.pose.yaw_deg)

            # Stop if inside tolerances
            if math.hypot(ex, ey) < pos_tol and abs(ez) < pos_tol and abs(eyaw) < yaw_tol_deg:
                self.drv.stop()
                return True

            # PID world-frame velocity commands (m/s, deg/s)
            vx_w = clamp(self.pid_x.step(ex, dt), -self.vel_limit_xy, self.vel_limit_xy)
            vy_w = clamp(self.pid_y.step(ey, dt), -self.vel_limit_xy, self.vel_limit_xy)
            vz   = clamp(self.pid_z.step(ez, dt), -self.vel_limit_z,  self.vel_limit_z)
            wyaw = clamp(self.pid_yaw.step(eyaw, dt), -self.yaw_rate_limit, self.yaw_rate_limit)

            # Map to body-frame velocities for rc control
            vx_b, vy_b = rot_world_to_body(self.pose.yaw_deg, vx_w, vy_w)

            # Convert to RC units [-100,100]
            # Scale factors: choose conservative to match Tello dynamics
            fb = int(clamp(vx_b / 1.0 * 100, -100, 100))  # forward/back
            lr = int(clamp(vy_b / 1.0 * 100, -100, 100))  # left/right (+right)
            ud = int(clamp(vz   / 0.6 * 100, -100, 100))  # up/down
            yw = int(clamp(wyaw / 80.0 * 100, -100, 100)) # yaw rate

            self.drv.send_rc_control(lr, fb, ud, yw)

            # Dead-reckon XY by integrating **commanded** body velocities
            vx_b_mps = fb / 100.0 * 1.0      # match scaling above
            vy_b_mps = lr / 100.0 * 1.0
            dx_w, dy_w = rot_body_to_world(self.pose.yaw_deg, vx_b_mps * dt, vy_b_mps * dt)

            # Keep these small increments in case telemetry later corrects Z/Yaw
            self.pose.x += dx_w
            self.pose.y += dy_w
            # z,yaw are updated by telemetry thread; as fallback, use commands
            if self.drv.get_height() is None:
                self.pose.z = max(0.0, self.pose.z + (ud / 100.0 * 0.6) * dt)
            if self.drv.get_yaw() is None:
                self.pose.yaw_deg = wrap_deg(self.pose.yaw_deg + (yw / 100.0 * 80.0) * dt)

            time.sleep(0.03)

        # Timeout
        self.drv.stop()
        return False


# ------------------------------ Example usage ------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Local-frame coordinate control for Tello EDU")
    ap.add_argument("--sim", action="store_true", help="Run with simulated Tello")
    ap.add_argument("--fly", action="store_true", help="Actually take off (omit to dry-run)")
    args = ap.parse_args()

    driver = SimTello() if args.sim else DjitelloDriver()
    ctl = CoordinateTello(driver=driver)

    try:
        ctl.connect()
        if args.fly:
            ctl.takeoff()
            print("Taking off...")

            # Go to (x=0.5m East, y=1.0m North, z=1.0m Up)
            print("Goto (0.5, 1.0, 1.0)")
            ctl.goto(0.5, 1.0, 1.0, yaw_deg=0.0, timeout=25.0)

            # Move by -0.5m East, +0.0m North, -0.2m Up
            print("Move_by (-0.5, 0.0, -0.2)")
            ctl.move_by(-0.5, 0.0, -0.2, timeout=15.0)

            # Face 90 deg (East) and hold
            ctl.goto(ctl.pose.x, ctl.pose.y, ctl.pose.z, yaw_deg=90.0, timeout=10.0)

            # Return near home
            print("Return near home")
            ctl.goto(0.0, 0.2, 0.7, yaw_deg=0.0, timeout=25.0)

            ctl.land()
        else:
            print("Connected. Dry-run complete (use --fly to take off).")
    except KeyboardInterrupt:
        print("CTRL-C: landing...")
        try:
            ctl.land()
        except Exception:
            pass
        ctl.end()
