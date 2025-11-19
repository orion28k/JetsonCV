from djitellopy import Tello
import cv2
import time
import math

tello = Tello()
tello.connect()
print("Battery Life:" + str(tello.get_battery()))

# Initialize coordinates and yaw
coords = [0.0, 0.0, 0.0]  # x, y, z
yaw = 0.0  # in degrees

def update_coords(coords, lr, fb, ud, yaw, speed_per_command=1):
    yaw_rad = math.radians(yaw)
    dx = lr * math.cos(yaw_rad) - fb * math.sin(yaw_rad)
    dy = lr * math.sin(yaw_rad) + fb * math.cos(yaw_rad)
    dz = ud
    coords[0] += dx * speed_per_command
    coords[1] += dy * speed_per_command
    coords[2] += dz * speed_per_command
    return coords


def move_to_coords(drone, current_coords, current_yaw, target_coords=[0, 0, 0, 0], speed=50, tolerance=10):
    """
    Move the drone to target_coords (x, y, z, yaw) relative to takeoff.
    Assumes current_coords and target_coords are [x, y, z, yaw].
    """
    dx = target_coords[0] - current_coords[0]
    dy = target_coords[1] - current_coords[1]
    dz = target_coords[2] - current_coords[2]
    dyaw = target_coords[3] - current_yaw

    # Calculate distance and yaw difference
    distance = math.sqrt(dx**2 + dy**2 + dz**2)
    while distance > tolerance or abs(dyaw) > 2:  # 2 degrees tolerance for yaw
        # Calculate movement in drone's frame
        yaw_rad = math.radians(current_yaw)
        # Transform global dx/dy to drone's local frame
        fb = int(dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad))
        lr = int(-dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad))
        ud = int(dz)
        # Clamp speed
        fb = max(-speed, min(speed, fb))
        lr = max(-speed, min(speed, lr))
        ud = max(-speed, min(speed, ud))
        # Yaw velocity: positive for right, negative for left
        yv = 0
        if abs(dyaw) > 2:
            yv = int(max(-speed, min(speed, dyaw)))
        drone.send_rc_control(lr, fb, ud, yv)
        time.sleep(0.1)
        # Estimate new position and yaw
        current_coords[0] += lr * 0.1
        current_coords[1] += fb * 0.1
        current_coords[2] += ud * 0.1
        current_yaw += yv * 0.1  # You may need to calibrate this for real yaw rate
        dx = target_coords[0] - current_coords[0]
        dy = target_coords[1] - current_coords[1]
        dz = target_coords[2] - current_coords[2]
        dyaw = target_coords[3] - current_yaw
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
    drone.send_rc_control(0, 0, 0, 0)

tello.takeoff()

# Example movement: move forward, then rotate, then move right
tello.send_rc_control(0, 50, 0, 0)  # Move forward
coords = update_coords(coords, 0, 50, 0, yaw, speed_per_command=1)
print(f"Coords after forward: {coords}")

time.sleep(1)
tello.send_rc_control(0, 0, 0, -50)  # Rotate left
yaw += 50 * 1  # Estimate yaw change (degrees per second * seconds)
print(f"Yaw after rotation: {yaw}")

time.sleep(1)
tello.send_rc_control(-50, 0, 0, 0)  # Move right
coords = update_coords(coords, 50, 0, 0, yaw, speed_per_command=1)
print(f"Coords after right: {coords}")

move_to_coords(tello, coords, yaw, target_coords=[0, 0, 65, 0], speed=30)

time.sleep(1)
tello.send_rc_control(0, 0, 0, 0)  # Stop

tello.land()
tello.streamoff()
tello.end()
cv2.destroyAllWindows()