"""Default settings. Override the IP per run with --ip or the PHONE_IP environment variable."""
import os

PHONE_IP = os.environ.get("PHONE_IP", "172.20.65.194")
PORT = 8080
SENSORS = ("accel", "gyro", "lin_accel")
ROTATE_CLOCKWISE = True   # phone is mounted so the image arrives 90 degrees off
RECORD_FPS = 15           # matches the 15 fps IP Webcam setting in the runbook

# Shock detection on |lin_accel| in m/s^2. Placeholders: set these from imu_shock.py --calibrate.
SHOCK_THRESHOLD = 3.0
SEVERITY_BANDS = (6.0, 10.0)  # medium from, high from
SHOCK_WINDOW_S = 0.2          # how long to track the peak after a trigger
SHOCK_REFRACTORY_S = 0.5      # ignore new triggers for this long after one starts
