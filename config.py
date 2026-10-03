"""Default settings. Override the IP per run with --ip or the PHONE_IP environment variable."""
import os

PHONE_IP = os.environ.get("PHONE_IP", "172.20.65.194")
PORT = 8080
SENSORS = ("accel", "gyro", "lin_accel")
ROTATE_CLOCKWISE = True   # phone is mounted so the image arrives 90 degrees off
RECORD_FPS = 15           # matches the 15 fps IP Webcam setting in the runbook

# Shock detection. Set from manual bumps on 2026-10-03 (jolts 2-5.7 m/s^2, twists 2-3.2 rad/s at ~15 Hz);
# re-tune on the carrier with imu_shock.py --calibrate.
SHOCK_THRESHOLD = 3.0         # |lin_accel| in m/s^2 that starts a shock (the jolt)
SEVERITY_BANDS = (4.0, 5.5)   # lin_accel: medium from, high from
GYRO_THRESHOLD = 2.0          # |gyro| in rad/s that starts a shock (the twist)
GYRO_BANDS = (3.0, 4.0)       # gyro: medium from, high from
SHOCK_WINDOW_S = 0.3          # how long to track the peak after a trigger (~4 samples at 15 Hz)
SHOCK_REFRACTORY_S = 1.0      # ignore new triggers for this long after one starts (skips the rebound)

# Pothole detector: DityaEn YOLO11 weights, copied from the USB (models/pothole-dityaen-yolo11/my_model.pt)
DETECTOR_WEIGHTS = os.path.join(os.path.dirname(__file__), "models", "pothole_yolo11.pt")
DETECT_CONF = 0.4
DETECT_IMGSZ = 640
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")  # live_view.py writes potholes.csv + impacts.csv here
POTHOLE_SIZE_BANDS = (0.03, 0.10)  # share of the frame the box covers: medium from 3%, large from 10%
