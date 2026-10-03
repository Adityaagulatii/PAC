"""Simple OpenCV viewer for the Android IP Webcam app, with live IMU overlay.

Usage:
    python ipcam_simple.py              # uses the default IP below
    python ipcam_simple.py 192.168.1.42
Press q to quit.
"""
import sys
import threading
import time

import cv2
import requests

ip = sys.argv[1] if len(sys.argv) > 1 else "172.20.65.194"  # change to your phone's IP
base = f"http://{ip}:8080"
SENSORS = "accel,gyro,lin_accel"

# Latest reading per sensor: {"accel": (timestamp_ms, [x, y, z]), ...}
imu = {}
running = True


def poll_imu():
    """Poll sensors.json in the background, asking only for samples newer than the last one."""
    last_ts = 0
    while running:
        try:
            r = requests.get(f"{base}/sensors.json",
                             params={"sense": SENSORS, "from": last_ts}, timeout=2)
            for name, sensor in r.json().items():
                if sensor["data"]:
                    ts, values = sensor["data"][-1]
                    imu[name] = (ts, values)
                    last_ts = max(last_ts, ts)
                    print(name, ts, [round(v, 3) for v in values])
        except (requests.RequestException, ValueError) as e:
            print("IMU poll failed:", e)
            time.sleep(1)
        time.sleep(0.05)


threading.Thread(target=poll_imu, daemon=True).start()

cap = cv2.VideoCapture(f"{base}/video")
if not cap.isOpened():
    sys.exit(f"Could not open {base}/video - check the IP and that both devices are on the same Wi-Fi.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame grab failed, stopping.")
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    y = 30
    for name, (ts, v) in list(imu.items()):
        text = f"{name:9s} " + "  ".join(f"{x:+7.2f}" for x in v[:3])
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        y += 28

    cv2.imshow("IP Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

running = False
cap.release()
cv2.destroyAllWindows()
