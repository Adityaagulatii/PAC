"""Video and IMU readers for the Android IP Webcam app."""
import threading
import time

import cv2
import requests


class VideoStream:
    """Reads frames in a background thread, keeps only the latest one, and reconnects on drops."""

    def __init__(self, url):
        self.url = url
        self.frame = None
        self.frame_time = 0.0
        self.running = False
        self.lock = threading.Lock()
        self.cap = None

    def start(self):
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()
        return self

    def _reader(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                print(f"Connecting to {self.url} ...")
                self.cap = cv2.VideoCapture(self.url)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not self.cap.isOpened():
                    time.sleep(1.0)
                    continue
                print("Video connected.")

            ok, frame = self.cap.read()
            if not ok:
                print("Video dropped, reconnecting ...")
                self.cap.release()
                self.cap = None
                time.sleep(0.5)
                continue

            with self.lock:
                self.frame = frame
                self.frame_time = time.time()

    def read(self):
        """Return (frame, pc_time) of the newest frame, or (None, 0.0) before the first one."""
        with self.lock:
            if self.frame is None:
                return None, 0.0
            return self.frame.copy(), self.frame_time

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()


class IMUStream:
    """Polls sensors.json in a background thread, fetching only samples newer than the last one."""

    def __init__(self, base_url, sensors=("accel", "gyro", "lin_accel"), poll_interval=0.05):
        self.url = f"{base_url}/sensors.json"
        self.sensors = ",".join(sensors)
        self.poll_interval = poll_interval
        self.latest = {}    # sensor -> (phone_ts_ms, [x, y, z])
        self.pending = []   # (sensor, phone_ts_ms, values, pc_time) not yet drained
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        threading.Thread(target=self._poller, daemon=True).start()
        return self

    def _poller(self):
        last = {}  # sensor -> newest phone timestamp seen
        while self.running:
            try:
                since = min(last.values()) if last else 0
                r = requests.get(self.url, params={"sense": self.sensors, "from": since}, timeout=2)
                now = time.time()
                with self.lock:
                    for name, sensor in r.json().items():
                        for ts, values in sensor["data"]:
                            if ts <= last.get(name, 0):
                                continue
                            self.pending.append((name, ts, values, now))
                            self.latest[name] = (ts, values)
                            last[name] = ts
            except (requests.RequestException, ValueError, KeyError) as e:
                print("IMU poll failed:", e)
                time.sleep(1.0)
            time.sleep(self.poll_interval)

    def snapshot(self):
        """Latest reading per sensor."""
        with self.lock:
            return dict(self.latest)

    def drain(self):
        """Return and clear all samples received since the last drain, in phone-time order."""
        with self.lock:
            out, self.pending = self.pending, []
        return sorted(out, key=lambda s: s[1])

    def stop(self):
        self.running = False
