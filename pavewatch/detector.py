"""Pothole detector (YOLO11, DityaEn weights) that runs on the newest frame in a background thread."""
import threading
import time

from ultralytics import YOLO


class PotholeDetector:
    """Call submit(frame) as often as you like; the detector always works on the newest one.

    Detections are in the coordinates of the submitted frame:
    [{"box": (x1, y1, x2, y2), "conf": 0.87, "label": "pothole",
      "area_frac": 0.06, "size": "medium"}, ...]

    Size is graded from the share of the frame the box covers (`size_bands` =
    (medium_from, large_from)). It's an apparent size: the same pothole looks
    smaller further away, so tune the bands for the camera mount.
    """

    def __init__(self, weights, conf=0.4, imgsz=640, size_bands=(0.03, 0.10), on_result=None):
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.size_bands = size_bands
        self.on_result = on_result  # called as on_result(detections, pc_time) after each background run
        self.frame = None
        self.detections = []
        self.infer_ms = 0.0
        self.running = False
        self.lock = threading.Lock()
        self.new_frame = threading.Event()

    def start(self):
        self.running = True
        threading.Thread(target=self._worker, daemon=True).start()
        return self

    def submit(self, frame):
        with self.lock:
            self.frame = frame
        self.new_frame.set()

    def latest(self):
        with self.lock:
            return list(self.detections), self.infer_ms

    def detect(self, frame):
        """Run once, synchronously, on one frame."""
        result = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        h, w = frame.shape[:2]
        detections = []
        for b in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            area_frac = (x2 - x1) * (y2 - y1) / (w * h)
            detections.append({
                "box": (x1, y1, x2, y2),
                "conf": round(float(b.conf[0]), 3),
                "label": self.model.names[int(b.cls[0])],
                "area_frac": round(area_frac, 4),
                "size": self.size(area_frac),
            })
        return detections

    def size(self, area_frac):
        medium_from, large_from = self.size_bands
        if area_frac >= large_from:
            return "large"
        if area_frac >= medium_from:
            return "medium"
        return "small"

    def _worker(self):
        while self.running:
            if not self.new_frame.wait(timeout=0.5):
                continue
            self.new_frame.clear()
            with self.lock:
                frame, self.frame = self.frame, None
            if frame is None:
                continue
            t0 = time.time()
            dets = self.detect(frame)
            with self.lock:
                self.detections = dets
                self.infer_ms = (time.time() - t0) * 1000
            if self.on_result and dets:
                self.on_result(dets, t0)

    def stop(self):
        self.running = False
        self.new_frame.set()
