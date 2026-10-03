"""Shock detection and impact grading from IMU samples (the "Feel" stage).

Two signals are used:
  lin_accel  |a| in m/s^2   - the jolt (vertical hit when a wheel drops in or hits an edge)
  gyro       |w| in rad/s   - the twist (pitch/roll when one side drops)
Each one is graded low/medium/high on its own bands; the impact is the worse of the two.
"""
import csv
import math
from collections import deque

LEVELS = ("low", "medium", "high")


def severity(peak, bands):
    """bands = (medium_from, high_from); anything below medium_from is low."""
    medium_from, high_from = bands
    if peak >= high_from:
        return "high"
    if peak >= medium_from:
        return "medium"
    return "low"


def magnitude(xyz):
    return math.sqrt(sum(v * v for v in xyz[:3]))


class ShockDetector:
    """Feed lin_accel and gyro samples; get one event per shock.

    A shock starts when |lin_accel| crosses `accel_threshold` or |gyro| crosses
    `gyro_threshold`. Both peaks are tracked for `window_s`, then the event is
    emitted and new triggers are ignored until `refractory_s` after the start,
    so one bump gives one event.
    """

    def __init__(self, accel_threshold, accel_bands, gyro_threshold, gyro_bands,
                 window_s=0.2, refractory_s=0.5):
        self.thresholds = {"lin_accel": accel_threshold, "gyro": gyro_threshold}
        self.bands = {"lin_accel": accel_bands, "gyro": gyro_bands}
        self.window_ms = window_s * 1000
        self.refractory_ms = refractory_s * 1000
        self.start_ms = None
        self.trigger = None
        self.peaks = {}
        self.peak_ms = 0
        self.quiet_until_ms = -math.inf
        self.recent = {s: deque() for s in self.thresholds}  # (t_ms, mag) within the last window

    def feed(self, t_ms, sensor, xyz):
        """Process one sample of `sensor` ("lin_accel" or "gyro").

        Returns an event dict when a shock window closes, else None.
        """
        if sensor not in self.thresholds:
            return None
        mag = magnitude(xyz)
        event = None

        recent = self.recent[sensor]
        recent.append((t_ms, mag))
        while recent and t_ms - recent[0][0] > self.window_ms:
            recent.popleft()

        if self.start_ms is not None:
            if t_ms - self.start_ms >= self.window_ms:
                event = self._emit()
            else:
                self._update_peak(sensor, mag, t_ms)

        if (self.start_ms is None and mag >= self.thresholds[sensor]
                and t_ms >= self.quiet_until_ms):
            self.start_ms = t_ms
            self.trigger = sensor
            self.peaks = {}
            self.peak_ms = t_ms
            # The other sensor's samples from this same moment may already have gone by
            # (the phone sends each sensor in its own batch), so include its recent ones.
            for other, samples in self.recent.items():
                for t, m in samples:
                    if other != sensor and t_ms - t <= self.window_ms / 2:
                        self._update_peak(other, m, t)
            self._update_peak(sensor, mag, t_ms)
            self.quiet_until_ms = t_ms + self.refractory_ms
        return event

    def _update_peak(self, sensor, mag, t_ms):
        if mag > self.peaks.get(sensor, 0.0):
            self.peaks[sensor] = mag
            if sensor == self.trigger:
                self.peak_ms = t_ms

    def flush(self):
        """Emit a shock still in progress (call at end of a recording)."""
        return self._emit() if self.start_ms is not None else None

    def _emit(self):
        accel = self.peaks.get("lin_accel", 0.0)
        gyro = self.peaks.get("gyro", 0.0)
        accel_sev = severity(accel, self.bands["lin_accel"])
        gyro_sev = severity(gyro, self.bands["gyro"])
        event = {
            "t_ms": self.peak_ms,
            "start_ms": self.start_ms,
            "trigger": self.trigger,
            "peak": round(accel, 3),          # m/s^2
            "gyro_peak": round(gyro, 3),      # rad/s
            "accel_severity": accel_sev,
            "gyro_severity": gyro_sev,
            "severity": max(accel_sev, gyro_sev, key=LEVELS.index),
        }
        self.start_ms = None
        return event


class GravityRemover:
    """Turns raw acceleration (with gravity) into linear acceleration with a low-pass gravity estimate."""

    def __init__(self, alpha=0.02):
        self.alpha = alpha
        self.g = None

    def __call__(self, xyz):
        if self.g is None:
            self.g = list(xyz[:3])
        self.g = [g + self.alpha * (v - g) for g, v in zip(self.g, xyz)]
        return [v - g for v, g in zip(xyz, self.g)]


def _read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        rows = []
        for row in reader:
            if len(row) < 4 or not row[0].strip():
                continue
            rows.append((float(row[0]) * 1000, [float(v) for v in row[1:4]]))
    return header, rows


def read_phyphox_csv(path, gyro_path=None):
    """Yield (t_ms, sensor, [x, y, z]) from phyphox exports, in time order.

    `path` is a "Linear Acceleration" export, or an "Acceleration with g" export
    (gravity is removed here). `gyro_path` is an optional "Gyroscope" export.
    """
    header, rows = _read_csv(path)
    if not any("linear" in h for h in header[1:4]):
        remove_g = GravityRemover()
        rows = [(t, remove_g(xyz)) for t, xyz in rows]
    samples = [(t, "lin_accel", xyz) for t, xyz in rows]

    if gyro_path:
        _, gyro_rows = _read_csv(gyro_path)
        samples += [(t, "gyro", xyz) for t, xyz in gyro_rows]
        samples.sort(key=lambda s: s[0])
    yield from samples
