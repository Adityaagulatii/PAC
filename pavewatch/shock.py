"""Shock detection and severity grading from IMU samples (the "Feel" stage)."""
import csv
import math


def severity(peak, bands):
    """bands = (medium_from, high_from) in m/s^2; anything below medium_from is low."""
    medium_from, high_from = bands
    if peak >= high_from:
        return "high"
    if peak >= medium_from:
        return "medium"
    return "low"


class ShockDetector:
    """Feed linear-acceleration samples; get one event per shock.

    A shock starts when |lin_accel| crosses `threshold`. The peak is tracked for
    `window_s`, then the event is emitted and new triggers are ignored until
    `refractory_s` after the start, so one bump gives one event.
    """

    def __init__(self, threshold, bands, window_s=0.2, refractory_s=0.5):
        self.threshold = threshold
        self.bands = bands
        self.window_ms = window_s * 1000
        self.refractory_ms = refractory_s * 1000
        self.start_ms = None
        self.peak = 0.0
        self.peak_ms = 0
        self.quiet_until_ms = -math.inf

    def feed(self, t_ms, xyz):
        """Process one sample. Returns an event dict when a shock window closes, else None."""
        mag = math.sqrt(sum(v * v for v in xyz[:3]))
        event = None

        if self.start_ms is not None:
            if mag > self.peak:
                self.peak, self.peak_ms = mag, t_ms
            if t_ms - self.start_ms >= self.window_ms:
                event = self._emit()

        if self.start_ms is None and mag >= self.threshold and t_ms >= self.quiet_until_ms:
            self.start_ms = t_ms
            self.peak, self.peak_ms = mag, t_ms
            self.quiet_until_ms = t_ms + self.refractory_ms
        return event

    def flush(self):
        """Emit a shock still in progress (call at end of a recording)."""
        return self._emit() if self.start_ms is not None else None

    def _emit(self):
        event = {
            "t_ms": self.peak_ms,
            "start_ms": self.start_ms,
            "peak": round(self.peak, 3),
            "severity": severity(self.peak, self.bands),
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


def read_phyphox_csv(path):
    """Yield (t_ms, [x, y, z]) linear acceleration from a phyphox export.

    Accepts a "Linear Acceleration" export directly, or an "Acceleration with g"
    export, in which case gravity is removed here.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        has_gravity = not any("linear" in h for h in header[1:4])
        remove_g = GravityRemover() if has_gravity else None
        for row in reader:
            if len(row) < 4 or not row[0].strip():
                continue
            t_ms = float(row[0]) * 1000
            xyz = [float(v) for v in row[1:4]]
            yield t_ms, remove_g(xyz) if remove_g else xyz
