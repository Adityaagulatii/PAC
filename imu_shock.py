"""Standalone IMU stage: detect shocks and grade severity, live or from a recording.

Usage:
    python imu_shock.py                              # live from the phone (IP Webcam)
    python imu_shock.py --csv data/pass1/imu.csv     # replay a phyphox export
    python imu_shock.py --calibrate                  # print magnitude stats to pick a threshold
    python imu_shock.py --out events.jsonl           # also append events to a file
Ctrl+C to stop live mode.
"""
import argparse
import json
import math
import time

import config
from pavewatch.shock import ShockDetector, read_phyphox_csv


def live_samples(ip, port):
    from pavewatch.ipcam import IMUStream

    imu = IMUStream(f"http://{ip}:{port}", sensors=("lin_accel",)).start()
    print(f"Listening to lin_accel from {ip}:{port} ...")
    try:
        while True:
            for name, ts, values, _ in imu.drain():
                yield ts, values
            time.sleep(0.05)
    finally:
        imu.stop()


def calibrate(samples):
    mags, last_print, t0 = [], time.time(), None
    try:
        for t_ms, xyz in samples:
            t0 = t_ms if t0 is None else t0
            mags.append(math.sqrt(sum(v * v for v in xyz[:3])))
            if time.time() - last_print >= 1.0:
                print(f"{(t_ms - t0) / 1000:6.1f}s  last-second max {max(mags[-20:]):.2f}")
                last_print = time.time()
    except KeyboardInterrupt:
        pass
    if not mags:
        print("No samples received.")
        return
    s = sorted(mags)
    pct = lambda p: s[min(len(s) - 1, int(p / 100 * len(s)))]
    print(f"\n{len(s)} samples  median {pct(50):.2f}  p95 {pct(95):.2f}  "
          f"p99 {pct(99):.2f}  max {s[-1]:.2f}  (m/s^2)")
    print("Smooth riding: set SHOCK_THRESHOLD a bit above p99. Spike test: use max to set the severity bands.")


def main():
    parser = argparse.ArgumentParser(description="PaveWatch IMU shock detector")
    parser.add_argument("--ip", default=config.PHONE_IP)
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument("--csv", help="phyphox export to replay instead of the live phone")
    parser.add_argument("--threshold", type=float, default=config.SHOCK_THRESHOLD)
    parser.add_argument("--calibrate", action="store_true", help="print magnitude stats, no detection")
    parser.add_argument("--out", help="append events as JSON lines to this file")
    args = parser.parse_args()

    samples = read_phyphox_csv(args.csv) if args.csv else live_samples(args.ip, args.port)
    if args.calibrate:
        calibrate(samples)
        return

    detector = ShockDetector(args.threshold, config.SEVERITY_BANDS,
                             config.SHOCK_WINDOW_S, config.SHOCK_REFRACTORY_S)
    out = open(args.out, "a") if args.out else None
    count = 0

    def report(event):
        nonlocal count
        count += 1
        print(f"SHOCK  t={event['t_ms'] / 1000:.3f}s  peak={event['peak']:.2f} m/s^2  {event['severity'].upper()}")
        if out:
            out.write(json.dumps(event) + "\n")
            out.flush()

    try:
        for t_ms, xyz in samples:
            event = detector.feed(t_ms, xyz)
            if event:
                report(event)
    except KeyboardInterrupt:
        pass
    finally:
        event = detector.flush()
        if event:
            report(event)
        if out:
            out.close()
        print(f"{count} shock(s) detected.")


if __name__ == "__main__":
    main()
