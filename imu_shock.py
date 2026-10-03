"""Standalone IMU stage: detect shocks and grade impact (low/medium/high), live or from a recording.

Impact uses both the jolt (|lin_accel|, m/s^2) and the twist (|gyro|, rad/s); the worse one wins.

Usage:
    python imu_shock.py                                   # live from the phone (IP Webcam)
    python imu_shock.py --csv accel.csv --gyro-csv gyro.csv   # replay phyphox exports
    python imu_shock.py --calibrate                       # print stats to pick thresholds
    python imu_shock.py --out events.jsonl                # also append events to a file
Ctrl+C to stop live mode.
"""
import argparse
import json
import time

import config
from pavewatch.shock import ShockDetector, magnitude, read_phyphox_csv

UNITS = {"lin_accel": "m/s^2", "gyro": "rad/s"}


def live_samples(ip, port):
    from pavewatch.ipcam import IMUStream

    imu = IMUStream(f"http://{ip}:{port}", sensors=("lin_accel", "gyro")).start()
    print(f"Listening to lin_accel + gyro from {ip}:{port} ...")
    try:
        while True:
            for name, ts, values, _ in imu.drain():
                yield ts, name, values
            time.sleep(0.05)
    finally:
        imu.stop()


def calibrate(samples):
    mags = {"lin_accel": [], "gyro": []}
    last_print, t0 = time.time(), None
    try:
        for t_ms, sensor, xyz in samples:
            t0 = t_ms if t0 is None else t0
            mags[sensor].append(magnitude(xyz))
            if time.time() - last_print >= 1.0:
                parts = [f"{s} max {max(m[-20:]):.2f}" for s, m in mags.items() if m]
                print(f"{(t_ms - t0) / 1000:6.1f}s  last-second  " + "   ".join(parts))
                last_print = time.time()
    except KeyboardInterrupt:
        pass
    print()
    for sensor, m in mags.items():
        if not m:
            print(f"{sensor}: no samples")
            continue
        s = sorted(m)
        pct = lambda p: s[min(len(s) - 1, int(p / 100 * len(s)))]
        print(f"{sensor:9s} {len(s)} samples  median {pct(50):.2f}  p95 {pct(95):.2f}  "
              f"p99 {pct(99):.2f}  max {s[-1]:.2f}  ({UNITS[sensor]})")
    print("Smooth riding: set SHOCK_THRESHOLD / GYRO_THRESHOLD a bit above p99.\n"
          "Spike test: use max to set SEVERITY_BANDS / GYRO_BANDS.")


def format_event(event):
    return (f"SHOCK {event['severity'].upper():6s}  jolt {event['peak']:.2f} m/s^2 ({event['accel_severity']})  "
            f"twist {event['gyro_peak']:.2f} rad/s ({event['gyro_severity']})  trigger={event['trigger']}")


def main():
    parser = argparse.ArgumentParser(description="PaveWatch IMU shock detector")
    parser.add_argument("--ip", default=config.PHONE_IP)
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument("--csv", help="phyphox acceleration export to replay instead of the live phone")
    parser.add_argument("--gyro-csv", help="phyphox gyroscope export to replay alongside --csv")
    parser.add_argument("--threshold", type=float, default=config.SHOCK_THRESHOLD)
    parser.add_argument("--gyro-threshold", type=float, default=config.GYRO_THRESHOLD)
    parser.add_argument("--calibrate", action="store_true", help="print magnitude stats, no detection")
    parser.add_argument("--out", help="append events as JSON lines to this file")
    args = parser.parse_args()

    if args.csv:
        samples = read_phyphox_csv(args.csv, args.gyro_csv)
    else:
        samples = live_samples(args.ip, args.port)
    if args.calibrate:
        calibrate(samples)
        return

    detector = ShockDetector(args.threshold, config.SEVERITY_BANDS,
                             args.gyro_threshold, config.GYRO_BANDS,
                             config.SHOCK_WINDOW_S, config.SHOCK_REFRACTORY_S)
    out = open(args.out, "a") if args.out else None
    count = 0

    def report(event):
        nonlocal count
        count += 1
        print(f"t={event['t_ms'] / 1000:.3f}s  " + format_event(event))
        if out:
            out.write(json.dumps(event) + "\n")
            out.flush()

    try:
        for t_ms, sensor, xyz in samples:
            event = detector.feed(t_ms, sensor, xyz)
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
