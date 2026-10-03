"""Live PaveWatch view: IP Webcam video (rotated) with IMU overlay and shock alerts, optional recording.

Usage:
    python live_view.py
    python live_view.py --ip 192.168.1.42
    python live_view.py --record data/pass1     # saves video.mp4, imu.csv, frames.csv, events.jsonl
Press q to quit.
"""
import argparse
import csv
import json
import os
import time

import cv2

import config
from pavewatch import IMUStream, VideoStream
from pavewatch.shock import ShockDetector


def draw_imu(frame, readings):
    y = 30
    for name, (_, v) in readings.items():
        text = f"{name:9s} " + "  ".join(f"{x:+7.2f}" for x in v[:3])
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        y += 28


SEVERITY_COLORS = {"low": (0, 255, 255), "medium": (0, 165, 255), "high": (0, 0, 255)}  # BGR


def draw_shocks(frame, shocks, banner_s=1.5):
    """Flash a coloured border + banner for a fresh shock, and list the last few at the bottom."""
    h, w = frame.shape[:2]
    if shocks and time.time() - shocks[-1][0] < banner_s:
        event = shocks[-1][1]
        color = SEVERITY_COLORS[event["severity"]]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 12)
        text = f"SHOCK {event['severity'].upper()}  {event['peak']:.1f} m/s2"
        cv2.putText(frame, text, (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
        cv2.putText(frame, text, (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    y = h - 15
    for shown, event in reversed(shocks[-3:]):
        text = f"{time.strftime('%H:%M:%S', time.localtime(shown))}  {event['severity']:6s} {event['peak']:.1f}"
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, SEVERITY_COLORS[event["severity"]], 1)
        y -= 24


def main():
    parser = argparse.ArgumentParser(description="PaveWatch live video + IMU viewer")
    parser.add_argument("--ip", default=config.PHONE_IP, help="Phone IP shown in IP Webcam")
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument("--no-rotate", action="store_true", help="Show the image as the phone sends it")
    parser.add_argument("--record", metavar="DIR", help="Folder to save video.mp4, imu.csv and frames.csv")
    args = parser.parse_args()

    base = f"http://{args.ip}:{args.port}"
    video = VideoStream(f"{base}/video").start()
    imu = IMUStream(base, config.SENSORS).start()

    writer = imu_csv = frames_csv = None
    files = []
    if args.record:
        os.makedirs(args.record, exist_ok=True)
        files = [open(os.path.join(args.record, n), "w", newline="") for n in ("imu.csv", "frames.csv")]
        imu_csv, frames_csv = (csv.writer(f) for f in files)
        imu_csv.writerow(["sensor", "phone_ts_ms", "x", "y", "z", "pc_time"])
        frames_csv.writerow(["frame", "pc_time"])
        print(f"Recording to {args.record}")

    detector = ShockDetector(config.SHOCK_THRESHOLD, config.SEVERITY_BANDS,
                             config.SHOCK_WINDOW_S, config.SHOCK_REFRACTORY_S)
    shocks = []  # (pc_time shown, event)
    events_out = open(os.path.join(args.record, "events.jsonl"), "a") if args.record else None

    last_t = 0.0
    n_frames = 0
    try:
        while True:
            for name, ts, v, pc_time in imu.drain():
                if imu_csv:
                    imu_csv.writerow([name, ts, *v[:3], f"{pc_time:.3f}"])
                if name == "lin_accel":
                    event = detector.feed(ts, v)
                    if event:
                        print(f"SHOCK  peak={event['peak']:.2f} m/s^2  {event['severity'].upper()}")
                        shocks.append((time.time(), event))
                        if events_out:
                            events_out.write(json.dumps(event) + "\n")
                            events_out.flush()

            frame, t = video.read()
            if frame is None or t == last_t:
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
                continue
            last_t = t

            if config.ROTATE_CLOCKWISE and not args.no_rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            if args.record:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(os.path.join(args.record, "video.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), config.RECORD_FPS, (w, h))
                writer.write(frame)
                frames_csv.writerow([n_frames, f"{t:.3f}"])
                n_frames += 1

            draw_imu(frame, imu.snapshot())
            draw_shocks(frame, shocks)
            if args.record:
                cv2.circle(frame, (frame.shape[1] - 25, 25), 10, (0, 0, 255), -1)
            cv2.imshow("PaveWatch", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        video.stop()
        imu.stop()
        if writer is not None:
            writer.release()
        for f in files:
            f.close()
        if events_out:
            events_out.close()
        cv2.destroyAllWindows()
        if args.record:
            print(f"Saved {n_frames} frames to {args.record}")


if __name__ == "__main__":
    main()
