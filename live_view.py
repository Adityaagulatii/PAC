"""Live PaveWatch view: IP Webcam video (rotated) with YOLO pothole boxes, IMU overlay and shock alerts,
optional recording.

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
from imu_shock import format_event
from pavewatch.shock import ShockDetector


def label(frame, text, org, color, scale=0.6, thick=1):
    """Coloured text on a dark box, readable on any background."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(frame, (x - 4, y - th - 5), (x + tw + 4, y + base + 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def draw_imu(frame, readings):
    y = 30
    if not readings:
        label(frame, "IMU: no sensor data from phone", (10, y), (0, 0, 255))
    for name, (_, v) in readings.items():
        label(frame, f"{name:9s} " + "  ".join(f"{x:+7.2f}" for x in v[:3]), (10, y), (0, 255, 0))
        y += 28


SEVERITY_COLORS = {"low": (0, 255, 255), "medium": (0, 165, 255), "high": (0, 0, 255)}  # BGR


def draw_shocks(frame, shocks, banner_s=3.0):
    """Flash a coloured border + banner for a fresh shock, and list the last few at the bottom."""
    h, w = frame.shape[:2]
    if shocks and time.time() - shocks[-1][0] < banner_s:
        event = shocks[-1][1]
        color = SEVERITY_COLORS[event["severity"]]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 12)
        lines = [
            (f"IMPACT {event['severity'].upper()}", 1.2, 3),
            (f"jolt  {event['peak']:.1f} m/s2  {event['accel_severity']}", 0.7, 2),
            (f"twist {event['gyro_peak']:.1f} rad/s  {event['gyro_severity']}", 0.7, 2),
        ]
        y = h // 2
        for text, scale, thick in lines:
            label(frame, text, (20, y), color, scale, thick)
            y += int(50 * scale)

    y = h - 15
    for shown, event in reversed(shocks[-3:]):
        text = (f"{time.strftime('%H:%M:%S', time.localtime(shown))}  {event['severity']:6s} "
                f"jolt {event['peak']:.1f}  twist {event['gyro_peak']:.1f}")
        label(frame, text, (10, y), SEVERITY_COLORS[event["severity"]], 0.55)
        y -= 26


SIZE_COLORS = {"small": (0, 255, 255), "medium": (0, 165, 255), "large": (0, 0, 255)}  # BGR


def draw_potholes(frame, detections, infer_ms):
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color = SIZE_COLORS[d["size"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        text = f"{d['size'].upper()} {d['label']} {d['conf']:.2f}"
        label(frame, text, (x1 + 4, max(24, y1 - 10)), color, 0.7, 2)
    if infer_ms:
        sizes = ", ".join(f"{d['size']}" for d in detections) or "none"
        label(frame, f"YOLO {infer_ms:.0f} ms  potholes: {sizes}", (10, 120), (255, 0, 255))


def main():
    parser = argparse.ArgumentParser(description="PaveWatch live video + IMU viewer")
    parser.add_argument("--ip", default=config.PHONE_IP, help="Phone IP shown in IP Webcam")
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument("--no-rotate", action="store_true", help="Show the image as the phone sends it")
    parser.add_argument("--record", metavar="DIR", help="Folder to save video.mp4, imu.csv and frames.csv")
    parser.add_argument("--no-detect", action="store_true", help="Turn off the YOLO pothole detector")
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

    shock_detector = ShockDetector(config.SHOCK_THRESHOLD, config.SEVERITY_BANDS,
                                   config.GYRO_THRESHOLD, config.GYRO_BANDS,
                                   config.SHOCK_WINDOW_S, config.SHOCK_REFRACTORY_S)
    shocks = []  # (pc_time shown, event)

    # Every run logs pothole sizes and impacts to logs/<date_time>/
    log_dir = os.path.join(config.LOG_DIR, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    pothole_log_f = open(os.path.join(log_dir, "potholes.csv"), "w", newline="")
    impact_log_f = open(os.path.join(log_dir, "impacts.csv"), "w", newline="")
    files += [pothole_log_f, impact_log_f]
    pothole_log, impact_log = csv.writer(pothole_log_f), csv.writer(impact_log_f)
    pothole_log.writerow(["time", "pc_time", "size", "area_pct", "conf", "x1", "y1", "x2", "y2"])
    impact_log.writerow(["time", "pc_time", "phone_ts_ms", "impact", "jolt_m_s2", "jolt_level",
                         "twist_rad_s", "twist_level", "trigger"])
    print(f"Logging to {log_dir}")

    def log_potholes(detections, pc_time):
        stamp = time.strftime("%H:%M:%S", time.localtime(pc_time))
        for d in detections:
            pothole_log.writerow([stamp, f"{pc_time:.3f}", d["size"], f"{d['area_frac'] * 100:.2f}",
                                  d["conf"], *d["box"]])
        pothole_log_f.flush()

    potholes = None
    if not args.no_detect:
        from pavewatch.detector import PotholeDetector
        print(f"Loading pothole model {config.DETECTOR_WEIGHTS} ...")
        potholes = PotholeDetector(config.DETECTOR_WEIGHTS, config.DETECT_CONF, config.DETECT_IMGSZ,
                                   config.POTHOLE_SIZE_BANDS, on_result=log_potholes).start()
    events_out = open(os.path.join(args.record, "events.jsonl"), "a") if args.record else None

    last_t = 0.0
    n_frames = 0
    try:
        while True:
            for name, ts, v, pc_time in imu.drain():
                if imu_csv:
                    imu_csv.writerow([name, ts, *v[:3], f"{pc_time:.3f}"])
                event = shock_detector.feed(ts, name, v)
                if event:
                    print(format_event(event))
                    now = time.time()
                    shocks.append((now, event))
                    impact_log.writerow([time.strftime("%H:%M:%S", time.localtime(now)), f"{now:.3f}",
                                         event["t_ms"], event["severity"], event["peak"], event["accel_severity"],
                                         event["gyro_peak"], event["gyro_severity"], event["trigger"]])
                    impact_log_f.flush()
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

            if potholes:
                potholes.submit(frame.copy())
                draw_potholes(frame, *potholes.latest())
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
        if potholes:
            potholes.stop()
        if writer is not None:
            writer.release()
        for f in files:
            f.close()
        if events_out:
            events_out.close()
        cv2.destroyAllWindows()
        print(f"Pothole and impact logs saved in {log_dir}")
        if args.record:
            print(f"Saved {n_frames} frames to {args.record}")


if __name__ == "__main__":
    main()
