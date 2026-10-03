"""Turn one pass of the robot into one pothole observation: photo + YOLO size + VLM verdict + IMU impact + GPS.

Runs where OpenCV, ultralytics and the GPU are available (the pavewatch-detector container).

Usage:
    python -m pavewatch_agent.observe --pass data/week1 --out data/history/obs/week1
    python -m pavewatch_agent.observe --image images/1.jpg --impact low --out /tmp/obs1   # no video/IMU recording

A recorded pass is the folder written by `live_view.py --record DIR` (video.mp4, frames.csv, imu.csv).
Writes observation.json, photo.jpg (raw frame, what the VLM saw) and annotated.jpg (with the YOLO box).
"""
import argparse
import csv
import importlib.util
import json
import os
import time

import cv2

import config as team
from pavewatch.detector import PotholeDetector
from pavewatch.shock import LEVELS, ShockDetector
from pavewatch_agent import config as cfg


def load_vlm_flagger():
    for path in cfg.VLM_FLAGGER_PATHS:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("vlm_flagger", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise SystemExit("vlm_flagger.py not found in: " + ", ".join(cfg.VLM_FLAGGER_PATHS))


def read_frame_times(pass_dir):
    path = os.path.join(pass_dir, "frames.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {int(r["frame"]): float(r["pc_time"]) for r in csv.DictReader(f)}


def best_detection_in_video(detector, pass_dir):
    """Run YOLO over the recorded video; keep the most confident detection (ties: the bigger box)."""
    frame_times = read_frame_times(pass_dir)
    cap = cv2.VideoCapture(os.path.join(pass_dir, "video.mp4"))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {pass_dir}/video.mp4")
    best, n = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % cfg.FRAME_STEP == 0:
            for d in detector.detect(frame):
                key = (d["conf"], d["area_frac"])
                if d["conf"] >= cfg.MIN_YOLO_CONF and (best is None or key > best[0]):
                    best = (key, d, frame.copy(), frame_times.get(n))
        n += 1
    cap.release()
    print(f"YOLO scanned {n} frames")
    return (best[1], best[2], best[3]) if best else (None, None, None)


def impacts_from_imu_csv(pass_dir):
    """Replay the pass's imu.csv through the team's ShockDetector; return events with pc_time attached."""
    path = os.path.join(pass_dir, "imu.csv")
    if not os.path.exists(path):
        return []
    detector = ShockDetector(team.SHOCK_THRESHOLD, team.SEVERITY_BANDS, team.GYRO_THRESHOLD, team.GYRO_BANDS,
                             team.SHOCK_WINDOW_S, team.SHOCK_REFRACTORY_S)
    pc_at = {}
    events = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            ts = float(r["phone_ts_ms"])
            pc_at[ts] = float(r["pc_time"])
            event = detector.feed(ts, r["sensor"], [float(r["x"]), float(r["y"]), float(r["z"])])
            if event:
                events.append(event)
    last = detector.flush()
    if last:
        events.append(last)
    for e in events:
        e["pc_time"] = pc_at.get(e["t_ms"])
    return events


def pick_impact(events, frame_time):
    """The worst impact near the chosen frame; if none is near (or times are unknown), the worst of the pass."""
    if not events:
        return {"severity": "none", "jolt": 0.0, "twist": 0.0, "aligned": False, "count": 0}
    worst = lambda evs: max(evs, key=lambda e: (LEVELS.index(e["severity"]), e["peak"]))
    near = [e for e in events if frame_time and e.get("pc_time")
            and abs(e["pc_time"] - frame_time) <= cfg.IMU_WINDOW_S]
    e = worst(near or events)
    return {"severity": e["severity"], "jolt": e["peak"], "twist": e["gyro_peak"],
            "aligned": bool(near), "count": len(events)}


def annotate(frame, det):
    out = frame.copy()
    x1, y1, x2, y2 = det["box"]
    color = {"small": (0, 255, 255), "medium": (0, 165, 255), "large": (0, 0, 255)}[det["size"]]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 4)
    cv2.putText(out, f"{det['size'].upper()} pothole {det['conf']:.2f}", (x1 + 4, max(30, y1 - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    return out


def main():
    ap = argparse.ArgumentParser(description="PaveWatch: build one pothole observation from a pass")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pass", dest="pass_dir", help="folder recorded with live_view.py --record")
    src.add_argument("--image", help="a single road photo instead of a recorded pass")
    ap.add_argument("--impact", choices=["none", "low", "medium", "high"],
                    help="IMU impact to use with --image (there is no recording to read it from)")
    ap.add_argument("--gps", nargs=2, type=float, metavar=("LAT", "LON"), default=cfg.DEFAULT_GPS)
    ap.add_argument("--out", required=True, help="folder for observation.json and photos")
    args = ap.parse_args()

    detector = PotholeDetector(team.DETECTOR_WEIGHTS, team.DETECT_CONF, team.DETECT_IMGSZ, team.POTHOLE_SIZE_BANDS)
    if args.pass_dir:
        det, frame, frame_time = best_detection_in_video(detector, args.pass_dir)
        imu = pick_impact(impacts_from_imu_csv(args.pass_dir), frame_time)
        source = os.path.relpath(os.path.abspath(args.pass_dir), cfg.ROOT)
    else:
        frame, frame_time = cv2.imread(args.image), None
        if frame is None:
            raise SystemExit(f"Could not read {args.image}")
        dets = [d for d in detector.detect(frame) if d["conf"] >= cfg.MIN_YOLO_CONF]
        det = max(dets, key=lambda d: (d["conf"], d["area_frac"])) if dets else None
        level = args.impact or "none"
        imu = {"severity": level, "jolt": None, "twist": None, "aligned": False, "count": int(level != "none"),
               "manual": True}
        source = os.path.relpath(os.path.abspath(args.image), cfg.ROOT)

    os.makedirs(args.out, exist_ok=True)
    obs = {"observed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": source,
           "gps": {"lat": args.gps[0], "lon": args.gps[1]}, "imu": imu}
    if det is None:
        obs.update(yolo=None, vlm=None, photo=None, annotated=None)
        print(f"No pothole with YOLO conf >= {cfg.MIN_YOLO_CONF} in this pass.")
    else:
        photo = os.path.join(args.out, "photo.jpg")
        annotated = os.path.join(args.out, "annotated.jpg")
        cv2.imwrite(photo, frame)
        cv2.imwrite(annotated, annotate(frame, det))
        vlm = load_vlm_flagger()
        verdict = vlm.classify(photo, cfg.VLM_URL, vlm.get_model(cfg.VLM_URL))
        obs.update(yolo={k: det[k] for k in ("size", "conf", "area_frac", "box")},
                   vlm={k: verdict[k] for k in ("label", "confidence", "size", "reason", "error")},
                   photo=os.path.relpath(os.path.abspath(photo), cfg.ROOT),
                   annotated=os.path.relpath(os.path.abspath(annotated), cfg.ROOT))  # relative: works in and out of the container
        print(f"YOLO {det['size']} {det['conf']:.2f} ({det['area_frac'] * 100:.1f}% of frame) | "
              f"VLM {verdict['label']} {verdict['confidence']:.2f} {verdict['size']} | "
              f"IMU {imu['severity']}{'' if imu['aligned'] or imu.get('manual') else ' (worst of pass)'}")
    with open(os.path.join(args.out, "observation.json"), "w") as f:
        json.dump(obs, f, indent=2)
    print(f"Wrote {args.out}/observation.json")


if __name__ == "__main__":
    main()
