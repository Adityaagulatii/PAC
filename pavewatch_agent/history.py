"""Pothole memory: match a new observation to a known pothole by GPS and work out how it is changing.

Stored as one JSON file (data/history/potholes.json):
  {"potholes": [{"id": "PH-0001", "lat": .., "lon": .., "observations": [observation, ...]}]}
Each observation is what observe.py wrote, plus "week" and the agent's "decision".
"""
import datetime
import json
import math
import os

from pavewatch_agent import config as cfg

RANK = {"none": 0, "low": 1, "small": 1, "medium": 2, "high": 3, "large": 3}


def distance_m(a, b):
    """Great-circle distance between two {"lat", "lon"} points, in metres."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


class History:
    def __init__(self, folder=cfg.HISTORY_DIR):
        self.path = os.path.join(folder, "potholes.json")
        self.data = {"potholes": []}
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.data = json.load(f)

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)

    def match(self, gps):
        """The known pothole nearest to gps within MATCH_RADIUS_M, or None."""
        best = None
        for p in self.data["potholes"]:
            d = distance_m(p, gps)
            if d <= cfg.MATCH_RADIUS_M and (best is None or d < best[0]):
                best = (d, p)
        return best[1] if best else None

    def add(self, obs, week):
        """Attach obs to its pothole (creating one if this location is new). Returns the pothole."""
        pothole = self.match(obs["gps"])
        if pothole is None:
            pothole = {"id": f"PH-{len(self.data['potholes']) + 1:04d}",
                       "lat": obs["gps"]["lat"], "lon": obs["gps"]["lon"], "observations": []}
            self.data["potholes"].append(pothole)
        obs = dict(obs, week=week)
        pothole["observations"].append(obs)
        return pothole


def severity_rank(obs):
    """0-3: the worst of YOLO size, VLM size and IMU impact."""
    yolo = (obs.get("yolo") or {}).get("size") or "none"
    vlm = (obs.get("vlm") or {}).get("size") or "none"
    imu = (obs.get("imu") or {}).get("severity") or "none"
    return max(RANK.get(yolo, 0), RANK.get(vlm, 0), RANK.get(imu, 0))


def repair_index(obs):
    """0..1 score of how close a defect is to needing repair; cfg.REPAIR_LINE is where repair is due.

    Weighted mix of the three sensors, each scaled to 0..1:
      camera  YOLO box share of the frame / REPAIR_AREA_FRAC (capped at 1) - the size seen from the fixed mount
      VLM     size estimate: small 1/3, medium 2/3, large 1
      IMU     impact felt by the robot: low 1/3, medium 2/3, high 1
    """
    y, v, i = obs.get("yolo") or {}, obs.get("vlm") or {}, obs.get("imu") or {}
    parts = {"camera": min(y.get("area_frac", 0) / cfg.REPAIR_AREA_FRAC, 1.0),
             "vlm": RANK.get(v.get("size") or "none", 0) / 3,
             "imu": RANK.get(i.get("severity") or "none", 0) / 3}
    return round(sum(cfg.INDEX_WEIGHTS[k] * parts[k] for k in parts), 3)


def forecast(points, line):
    """Least-squares line through (week, index); weeks from the last pass until it reaches `line`.

    Returns (slope per week, weeks to cross): weeks is 0 if already crossed, None if it isn't growing
    or there is only one pass.
    """
    week_now, index_now = points[-1]
    if index_now >= line:
        return None, 0.0
    if len(points) < 2:
        return None, None
    n = len(points)
    mw = sum(w for w, _ in points) / n
    mi = sum(i for _, i in points) / n
    var = sum((w - mw) ** 2 for w, _ in points)
    if var == 0:
        return None, None
    slope = sum((w - mw) * (i - mi) for w, i in points) / var
    if slope <= 0:
        return round(slope, 3), None
    return round(slope, 3), round((line - index_now) / slope, 1)


def trend(pothole):
    """How the defect has changed across its passes (oldest first) and when it will cross the repair line."""
    obs = [o for o in pothole["observations"] if o.get("yolo")]
    if not obs:
        return {"passes": 0}
    first, last = obs[0], obs[-1]
    index = [repair_index(o) for o in obs]
    slope, weeks = forecast([(o["week"], i) for o, i in zip(obs, index)], cfg.REPAIR_LINE)
    cross_date = None
    if weeks:
        observed = datetime.datetime.fromisoformat(last["observed_at"])
        cross_date = (observed + datetime.timedelta(days=weeks * cfg.DAYS_PER_PASS)).date().isoformat()
    if weeks == 0:
        status = "crossed"
    elif weeks is not None and weeks <= cfg.ALERT_HORIZON_WEEKS:
        status = "crossing soon"
    elif weeks is not None:
        status = "growing"
    else:
        status = "first pass" if len(obs) == 1 else "not growing"
    return {
        "passes": len(obs),
        "weeks_tracked": last["week"] - first["week"],
        "yolo_sizes": [o["yolo"]["size"] for o in obs],
        "vlm_sizes": [(o.get("vlm") or {}).get("size") for o in obs],
        "imu_impacts": [o["imu"]["severity"] for o in obs],
        "area_pct": [round(o["yolo"]["area_frac"] * 100, 1) for o in obs],
        "repair_index": index,
        "repair_line": cfg.REPAIR_LINE,
        "index_slope_per_week": slope,
        "weeks_to_repair_line": weeks,
        "repair_line_date": cross_date,
        "status": status,
        "severity_change": severity_rank(last) - severity_rank(first),
    }


def priorities(history):
    """Every tracked defect, most urgent first: crossed (highest index first), then soonest to cross."""
    rows = []
    for p in history.data["potholes"]:
        tr = trend(p)
        if not tr.get("passes"):
            continue
        weeks = tr["weeks_to_repair_line"]
        key = (0, -tr["repair_index"][-1]) if weeks == 0 else (1, weeks) if weeks is not None \
            else (2, -tr["repair_index"][-1])
        rows.append((key, p, tr))
    rows.sort(key=lambda r: r[0])
    return [(p, tr) for _, p, tr in rows]


def main():
    """Print the fix-first list: python3 -m pavewatch_agent.history [--history DIR]"""
    import argparse
    ap = argparse.ArgumentParser(description="PaveWatch: which defects to fix first")
    ap.add_argument("--history", default=cfg.HISTORY_DIR)
    args = ap.parse_args()
    rows = priorities(History(args.history))
    if not rows:
        print("No defects tracked yet.")
    for n, (p, tr) in enumerate(rows, 1):
        weeks = tr["weeks_to_repair_line"]
        when = "repair due now" if weeks == 0 else (f"crosses in ~{weeks} wk ({tr['repair_line_date']})"
                                                     if weeks is not None else tr["status"])
        last = p["observations"][-1]
        print(f"#{n}  {p['id']}  {p['lat']:.6f},{p['lon']:.6f}  index {tr['repair_index'][-1]:.2f}/{tr['repair_line']}"
              f"  {when}  | last decision {(last.get('decision') or {}).get('tier')}  | {tr['passes']} pass(es)")


if __name__ == "__main__":
    main()
