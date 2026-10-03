"""Pothole memory: match a new observation to a known pothole by GPS and work out how it is changing.

Stored as one JSON file (data/history/potholes.json):
  {"potholes": [{"id": "PH-0001", "lat": .., "lon": .., "observations": [observation, ...]}]}
Each observation is what observe.py wrote, plus "week" and the agent's "decision".
"""
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


def trend(pothole):
    """How the pothole has changed across its observations (oldest first)."""
    obs = [o for o in pothole["observations"] if o.get("yolo")]
    if not obs:
        return {"passes": 0}
    first, last = obs[0], obs[-1]
    weeks = max((last["week"] - first["week"]), 0)
    area_now, area_then = last["yolo"]["area_frac"], first["yolo"]["area_frac"]
    growth = (area_now - area_then) / weeks if weeks else None
    weeks_to_large = None
    if growth and growth > 0 and area_now < cfg.LARGE_AREA_FRAC:
        weeks_to_large = round((cfg.LARGE_AREA_FRAC - area_now) / growth, 1)
    return {
        "passes": len(obs),
        "weeks_tracked": weeks,
        "yolo_sizes": [o["yolo"]["size"] for o in obs],
        "vlm_sizes": [(o.get("vlm") or {}).get("size") for o in obs],
        "imu_impacts": [o["imu"]["severity"] for o in obs],
        "area_pct": [round(o["yolo"]["area_frac"] * 100, 1) for o in obs],
        "area_growth_pct_per_week": round(growth * 100, 2) if growth is not None else None,
        "severity_ranks": [severity_rank(o) for o in obs],
        "severity_change": severity_rank(last) - severity_rank(first),
        "weeks_to_large_estimate": weeks_to_large,
    }
