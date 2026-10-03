"""Settings for the Remember + Decide stage (match -> history -> trend -> agent -> Telegram)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where the robot is. The phone's GPS isn't available yet, so a fixed point is used unless --gps is given.
DEFAULT_GPS = (42.370162, -71.070815)
MATCH_RADIUS_M = 15.0        # observations closer than this are the same pothole

# Which YOLO detection represents the pass
MIN_YOLO_CONF = 0.6          # detections below this are ignored
FRAME_STEP = 2               # run YOLO on every Nth video frame of a recorded pass
IMU_WINDOW_S = 2.0           # an impact within this many seconds of the chosen frame belongs to it

# VLM gate: only potholes the VLM confirms go to the agent
VLM_URL = os.environ.get("VLM_URL", "http://localhost:8001")
VLM_MIN_CONF = 0.6
VLM_FLAGGER_PATHS = [os.path.join(ROOT, "vlm", "vlm_flagger.py")]

# Repair line and forecast (history.py). The repair index mixes camera, VLM and IMU into 0..1.
REPAIR_AREA_FRAC = 0.10      # YOLO box share of the frame that counts as full size (= "large" band in config.py)
INDEX_WEIGHTS = {"camera": 0.5, "vlm": 0.25, "imu": 0.25}
REPAIR_LINE = 0.9            # repair index at which the defect needs repair
ALERT_HORIZON_WEEKS = 2      # alert when the forecast says the line is crossed within this many passes
DAYS_PER_PASS = 7            # the robot drives the route once a week

# Agent + Telegram (OpenClaw inside the NemoClaw sandbox)
SANDBOX = "pavewatch"
SANDBOX_DIR = "/sandbox/.openclaw/workspace/pavewatch"   # OpenClaw only attaches media from its workspace
TELEGRAM_CHAT_ID = os.environ.get("PAVEWATCH_TELEGRAM_ID", "")   # your Telegram user id (ask @userinfobot)
NOTIFY_TIERS = ("FLAG", "SCHEDULE")   # WATCH is stored silently
AGENT_TIMEOUT_S = 600

HISTORY_DIR = os.path.join(ROOT, "data", "history")
