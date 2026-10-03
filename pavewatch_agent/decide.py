"""Remember + Decide: store an observation in the pothole history, let the agent make the call, alert on Telegram.

The agent (Nemotron via OpenClaw in the OpenShell sandbox) gets the current reading, the pothole's past
passes and the computed trend, and answers FLAG / SCHEDULE / WATCH. FLAG and SCHEDULE are sent to Telegram
from inside the sandbox; WATCH is only stored, and the next pass is compared against it.

Usage:
    python3 -m pavewatch_agent.decide --obs data/history/obs/week1 [--week 1]
"""
import argparse
import json
import os
import re
import subprocess
import uuid

from pavewatch_agent import config as cfg
from pavewatch_agent.history import History, severity_rank, trend

TIERS = ("FLAG", "SCHEDULE", "WATCH")

INSTRUCTIONS = """You are PaveWatch, a road-maintenance agent. A robot drives the same fixed route once a week.
Each time it passes a pothole it records: a YOLO detection (size small/medium/large from how much of the
camera frame the pothole covers), a VLM verdict (Qwen3-VL confirms it is a pothole and estimates its size),
an IMU impact (how hard the robot was jolted: low/medium/high) and the GPS position.
Matching by GPS tells you this pothole's earlier weekly passes.

Decide one tier:
- FLAG: severe now (large, or a HIGH impact confirmed by the camera) - the crew must repair it now.
- SCHEDULE: not severe yet, but it is getting worse and will likely become severe soon - plan a repair.
- WATCH: small and stable, or only one pass so far without severe signs - keep monitoring, no message.

Do not run tools or open files; everything you need is below.
Reply with ONLY a JSON object, no other text:
{"tier": "FLAG|SCHEDULE|WATCH", "severity": "low|medium|high", "weeks_to_severe": <number or null>,
 "reason": "<one sentence>", "telegram_message": "<2-4 line alert for the crew, include the location>"}"""


def describe(o):
    y, v, i = o.get("yolo") or {}, o.get("vlm") or {}, o.get("imu") or {}
    imu = i.get("severity", "none")
    if i.get("jolt") is not None:
        imu += f" (jolt {i['jolt']:.1f} m/s2, twist {i['twist']:.1f} rad/s)"
    return (f"week {o['week']} ({o['observed_at'][:10]}): YOLO {y.get('size')} conf {y.get('conf')} "
            f"area {y.get('area_frac', 0) * 100:.1f}% | VLM {v.get('label')} conf {v.get('confidence')} "
            f"size {v.get('size')} ('{v.get('reason')}') | IMU {imu}")


def build_prompt(pothole, obs, tr):
    past = [o for o in pothole["observations"][:-1] if o.get("yolo")]
    lines = [INSTRUCTIONS, "",
             f"Pothole {pothole['id']} at {obs['gps']['lat']:.6f}, {obs['gps']['lon']:.6f} "
             f"(map: https://maps.google.com/?q={obs['gps']['lat']},{obs['gps']['lon']})",
             "", "This pass:", describe(obs), "", "Earlier passes:"]
    lines += [describe(o) + f" -> decided {o.get('decision', {}).get('tier')}" for o in past] or ["none (first time seen)"]
    lines += ["", "Trend computed from all passes:", json.dumps(tr)]
    return "\n".join(lines)


def sandbox(*argv, timeout=120):
    r = subprocess.run(["openshell", "sandbox", "exec", "-n", cfg.SANDBOX, "--", *argv],
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"sandbox command failed ({r.returncode}): {r.stderr.strip()[-500:]}")
    return r.stdout


def upload(local, remote):
    subprocess.run(["openshell", "sandbox", "upload", cfg.SANDBOX, local, remote],
                   stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=120)


def ask_agent(prompt, tag, workdir):
    path = os.path.join(workdir, f"{tag}_prompt.txt")
    with open(path, "w") as f:
        f.write(prompt)
    sandbox("mkdir", "-p", cfg.SANDBOX_DIR)
    upload(path, f"{cfg.SANDBOX_DIR}/{tag}_prompt.txt")
    out = sandbox("openclaw", "agent", "--agent", "main", "--session-id", f"pavewatch-{uuid.uuid4().hex[:12]}",
                  "--message-file", f"{cfg.SANDBOX_DIR}/{tag}_prompt.txt", "--json", timeout=cfg.AGENT_TIMEOUT_S)
    result = json.loads(out[out.index("{"):])
    texts = [p.get("text") or "" for p in result.get("result", {}).get("payloads", [])]
    return "\n".join(texts)


def parse_decision(text):
    for m in reversed(list(re.finditer(r"\{.*?\}", text, re.DOTALL))):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        tier = str(d.get("tier", "")).upper()
        if tier in TIERS:
            d["tier"] = tier
            return d
    return None


def fallback_decision(obs, tr, why):
    """Used only when the agent's answer can't be read, so a severe pothole is never dropped."""
    rank = severity_rank(obs)
    tier = "FLAG" if rank >= 3 else "SCHEDULE" if tr.get("severity_change", 0) > 0 else "WATCH"
    loc = f"{obs['gps']['lat']:.6f}, {obs['gps']['lon']:.6f}"
    return {"tier": tier, "severity": ["low", "low", "medium", "high"][rank], "weeks_to_severe": None,
            "reason": f"rule fallback ({why})",
            "telegram_message": f"PaveWatch {tier}: pothole at {loc}, severity rank {rank}/3. "
                                f"https://maps.google.com/?q={obs['gps']['lat']},{obs['gps']['lon']}"}


def pass_lines(o):
    """Every logged value of one pass, for the crew."""
    y, v, i = o.get("yolo") or {}, o.get("vlm") or {}, o.get("imu") or {}
    imu = (i.get("severity") or "none").upper()
    if i.get("jolt") is not None:
        imu += f" - jolt {i['jolt']:.2f} m/s², twist {i['twist']:.2f} rad/s"
    if i.get("manual"):
        imu += " (entered by hand)"
    elif i.get("count") and not i.get("aligned"):
        imu += f" (worst of {i['count']} impacts in the pass)"
    return [f"Week {o['week']} - {o['observed_at'].replace('T', ' ')}",
            f"  YOLO: {str(y.get('size')).upper()} pothole, conf {y.get('conf')}, "
            f"{y.get('area_frac', 0) * 100:.1f}% of frame",
            f"  VLM: {v.get('label')} (conf {v.get('confidence')}), size {v.get('size')}",
            f"  VLM says: \"{v.get('reason')}\"",
            f"  IMU: {imu}"]


def report(decision, pothole, obs, tr):
    lat, lon = obs["gps"]["lat"], obs["gps"]["lon"]
    weeks = f" (in ~{decision['weeks_to_severe']} weeks)" if decision.get("weeks_to_severe") else ""
    lines = [f"🚧 PaveWatch {decision['tier']} - {pothole['id']}",
             f"Location: {lat:.6f}, {lon:.6f}",
             f"https://maps.google.com/?q={lat},{lon}",
             "",
             f"Agent decision: {decision['tier']}, severity {decision.get('severity')}{weeks}",
             f"Why: {decision.get('reason')}",
             "", decision.get("telegram_message", ""), "",
             "── This pass ──", *pass_lines(obs)]
    past = [o for o in pothole["observations"][:-1] if o.get("yolo")]
    if past:
        lines += ["", "── Earlier passes ──"]
        for o in past:
            lines += pass_lines(o) + [f"  Decision then: {(o.get('decision') or {}).get('tier')}"]
    if tr.get("passes", 0) > 1:
        lines += ["", "── Trend ──",
                  f"YOLO size: {' → '.join(tr['yolo_sizes'])}",
                  f"VLM size: {' → '.join(str(s) for s in tr['vlm_sizes'])}",
                  f"IMU impact: {' → '.join(tr['imu_impacts'])}",
                  f"Frame area %: {' → '.join(str(a) for a in tr['area_pct'])}"
                  + (f" ({tr['area_growth_pct_per_week']:+} %/week)" if tr["area_growth_pct_per_week"] is not None else "")]
    return "\n".join(lines)


def send_telegram(decision, pothole, obs, tr, tag):
    """Full report as text, then one photo per pass (oldest first) captioned with that pass's readings."""
    send = ["openclaw", "message", "send", "--channel", "telegram", "--target", cfg.TELEGRAM_CHAT_ID]
    sandbox(*send, "--message", report(decision, pothole, obs, tr), timeout=180)
    for o in pothole["observations"]:
        if not o.get("annotated"):
            continue
        remote_photo = f"{cfg.SANDBOX_DIR}/{pothole['id']}_w{o['week']}.jpg"
        upload(os.path.join(cfg.ROOT, o["annotated"]), remote_photo)
        caption = "\n".join(pass_lines(o)) + ("\n← this pass" if o is obs else "")
        sandbox(*send, "--media", remote_photo, "--message", caption[:1000], timeout=180)


def main():
    ap = argparse.ArgumentParser(description="PaveWatch: remember the observation and let the agent decide")
    ap.add_argument("--obs", required=True, help="folder with observation.json from observe.py")
    ap.add_argument("--week", type=int, help="pass number on the route (default: one after this pothole's last)")
    ap.add_argument("--history", default=cfg.HISTORY_DIR, help="history folder (default data/history)")
    ap.add_argument("--no-telegram", action="store_true", help="decide and store, but don't send messages")
    args = ap.parse_args()

    with open(os.path.join(args.obs, "observation.json")) as f:
        obs = json.load(f)
    if not obs.get("yolo"):
        print("No confident YOLO detection in this pass; nothing to remember.")
        return

    history = History(args.history)
    known = history.match(obs["gps"])
    week = args.week or ((known["observations"][-1]["week"] + 1) if known else 1)
    pothole = history.add(obs, week)
    obs = pothole["observations"][-1]
    tag = f"{pothole['id']}_w{week}"

    vlm = obs.get("vlm") or {}
    if vlm.get("label") != "pothole" or vlm.get("confidence", 0) < cfg.VLM_MIN_CONF:
        obs["decision"] = {"tier": "REJECTED", "reason": f"VLM says {vlm.get('label')} "
                                                         f"({vlm.get('confidence')}): {vlm.get('reason')}"}
        history.save()
        print(f"{pothole['id']} week {week}: VLM rejected the detection -> stored, agent not called.")
        return

    tr = trend(pothole)
    print(f"{pothole['id']} week {week}: {len(pothole['observations'])} pass(es) at this spot | trend {json.dumps(tr)}")
    try:
        answer = ask_agent(build_prompt(pothole, obs, tr), tag, args.obs)
        decision = parse_decision(answer) or fallback_decision(obs, tr, "agent reply had no decision")
        decision["agent_reply"] = answer[-2000:]
    except (RuntimeError, subprocess.TimeoutExpired, ValueError) as e:
        decision = fallback_decision(obs, tr, f"agent unavailable: {e}")
    print(f"Decision: {decision['tier']} | {decision.get('reason')}")
    print(report(decision, pothole, obs, tr))

    decision["sent"] = False
    if decision["tier"] in cfg.NOTIFY_TIERS and not args.no_telegram and not cfg.TELEGRAM_CHAT_ID:
        print("Not sending: set PAVEWATCH_TELEGRAM_ID to your Telegram user id.")
    elif decision["tier"] in cfg.NOTIFY_TIERS and not args.no_telegram:
        try:
            send_telegram(decision, pothole, obs, tr, tag)
            decision["sent"] = True
            print("Telegram alert sent.")
        except (RuntimeError, subprocess.SubprocessError) as e:
            print(f"Telegram send failed: {e}")
    obs["decision"] = decision
    history.save()
    print(f"Saved to {history.path}")


if __name__ == "__main__":
    main()
