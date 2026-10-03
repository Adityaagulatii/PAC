"""Remember + Forecast + Decide: store a pass in the defect's history, forecast when it crosses the repair
line, let the agent make the call, and alert a human on Telegram only when repair is due or about to be.

The agent (Nemotron via OpenClaw in the OpenShell sandbox) gets this pass, every earlier pass, the repair-line
forecast and the defect's fix-first priority, and answers:
  FLAG      the repair line is crossed - repair now                         -> Telegram
  SCHEDULE  forecast to cross within ALERT_HORIZON_WEEKS - plan the repair  -> Telegram
  WATCH     further away or not growing - keep tracking                      -> stored silently

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
from pavewatch_agent.history import History, priorities, repair_index, trend

TIERS = ("FLAG", "SCHEDULE", "WATCH")

INSTRUCTIONS = f"""You are PaveWatch, a road-maintenance agent for a small crew. A robot drives the same route
once a week. Each time it passes a road defect it records: a YOLO detection (size small/medium/large from how
much of the camera frame the defect covers, from a fixed mount), a VLM verdict (Qwen3-VL confirms it is a
pothole and explains why), an IMU impact (how hard the robot was jolted: low/medium/high) and GPS.
Matching by GPS gives this defect's earlier weekly passes.

The repair index (0..1) combines camera size, VLM size and IMU impact. Repair is due at {cfg.REPAIR_LINE}
(the repair line). A forecast fitted over all passes gives the weeks until the line is crossed.
The crew only wants a message when repair is due or about to be:
- FLAG: the repair line is crossed (or the evidence clearly shows it must be repaired now).
- SCHEDULE: forecast to cross the line within {cfg.ALERT_HORIZON_WEEKS} weeks - plan the repair.
- WATCH: further from the line, not growing, or only one pass without severe signs - no message.
Use the forecast, but overrule it if the evidence disagrees (e.g. a HIGH impact on a small-looking defect),
and say why.

Do not run tools or open files; everything you need is below.
Reply with ONLY a JSON object, no other text:
{{"tier": "FLAG|SCHEDULE|WATCH", "severity": "low|medium|high", "weeks_to_repair_line": <number or null>,
 "reason": "<one sentence>", "telegram_message": "<2-4 line message for the crew, include the location>"}}"""


def describe(o):
    y, v, i = o.get("yolo") or {}, o.get("vlm") or {}, o.get("imu") or {}
    imu = i.get("severity", "none")
    if i.get("jolt") is not None:
        imu += f" (jolt {i['jolt']:.1f} m/s2, twist {i['twist']:.1f} rad/s)"
    return (f"week {o['week']} ({o['observed_at'][:10]}): YOLO {y.get('size')} conf {y.get('conf')} "
            f"area {y.get('area_frac', 0) * 100:.1f}% | VLM {v.get('label')} conf {v.get('confidence')} "
            f"size {v.get('size')} ('{v.get('reason')}') | IMU {imu}")


def forecast_line(tr):
    weeks = tr.get("weeks_to_repair_line")
    if weeks == 0:
        return f"repair line crossed (index {tr['repair_index'][-1]} >= {tr['repair_line']})"
    if weeks is not None:
        return (f"crosses the repair line in ~{weeks} week(s), around {tr['repair_line_date']} "
                f"(index {tr['repair_index'][-1]} rising {tr['index_slope_per_week']}/week)")
    return f"{tr.get('status')} (index {tr['repair_index'][-1]} of {tr['repair_line']})"


def build_prompt(pothole, obs, tr, rank, total):
    past = [o for o in pothole["observations"][:-1] if o.get("yolo")]
    lines = [INSTRUCTIONS, "",
             f"Pothole {pothole['id']} at {obs['gps']['lat']:.6f}, {obs['gps']['lon']:.6f} "
             f"(map: https://maps.google.com/?q={obs['gps']['lat']},{obs['gps']['lon']})",
             "", "This pass:", describe(obs), "", "Earlier passes:"]
    lines += [describe(o) + f" -> decided {o.get('decision', {}).get('tier')}" for o in past] or ["none (first time seen)"]
    lines += ["", "Trend computed from all passes:", json.dumps(tr),
              "", f"Forecast: {forecast_line(tr)}",
              f"Fix-first priority: #{rank} of {total} tracked defects"]
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
    """Used only when the agent's answer can't be read: the forecast policy, so a due repair is never dropped."""
    weeks = tr.get("weeks_to_repair_line")
    tier = "FLAG" if weeks == 0 else "SCHEDULE" if weeks is not None and weeks <= cfg.ALERT_HORIZON_WEEKS \
        else "WATCH"
    loc = f"{obs['gps']['lat']:.6f}, {obs['gps']['lon']:.6f}"
    return {"tier": tier, "severity": "high" if tier == "FLAG" else "medium" if tier == "SCHEDULE" else "low",
            "weeks_to_repair_line": weeks, "reason": f"forecast policy fallback ({why})",
            "telegram_message": f"PaveWatch {tier}: defect at {loc} - {forecast_line(tr)}."}


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
            f"  IMU: {imu}",
            f"  Repair index: {repair_index(o)} (repair line {cfg.REPAIR_LINE})"]


def report(decision, pothole, obs, tr, rank, total):
    lat, lon = obs["gps"]["lat"], obs["gps"]["lon"]
    lines = [f"🚧 PaveWatch {decision['tier']} - {pothole['id']}  (fix-first priority #{rank} of {total})",
             f"Location: {lat:.6f}, {lon:.6f}",
             f"https://maps.google.com/?q={lat},{lon}",
             "",
             f"Forecast: {forecast_line(tr)}",
             f"Agent decision: {decision['tier']}, severity {decision.get('severity')}",
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
                  f"Frame area %: {' → '.join(str(a) for a in tr['area_pct'])}",
                  f"Repair index: {' → '.join(str(i) for i in tr['repair_index'])} (line {tr['repair_line']})"]
    return "\n".join(lines)


def send_telegram(decision, pothole, obs, tr, rank, total):
    """Full report as text, then one photo per pass (oldest first) captioned with that pass's readings."""
    send = ["openclaw", "message", "send", "--channel", "telegram", "--target", cfg.TELEGRAM_CHAT_ID]
    sandbox(*send, "--message", report(decision, pothole, obs, tr, rank, total), timeout=180)
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
    ranked = [p["id"] for p, _ in priorities(history)]
    rank, total = ranked.index(pothole["id"]) + 1, len(ranked)
    print(f"{pothole['id']} week {week}: {len(pothole['observations'])} pass(es) at this spot | trend {json.dumps(tr)}")
    print(f"Forecast: {forecast_line(tr)} | fix-first priority #{rank} of {total}")
    try:
        answer = ask_agent(build_prompt(pothole, obs, tr, rank, total), tag, args.obs)
        decision = parse_decision(answer) or fallback_decision(obs, tr, "agent reply had no decision")
        decision["agent_reply"] = answer[-2000:]
    except (RuntimeError, subprocess.TimeoutExpired, ValueError) as e:
        decision = fallback_decision(obs, tr, f"agent unavailable: {e}")
    print(f"Decision: {decision['tier']} | {decision.get('reason')}")
    print(report(decision, pothole, obs, tr, rank, total))

    decision["sent"] = False
    if decision["tier"] in cfg.NOTIFY_TIERS and not args.no_telegram and not cfg.TELEGRAM_CHAT_ID:
        print("Not sending: set PAVEWATCH_TELEGRAM_ID to your Telegram user id.")
    elif decision["tier"] in cfg.NOTIFY_TIERS and not args.no_telegram:
        try:
            send_telegram(decision, pothole, obs, tr, rank, total)
            decision["sent"] = True
            print("Telegram alert sent.")
        except (RuntimeError, subprocess.SubprocessError) as e:
            print(f"Telegram send failed: {e}")
    obs["decision"] = decision
    history.save()
    print(f"Saved to {history.path}")


if __name__ == "__main__":
    main()
