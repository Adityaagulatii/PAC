#!/bin/bash
# One weekly pass through PaveWatch's Remember + Decide stage:
#   observe (YOLO + IMU + VLM, in the pavewatch-detector container on the GPU)
#   -> decide (match by GPS, trend, agent call, Telegram) on the host.
#
# Usage:
#   pavewatch_agent/run_pass.sh data/week1                    # a pass recorded with live_view.py --record
#   pavewatch_agent/run_pass.sh images/1.jpg --impact low     # a single photo (IMU level given by hand)
# Extra options go to observe.py (--gps LAT LON, --impact) ; set WEEK=N / HISTORY=dir / NO_TELEGRAM=1 to override.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f infra/pavewatch.env ]] && source infra/pavewatch.env   # local settings, e.g. PAVEWATCH_TELEGRAM_ID
src="$1"; shift
name="$(basename "${src%.*}")_$(date +%Y%m%d_%H%M%S)"
out="data/history/obs/$name"
if [[ -d "$src" ]]; then input=(--pass "$src"); else input=(--image "$src"); fi

docker_cmd=(docker run --rm --gpus all --network host --user "$(id -u):$(id -g)"
  -e HOME=/tmp -e YOLO_CONFIG_DIR=/tmp -e YOLO_OFFLINE=1 -v "$PWD:/pac" -w /pac
  --entrypoint python3 pavewatch-detector:latest -m pavewatch_agent.observe "${input[@]}" --out "$out" "$@")
log="${HISTORY:-data/history}/pavewatch.log"; mkdir -p "$(dirname "$log")"
echo "════ $(date '+%Y-%m-%d %H:%M:%S')  pass: $src ════" | tee -a "$log"
if docker info >/dev/null 2>&1; then "${docker_cmd[@]}"; else sg docker -c "$(printf '%q ' "${docker_cmd[@]}")"; fi 2>&1 \
  | grep -vE 'Warning|warn\(' | tee -a "$log"

export PATH="$HOME/.local/bin:$PATH"
python3 -m pavewatch_agent.decide --obs "$out" ${WEEK:+--week "$WEEK"} ${HISTORY:+--history "$HISTORY"} \
  ${NO_TELEGRAM:+--no-telegram} 2>&1 | tee -a "$log"
