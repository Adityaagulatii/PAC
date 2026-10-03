#!/bin/bash
# VLM panel: whenever live_view.py logs a YOLO detection with conf >= CONF, send that moment's phone photo
# to the local VLM (vlm/vlm_flagger.py, Qwen3-VL on localhost:8001) and print the verdict.
# Usage: PHONE_IP=172.20.65.194 pavewatch_agent/panels/vlm_panel.sh      (run next to live_view.py)
cd "$(dirname "$0")/../.."
PHONE_IP=${PHONE_IP:-$(python3 -c 'import config; print(config.PHONE_IP)')}
CONF=${CONF:-0.6}
work=data/vlm_panel; mkdir -p "$work/in"
printf '\033]0;PaveWatch VLM\007'
echo "PaveWatch VLM panel - waits for YOLO conf >= $CONF, then asks Qwen3-VL (local). Ctrl+C to stop"
last_sent=0
while true; do
  log=$(ls -td logs/*/ 2>/dev/null | head -1)potholes.csv
  read -r t conf size <<<"$(tail -1 "$log" 2>/dev/null | awk -F, '{print $2, $5, $3}')"
  now=$(date +%s.%N)
  if [[ -n "$t" && "$t" != pc_time ]] && awk -v n="$now" -v t="$t" -v c="$conf" -v m="$CONF" -v l="$last_sent" \
        'BEGIN{exit !((n-t)<2 && c>=m && t>l)}'; then
    last_sent=$t
    curl -s -m 5 -o "$work/in/frame.jpg" "http://$PHONE_IP:8080/shot.jpg" || { echo "phone not reachable"; sleep 2; continue; }
    line=$(python3 vlm/vlm_flagger.py --images "$work/in" --out "$work/out" 2>&1 | grep -m1 'frame.jpg')
    verdict=$(python3 -c "import csv;r=next(csv.DictReader(open('$work/out/results.csv')));print(f\"{r['label'].upper():11s} conf {float(r['confidence']):.2f}  size {r['size'] or '-':6s} | {r['reason']}\")" 2>/dev/null)
    case "$line" in *FLAG*) c='\033[1;31m';; *pothole*) c='\033[1;33m';; *) c='\033[1;32m';; esac
    echo -e "$(date +%H:%M:%S)  YOLO ${size} ${conf} -> VLM: ${c}${verdict}\033[0m$( [[ $line == *FLAG* ]] && echo '  <- FLAGGED')"
  fi
  sleep 0.5
done
