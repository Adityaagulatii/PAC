#!/bin/bash
# Agent panel: live view of the Remember + Decide stage (readings, trend, agent decision, Telegram status).
# Usage: pavewatch_agent/panels/agent_panel.sh     (then run pavewatch_agent/run_pass.sh in another terminal)
cd "$(dirname "$0")/../.."
printf '\033]0;PaveWatch Agent\007'
mkdir -p data/history && touch data/history/pavewatch.log
echo "PaveWatch Agent panel - follows data/history/pavewatch.log - Ctrl+C to stop"
tail -n 0 -F data/history/pavewatch.log | sed -u \
  -e 's/\(Decision: FLAG.*\)/\x1b[1;31m\1\x1b[0m/' -e 's/\(Decision: SCHEDULE.*\)/\x1b[1;33m\1\x1b[0m/' \
  -e 's/\(Decision: WATCH.*\)/\x1b[1;32m\1\x1b[0m/' -e 's/\(Telegram alert sent.*\)/\x1b[1;36m\1\x1b[0m/'
