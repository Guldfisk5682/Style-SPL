#!/usr/bin/env bash
set -euo pipefail
# Persistent waiter. This process uses no GPU until first-wave verification.
STYLE_GPU="$1"
STYLE_ARM="$2"
STYLE_PYTHON="$HOME/workspace/Domain_Adaptation/.venv-crpl/bin/python"
STYLE_ROOT="$HOME/workspace/Style-SPL/runs/controlled_20261009_seed1"
STYLE_CONTROL="$HOME/workspace/Style-SPL-controlled"
mkdir -p "$STYLE_ROOT/wave2"
trap 'code=$?; echo "$(date -Iseconds) exit_code=$code" > "$STYLE_ROOT/wave2/${STYLE_ARM}_waiter_exit.txt"' EXIT
STYLE_FIRST_QUEUE=gpu0
STYLE_FIRST_ARM=r1_shared
if [[ "$STYLE_ARM" == r4_silu32 ]]; then STYLE_FIRST_QUEUE=gpu1; STYLE_FIRST_ARM=r2_pooled; fi
echo "Waiting for $STYLE_FIRST_ARM completion and verified S1 replay" > "$STYLE_ROOT/wave2/${STYLE_ARM}_phase.txt"
while [[ ! -f "$STYLE_ROOT/queues/$STYLE_FIRST_QUEUE/exit.txt" ]]; do
  sleep 5
done
[[ "$(cat "$STYLE_ROOT/queues/$STYLE_FIRST_QUEUE/exit.txt")" == *"exit_code=0" ]]
cd "$STYLE_CONTROL"
"$STYLE_PYTHON" - "$STYLE_ROOT" "$STYLE_FIRST_ARM" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
arm=sys.argv[2]
summary=json.loads((root/arm/'summary.json').read_text())
assert summary['complete_four_targets'], arm
assert all(t['training_valid'] and t['protocol_valid'] and t['steps']==1000 for t in summary['tasks']), arm
PY
if [[ "$STYLE_ARM" == r3_split ]]; then
  "$STYLE_PYTHON" scripts/verify_s1_replay.py "$HOME/workspace/Style-SPL/runs/s1_protocol_checked_20261009_seed1" \
    "$STYLE_ROOT/r1_shared" --output "$STYLE_ROOT/replay_verification.json" > "$STYLE_ROOT/replay_verification.log" 2>&1
  touch "$STYLE_ROOT/wave2/replay_verified.ready"
else
  while [[ ! -f "$STYLE_ROOT/wave2/replay_verified.ready" ]]; do
    if [[ -f "$STYLE_ROOT/wave2/r3_split_waiter_exit.txt" ]]; then exit 1; fi
    sleep 5
  done
fi
echo "Training $STYLE_ARM" > "$STYLE_ROOT/wave2/${STYLE_ARM}_phase.txt"
bash scripts/run_controlled_queue.sh "$STYLE_GPU" "wave2_$STYLE_ARM" "$STYLE_ARM"
echo "completed" > "$STYLE_ROOT/wave2/${STYLE_ARM}_phase.txt"
