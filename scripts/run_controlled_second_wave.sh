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
echo "Waiting for r1 replay + r2 pooled completion" > "$STYLE_ROOT/wave2/${STYLE_ARM}_phase.txt"
while [[ ! -f "$STYLE_ROOT/queues/gpu0/exit.txt" || ! -f "$STYLE_ROOT/queues/gpu1/exit.txt" ]]; do
  sleep 5
done
[[ "$(cat "$STYLE_ROOT/queues/gpu0/exit.txt")" == *"exit_code=0" ]]
[[ "$(cat "$STYLE_ROOT/queues/gpu1/exit.txt")" == *"exit_code=0" ]]
cd "$STYLE_CONTROL"
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
