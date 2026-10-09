#!/usr/bin/env bash
set -euo pipefail
# Usage: bash run_controlled_queue.sh GPU_UUID queue_name arm_name...
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$1"
STYLE_QUEUE_NAME="$2"
shift 2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
STYLE_PYTHON="$HOME/workspace/Domain_Adaptation/.venv-crpl/bin/python"
STYLE_BASE="$HOME/workspace/Style-SPL"
STYLE_QUEUE_ROOT="$STYLE_BASE/runs/controlled_20261009_seed1/queues/$STYLE_QUEUE_NAME"
[[ ! -e "$STYLE_QUEUE_ROOT" ]]
mkdir -p "$STYLE_QUEUE_ROOT"
trap 'code=$?; echo "$(date -Iseconds) exit_code=$code" > "$STYLE_QUEUE_ROOT/exit.txt"' EXIT
echo "$(date -Iseconds) gpu=$CUDA_VISIBLE_DEVICES" > "$STYLE_QUEUE_ROOT/start.txt"
for STYLE_ARM in "$@"; do
  STYLE_ARM_WORKTREE="$HOME/workspace/Style-SPL-experiments/$STYLE_ARM"
  STYLE_RUNTIME="$STYLE_QUEUE_ROOT/${STYLE_ARM}_runtime"
  mkdir -p "$STYLE_RUNTIME"
  export STYLE_CODE_COMMIT="$(git -C "$STYLE_ARM_WORKTREE" rev-parse HEAD)"
  git -C "$STYLE_ARM_WORKTREE" archive "$STYLE_CODE_COMMIT" | tar -x -C "$STYLE_RUNTIME"
  echo "$STYLE_ARM" > "$STYLE_QUEUE_ROOT/phase.txt"
  echo "$(date -Iseconds)" > "$STYLE_QUEUE_ROOT/${STYLE_ARM}_start.txt"
  cd "$STYLE_RUNTIME"
  "$STYLE_PYTHON" -u scripts/run_controlled_arm.py 2>&1 | tee "$STYLE_QUEUE_ROOT/${STYLE_ARM}.log"
  echo "$(date -Iseconds) exit_code=0" > "$STYLE_QUEUE_ROOT/${STYLE_ARM}_exit.txt"
done
echo "completed" > "$STYLE_QUEUE_ROOT/phase.txt"
