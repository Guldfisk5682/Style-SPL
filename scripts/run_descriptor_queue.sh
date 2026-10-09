#!/usr/bin/env bash
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$1"
STYLE_ARM="$2"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
STYLE_PYTHON="$HOME/workspace/Domain_Adaptation/.venv-crpl/bin/python"
STYLE_BASE="$HOME/workspace/Style-SPL"
STYLE_QUEUE_ROOT="$STYLE_BASE/runs/descriptor_20261010_seed1/queues/$STYLE_ARM"
[[ ! -e "$STYLE_QUEUE_ROOT" ]]
[[ -f "$STYLE_BASE/runs/descriptor_20261010_seed1/gate/acceptance.json" ]]
mkdir -p "$STYLE_QUEUE_ROOT/runtime"
trap 'code=$?; echo "$(date -Iseconds) exit_code=$code" > "$STYLE_QUEUE_ROOT/exit.txt"' EXIT
echo "$(date -Iseconds) gpu=$CUDA_VISIBLE_DEVICES" > "$STYLE_QUEUE_ROOT/start.txt"
STYLE_ARM_WORKTREE="$HOME/workspace/Style-SPL-experiments/$STYLE_ARM"
export STYLE_CODE_COMMIT="$(git -C "$STYLE_ARM_WORKTREE" rev-parse HEAD)"
git -C "$STYLE_ARM_WORKTREE" archive "$STYLE_CODE_COMMIT" | tar -x -C "$STYLE_QUEUE_ROOT/runtime"
cd "$STYLE_QUEUE_ROOT/runtime"
"$STYLE_PYTHON" -u scripts/run_descriptor_arm.py 2>&1 | tee "$STYLE_QUEUE_ROOT/train.log"
echo "completed" > "$STYLE_QUEUE_ROOT/phase.txt"
