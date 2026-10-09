#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
STYLE_REPO_ROOT="$PWD"
STYLE_ANALYSIS_RUNTIME="$PWD/runs/analysis_runtime_20261009"
[[ ! -e "$STYLE_ANALYSIS_RUNTIME" && ! -e runs/analysis_round_20261009 ]]
mkdir "$STYLE_ANALYSIS_RUNTIME"
cp scripts/analyze_saved_round.py scripts/round_analysis_metrics.py "$STYLE_ANALYSIS_RUNTIME/"
trap 'code=$?; echo "$(date -Iseconds) exit_code=$code" > "$STYLE_REPO_ROOT/runs/style_round_analysis_20261009_exit.txt"' EXIT
"$HOME/workspace/Domain_Adaptation/.venv-crpl/bin/python" -u "$STYLE_ANALYSIS_RUNTIME/analyze_saved_round.py" \
  --style_root "$PWD/runs/s1_protocol_checked_20261009_seed1" \
  --baseline_root "$PWD/runs/b0_protocol_checked_20261009_seed1" \
  --runtime_root "$PWD/runs/paired_protocol_checked_20261009/runtime" \
  --data_root "$HOME/workspace/da_lab/data/office_home" \
  --output "$PWD/runs/analysis_round_20261009" 2>&1 | tee runs/style_round_analysis_20261009.log
