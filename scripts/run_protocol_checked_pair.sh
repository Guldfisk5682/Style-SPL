#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg PYTHONUNBUFFERED=1
STYLE_REPO_ROOT="$PWD"
STYLE_DA_ROOT="$HOME/workspace/Domain_Adaptation"
STYLE_PYTHON="$STYLE_DA_ROOT/.venv-crpl/bin/python"
STYLE_DATA="$HOME/workspace/da_lab/data/office_home"
STYLE_PAIR_ROOT="$STYLE_REPO_ROOT/runs/paired_protocol_checked_20261009"
STYLE_BASE_ROOT="$STYLE_REPO_ROOT/runs/b0_protocol_checked_20261009_seed1"
STYLE_S1_ROOT="$STYLE_REPO_ROOT/runs/s1_protocol_checked_20261009_seed1"
export CRPL_OFFICIAL_RUN_ROOT="$STYLE_DA_ROOT/runs/crpl_spl_protocol_checked_20261009_seed1"
[[ ! -e "$STYLE_PAIR_ROOT" && ! -e "$STYLE_BASE_ROOT" && ! -e "$STYLE_S1_ROOT" && ! -e "$CRPL_OFFICIAL_RUN_ROOT" ]]
mkdir -p "$STYLE_PAIR_ROOT/runtime"
trap 'code=$?; echo "$(date -Iseconds) exit_code=$code" > "$STYLE_PAIR_ROOT/queue_exit.txt"' EXIT
export STYLE_CODE_COMMIT="$(git rev-parse HEAD)"
git archive "$STYLE_CODE_COMMIT" | tar -x -C "$STYLE_PAIR_ROOT/runtime"
echo "$(date -Iseconds) start gpu=$CUDA_VISIBLE_DEVICES" > "$STYLE_PAIR_ROOT/queue_start.txt"
echo "B0: corrected official SPL" > "$STYLE_PAIR_ROOT/phase.txt"
bash "$STYLE_DA_ROOT/scripts/run_crpl_protocol_checked.sh" 2>&1 | tee "$STYLE_PAIR_ROOT/b0_console.log"
cd "$STYLE_PAIR_ROOT/runtime"
"$STYLE_PYTHON" scripts/export_official_baseline.py "$CRPL_OFFICIAL_RUN_ROOT" "$STYLE_BASE_ROOT"
echo "S1: Style-SPL, same corrected data protocol" > "$STYLE_PAIR_ROOT/phase.txt"
"$STYLE_PYTHON" -u train.py --data_root "$STYLE_DATA" --output_dir "$STYLE_S1_ROOT" \
  --cache_dir "$STYLE_REPO_ROOT/cache/style_banks" --style_spl_enabled 1 2>&1 | tee "$STYLE_PAIR_ROOT/s1_console.log"
"$STYLE_PYTHON" scripts/plot_metrics.py "$STYLE_S1_ROOT" --baseline "$STYLE_BASE_ROOT"
echo "Completed" > "$STYLE_PAIR_ROOT/phase.txt"
