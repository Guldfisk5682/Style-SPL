#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg PYTHONUNBUFFERED=1
STYLE_PYTHON="$HOME/workspace/Domain_Adaptation/.venv-crpl/bin/python"
STYLE_DATA="$HOME/workspace/da_lab/data/office_home"
mkdir -p runs/acceptance
trap 'code=$?; echo "$(date -Iseconds) exit_code=$code" > runs/acceptance/validation_exit.txt' EXIT
"$STYLE_PYTHON" -m unittest discover -s tests -v
"$STYLE_PYTHON" scripts/verify_backbone.py \
  --baseline_root "$HOME/workspace/Domain_Adaptation/runs/crpl_spl_only_20261009_seed1/author_code" \
  --data_root "$STYLE_DATA" --output runs/acceptance/backbone_parity.json
"$STYLE_PYTHON" -u train.py --data_root "$STYLE_DATA" --output_dir runs/smoke_s1_50 \
  --targets art --stop_after 50 --smoke --evaluation_step 50 2>&1 | tee runs/acceptance/smoke_console.log
"$STYLE_PYTHON" scripts/verify_style_gradients.py --bank cache/style_banks/art.pt \
  --data_root "$STYLE_DATA" --output runs/acceptance/style_gradient_paths.json
"$STYLE_PYTHON" scripts/plot_metrics.py runs/smoke_s1_50
