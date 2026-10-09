#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg PYTHONUNBUFFERED=1
STYLE_PYTHON="$HOME/workspace/Domain_Adaptation/.venv-crpl/bin/python"
STYLE_DATA="$HOME/workspace/da_lab/data/office_home"
[[ ! -e runs/b0_officehome_seed1 && ! -e runs/s1_officehome_seed1 ]]
mkdir -p runs/paired_officehome
trap 'code=$?; echo "$(date -Iseconds) exit_code=$code" > runs/paired_officehome/queue_exit.txt' EXIT
"$STYLE_PYTHON" -u train.py --data_root "$STYLE_DATA" --output_dir runs/b0_officehome_seed1 \
  --style_spl_enabled 0 2>&1 | tee runs/paired_officehome/b0_console.log
"$STYLE_PYTHON" -u train.py --data_root "$STYLE_DATA" --output_dir runs/s1_officehome_seed1 \
  --style_spl_enabled 1 2>&1 | tee runs/paired_officehome/s1_console.log
"$STYLE_PYTHON" scripts/plot_metrics.py runs/s1_officehome_seed1 --baseline runs/b0_officehome_seed1
