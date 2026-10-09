#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg PYTHONUNBUFFERED=1
STYLE_PYTHON="${STYLE_PYTHON:-$HOME/workspace/Domain_Adaptation/.venv-crpl/bin/python}"
STYLE_DATA="${STYLE_DATA:-$HOME/workspace/da_lab/data/office_home}"
exec "$STYLE_PYTHON" -u train.py --data_root "$STYLE_DATA" --output_dir "${STYLE_RUN_ROOT:-runs/s1_officehome_seed1}" "$@"
