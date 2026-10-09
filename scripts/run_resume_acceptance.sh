#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg PYTHONUNBUFFERED=1
STYLE_PYTHON="$HOME/workspace/Domain_Adaptation/.venv-crpl/bin/python"
STYLE_DATA="$HOME/workspace/da_lab/data/office_home"
"$STYLE_PYTHON" -u train.py --data_root "$STYLE_DATA" --output_dir runs/smoke_resume_verified \
  --targets art --stop_after 20 --smoke --evaluation_step 50
"$STYLE_PYTHON" -u train.py --data_root "$STYLE_DATA" --output_dir runs/smoke_resume_verified \
  --targets art --stop_after 50 --smoke --evaluation_step 50 \
  --resume runs/smoke_resume_verified/art/checkpoints/last.pth
"$STYLE_PYTHON" scripts/verify_resume.py runs/smoke_s1_50/art/checkpoints/last.pth \
  runs/smoke_resume_verified/art/checkpoints/last.pth --output runs/acceptance/resume_parity.json
