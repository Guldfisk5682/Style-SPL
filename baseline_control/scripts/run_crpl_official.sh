#!/usr/bin/env bash
# Execute the pinned author README command, optionally with the documented control patch.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
# Read the launcher into memory before a long run, so later file updates cannot
# shift Bash's read offset while it waits for the training subprocess.
if [[ "${CRPL_RUNNER_FROZEN:-0}" != 1 ]]; then
  export CRPL_RUNNER_FROZEN=1
  exec bash -c "$(cat "$REPO_ROOT/scripts/run_crpl_official.sh")" "$REPO_ROOT/scripts/run_crpl_official.sh"
fi
CRPL_SOURCE="$REPO_ROOT/third_party/crpl"
CRPL_REVISION=2c032a62fc706b643bfc30f49c9a064dd24aa7ac
CRPL_PYTHON="${CRPL_PYTHON:-$REPO_ROOT/.venv-crpl/bin/python}"
DATA_ROOT="${DATA_ROOT:-$HOME/workspace/da_lab/data/office_home}"
RUN_ROOT="${CRPL_OFFICIAL_RUN_ROOT:-$REPO_ROOT/runs/crpl_official_20261009_seed1}"
export CRPL_CONTROL_VARIANT="${CRPL_CONTROL_VARIANT:-unmodified}"
[[ "$CRPL_CONTROL_VARIANT" == unmodified || "$CRPL_CONTROL_VARIANT" == init_checked || "$CRPL_CONTROL_VARIANT" == protocol_checked ]]
export CRPL_OT_CLUSTERING="${CRPL_OT_CLUSTERING:-1}"
[[ "$CRPL_OT_CLUSTERING" == 0 || "$CRPL_OT_CLUSTERING" == 1 ]]
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLBACKEND=Agg

[[ "$(git -C "$CRPL_SOURCE" rev-parse HEAD)" == "$CRPL_REVISION" ]]
[[ -z "$(git -C "$CRPL_SOURCE" status --porcelain)" ]]
[[ -d "$DATA_ROOT" && -x "$CRPL_PYTHON" ]]
if [[ -e "$RUN_ROOT" ]]; then
  echo "Output already exists; preserve it and choose a new CRPL_OFFICIAL_RUN_ROOT: $RUN_ROOT" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT"
trap 'code=$?; echo "$(date -Iseconds) exit_code=$code" > "$RUN_ROOT/queue_exit.txt"' EXIT
printf '%s\n' "$BASH_EXECUTION_STRING" > "$RUN_ROOT/launcher_snapshot.sh"
cp "$REPO_ROOT/scripts/summarize_crpl_official.py" "$RUN_ROOT/summarize_snapshot.py"
mkdir -p "$RUN_ROOT/author_code" "$RUN_ROOT/data_parent"
git -C "$CRPL_SOURCE" archive "$CRPL_REVISION" | tar -x -C "$RUN_ROOT/author_code"
if [[ "$CRPL_CONTROL_VARIANT" != unmodified ]]; then
  CRPL_CEILING="$(cd "$RUN_ROOT" && pwd)"
  CRPL_PATCH="$REPO_ROOT/patches/crpl_init_checked.patch"
  GIT_CEILING_DIRECTORIES="$CRPL_CEILING" git -C "$RUN_ROOT/author_code" apply --check "$CRPL_PATCH"
  GIT_CEILING_DIRECTORIES="$CRPL_CEILING" git -C "$RUN_ROOT/author_code" apply "$CRPL_PATCH"
  cp "$REPO_ROOT/scripts/crpl_numerical_guard.py" "$RUN_ROOT/author_code/numerical_guard.py"
  export CRPL_CHECKS_ROOT="$RUN_ROOT"
fi
if [[ "$CRPL_CONTROL_VARIANT" == protocol_checked ]]; then
  CRPL_PROTOCOL_PATCH="$REPO_ROOT/patches/crpl_protocol_checked.patch"
  GIT_CEILING_DIRECTORIES="$CRPL_CEILING" git -C "$RUN_ROOT/author_code" apply --check "$CRPL_PROTOCOL_PATCH"
  GIT_CEILING_DIRECTORIES="$CRPL_CEILING" git -C "$RUN_ROOT/author_code" apply "$CRPL_PROTOCOL_PATCH"
  cp "$REPO_ROOT/scripts/crpl_task_data_audit.py" "$RUN_ROOT/author_code/task_data_audit.py"
  cp "$REPO_ROOT/scripts/crpl_metrics.py" "$RUN_ROOT/author_code/official_metrics.py"
fi
# The author appends OfficeHome/ to --data_root. Use a link, not a loader patch.
ln -s "$(realpath "$DATA_ROOT")" "$RUN_ROOT/data_parent/OfficeHome"
printf '%s\n' "$CRPL_REVISION" > "$RUN_ROOT/UPSTREAM_REVISION"
echo "$(date -Iseconds) start pid=$$ gpu=$CUDA_VISIBLE_DEVICES" > "$RUN_ROOT/queue_start.txt"
command=("$CRPL_PYTHON" -u main.py --M1 16 --M2 16
         --data_root "$RUN_ROOT/data_parent/" --dataset OfficeHome
         --output_dir "$RUN_ROOT/outputs/" --evaluation_step 200
         --prompt_learning_rate 0.005 --t_weight 0.5
         --enhanced_pseudo_label 1 --OT_clustering "$CRPL_OT_CLUSTERING" --training_mode multi-source)
printf '%q ' "${command[@]}" > "$RUN_ROOT/command.sh"
printf '\n' >> "$RUN_ROOT/command.sh"
"$CRPL_PYTHON" - "$RUN_ROOT" "$CRPL_REVISION" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
record = {
    "upstream_commit": sys.argv[2],
    "author_python_code_modified": os.environ["CRPL_CONTROL_VARIANT"] != "unmodified",
    "control_variant": os.environ["CRPL_CONTROL_VARIANT"],
    "algorithm_changes": (["initialize ctx_source_combined with Normal(0, 0.02)"]
                          if os.environ["CRPL_CONTROL_VARIANT"] != "unmodified" else []),
    "protocol_fixes": (["reset both data iterators before each target task", "assert dataset ownership and labelled-source exclusion every update", "remove target-GT pseudo-label diagnostics"]
                       if os.environ["CRPL_CONTROL_VARIANT"] == "protocol_checked" else []),
    "numerical_checks": ("every step: finite losses, present/finite gradients, finite gradient norms, finite updated parameters; final buffers"
                         if os.environ["CRPL_CONTROL_VARIANT"] != "unmodified" else None),
    "protocol": "multi-source single-target; four targets in one process",
    "seed": 1,
    "seed_policy": "author default; seed once before training all four targets",
    "settings": "author README OfficeHome multi-source command",
    "ablation": "SPL only" if os.environ["CRPL_OT_CLUSTERING"] == "0" else "SPL with OT clustering",
    "enhanced_pseudo_label": 1,
    "OT_clustering": int(os.environ["CRPL_OT_CLUSTERING"]),
    "loss_semantics": ("source CE + 0.5 * SPL soft CE; no OT cost or hard self-label CE"
                       if os.environ["CRPL_OT_CLUSTERING"] == "0" else
                       "source CE + 0.5 * mean(SPL soft CE, hard self-label CE) + 0.5 * OT cost"),
    "runtime_sha256": {
        name: hashlib.sha256((root / "author_code" / name).read_bytes()).hexdigest()
        for name in ("main.py", "model.py", "dataloader.py", "dataset.py", "samplers.py", "numerical_guard.py", "task_data_audit.py", "official_metrics.py")
        if (root / "author_code" / name).exists()
    },
    "dependencies": {
        name: importlib.metadata.version(name)
        for name in ("torch", "torchvision", "POT", "numpy", "scipy", "torchinfo")
    },
    "environment": {
        name: os.environ[name]
        for name in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    },
    "evaluation": "author batch 30, shuffle=True, drop_last=True; temporal text-feature average",
}
(root / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
PY
cd "$RUN_ROOT/author_code"
# One invocation is essential: do not reset the seed between target domains.
"${command[@]}" 2>&1 | tee "$RUN_ROOT/console.log"
"$CRPL_PYTHON" "$RUN_ROOT/summarize_snapshot.py" "$RUN_ROOT"
