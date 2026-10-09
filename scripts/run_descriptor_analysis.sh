#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/style-spl-mpl
STYLE_PYTHON="$HOME/workspace/Domain_Adaptation/.venv-crpl/bin/python"
STYLE_ROOT="$HOME/workspace/Style-SPL/runs/descriptor_20261010_seed1"
STYLE_ANALYSIS="$STYLE_ROOT/analysis"
STYLE_IMAGE_CACHE="$HOME/workspace/Style-SPL/runs/analysis_round_20261009/image_embeddings"
mkdir -p "$STYLE_ANALYSIS"
trap 'code=$?; echo "$(date -Iseconds) exit_code=$code" > "$STYLE_ANALYSIS/exit.txt"' EXIT
while true; do
    STYLE_READY=1
    for STYLE_ARM in r4_shuffled r4_random; do
        if [[ -f "$STYLE_ROOT/queues/$STYLE_ARM/exit.txt" ]]; then
            if [[ "$(< "$STYLE_ROOT/queues/$STYLE_ARM/exit.txt")" != *exit_code=0 ]]; then
                cat "$STYLE_ROOT/queues/$STYLE_ARM/exit.txt"
                exit 1
            fi
        else
            STYLE_READY=0
        fi
    done
    [[ "$STYLE_READY" == 1 ]] && break
    sleep 15
done
"$STYLE_PYTHON" scripts/verify_descriptor_pairing.py --root "$STYLE_ROOT" --output "$STYLE_ANALYSIS/pairing_verification.json"
"$STYLE_PYTHON" scripts/analyze_descriptor_predictions.py --root "$STYLE_ROOT" --image_cache "$STYLE_IMAGE_CACHE" --output "$STYLE_ANALYSIS/paired_predictions.json"
"$STYLE_PYTHON" scripts/analyze_source_routing_headroom.py --root "$STYLE_ROOT" --image_cache "$STYLE_IMAGE_CACHE" --output "$STYLE_ANALYSIS/source_routing_headroom.json" --arms r4_real r4_shuffled r4_random
"$STYLE_PYTHON" scripts/analyze_descriptor_geometry.py --bank_root "$HOME/workspace/Style-SPL/runs/s1_protocol_checked_20261009_seed1" --output "$STYLE_ANALYSIS/descriptor_geometry.json"
"$STYLE_PYTHON" scripts/summarize_controlled_round.py --run_root "$STYLE_ROOT" --output "$STYLE_ANALYSIS" --arms r4_real r4_shuffled r4_random --descriptor_controls
cp "$STYLE_ROOT/gate/acceptance.json" "$STYLE_ANALYSIS/acceptance.json"
cp "$STYLE_ROOT/gate/unit_tests.log" "$STYLE_ANALYSIS/unit_tests.log"
cp configs/descriptor_experiment_plan.json "$STYLE_ANALYSIS/experiment_plan.json"
"$STYLE_PYTHON" scripts/plot_controlled_round.py "$STYLE_ANALYSIS" --arms r4_real r4_shuffled r4_random --labels 'R4 / Real' 'R4 / Shuffled' 'R4 / Random' --actual_domain_swaps
"$STYLE_PYTHON" scripts/build_descriptor_report.py "$STYLE_ANALYSIS"
echo "completed" > "$STYLE_ANALYSIS/phase.txt"
