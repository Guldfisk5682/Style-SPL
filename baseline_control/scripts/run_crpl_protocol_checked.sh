#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CRPL_CONTROL_VARIANT=protocol_checked
export CRPL_OT_CLUSTERING=0
export CRPL_OFFICIAL_RUN_ROOT="${CRPL_OFFICIAL_RUN_ROOT:-$PWD/runs/crpl_spl_protocol_checked_20261009_seed1}"
exec bash scripts/run_crpl_official.sh
