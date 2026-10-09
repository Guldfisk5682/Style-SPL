#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
STYLE_TMUX="$HOME/workspace/domain_adaptation_tools/bin/tmux"
STYLE_SESSION=style_spl_officehome
[[ ! -e runs/b0_officehome_seed1 && ! -e runs/s1_officehome_seed1 ]]
if "$STYLE_TMUX" has-session -t "$STYLE_SESSION" 2>/dev/null; then
  echo "Session already exists: $STYLE_SESSION" >&2
  exit 1
fi
"$STYLE_TMUX" new-session -d -s "$STYLE_SESSION" -c "$PWD" \
  'exec bash -c "$(cat scripts/run_paired_officehome.sh)" "$PWD/scripts/run_paired_officehome.sh"'
"$STYLE_TMUX" set-option -t "$STYLE_SESSION" remain-on-exit on
echo "Session: $STYLE_SESSION"
