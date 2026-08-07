#!/usr/bin/env bash
# Is the run healthy? One screen, no wandb, no network.
#   bash autoscaffold/status.sh [--exp NAME] [--watch]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ARM_ROOT="${ARM_ROOT:-$(cd "$HERE/.." && pwd)}"
WATCH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp) export ARM_EXP="$2"; shift 2 ;;
    --watch) WATCH=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done
source "$ARM_ROOT/autoscaffold/env.sh"
DIR="$ARM_EXP_ROOT/$ARM_EXP"
show() {
  if [[ ! -f "$DIR/status.json" ]]; then
    echo "no status.json under $DIR (written after the first eval; before that see run_latest.log)"
    return 1
  fi
  "$ARM_PYTHON" -m autoscaffold.monitor "$DIR"
}
if (( WATCH )); then while :; do clear; show; sleep 60; done; else show; fi
