#!/usr/bin/env bash
# After the arm finishes: the plain-RL baseline (same data, same hyperparameters,
# injection off) under a wall-clock budget, so a reclaim policy does not take the
# node and the comparison run gets produced anyway.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ARM_ROOT="${ARM_ROOT:-$(cd "$HERE/.." && pwd)}"
source "$ARM_ROOT/autoscaffold/env.sh"
HOURS="${ARM_KEEPALIVE_HOURS:-24}"
DEADLINE=$(( $(date +%s) + HOURS * 3600 ))
export ARM_EXP="${ARM_KEEPALIVE_EXP:-${ARM_EXP}_plainrl}"   # never the arm's own dirs
BASE_LOG="$ARM_EXP_ROOT/$ARM_EXP/train.log"
mkdir -p "$(dirname "$BASE_LOG")" || { echo "cannot create $(dirname "$BASE_LOG")"; exit 1; }
steps="${ARM_KEEPALIVE_STEPS:-${ARM_TARGET_STEP:-300}}"
echo "[keepalive] plain RL as $ARM_EXP to step $steps, budget ${HOURS}h, log $BASE_LOG"
fast=0
while (( $(date +%s) < DEADLINE )); do
  t0=$(date +%s)
  SAVE_FREQ=50 TEST_FREQ=10 VAL_BEFORE=True \
    bash "$ARM_ROOT/autoscaffold/train_alfworld.sh" none "$steps" >> "$BASE_LOG" 2>&1
  rc=$?; ran=$(( $(date +%s) - t0 ))
  echo "[keepalive] exited rc=$rc after ${ran}s"
  if (( rc != 0 && ran < 60 )); then
    fast=$(( fast + 1 )); (( fast >= 3 )) && { echo "[keepalive] 3 fast failures — stopping"; exit 1; }
  else fast=0; fi
  (( $(date +%s) < DEADLINE )) || break
  sleep 120
done
echo "[keepalive] budget reached"
