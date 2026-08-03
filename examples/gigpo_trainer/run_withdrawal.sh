#!/usr/bin/env bash
# Phase 3 full run: curriculum withdrawal for the trained-1.5B + general skill.
# 3 chained GiGPO phases (resume_mode=auto continues the same model/optimizer):
#   p=1 (steps 0..N) -> p=0.5 (N..2N) -> p=0 / none-mode (2N..3N).
# Then a clean none-mode eval vs the 0.896 bar happens separately (final_eval).
set -x
N=${1:-30}
VENV=/mnt/data1/zha00175/miniconda/envs/verl
LOGDIR=/mnt/data1/zha00175/gigpo_helper_logs
S=/mnt/data1/zha00175/verl-agent/examples/gigpo_trainer/run_alfworld_skilltrain_1.5b.sh

cleanup() { $VENV/bin/ray stop --force >/dev/null 2>&1; sleep 10; }

echo "[withdrawal $(date +%H:%M:%S)] PHASE 1 p=1 -> step $N (val_before_train)"
bash "$S" full "$N" > "$LOGDIR/withdraw_p1_full.log" 2>&1
cleanup

echo "[withdrawal $(date +%H:%M:%S)] PHASE 2 p=0.5 -> step $((2*N)) (resume)"
bash "$S" half "$((2*N))" trainer.val_before_train=False > "$LOGDIR/withdraw_p2_half.log" 2>&1
cleanup

echo "[withdrawal $(date +%H:%M:%S)] PHASE 3 p=0/none -> step $((3*N)) (resume, withdrawn)"
bash "$S" none "$((3*N))" trainer.val_before_train=False > "$LOGDIR/withdraw_p3_none.log" 2>&1
cleanup

echo "[withdrawal $(date +%H:%M:%S)] DONE - final ckpt at gigpo_helper_ckpts/skilltrain_trained1.5b_withdraw"
