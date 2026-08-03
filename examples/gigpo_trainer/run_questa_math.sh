#!/usr/bin/env bash
# QuestA-style math RL on OpenMath-Nemotron-1.5B, scaled to 2 GPUs.
#
#   bash run_questa_math.sh <arm> <to_step> [extra hydra overrides...]
#     arm = bare  -> no hint at all                (QuestA's "Hard-Nemotron-1.5B", the floor)
#         = p50   -> every prompt carries 50% of the reference solution
#         = p25   -> every prompt carries 25%
#         = scaf  -> per-instance alpha from scaffold.json (our Teacher-scheduled arm)
#         = eval  -> standalone eval only, ALWAYS unhinted
#
# MATCHES QuestA (AReaL/scripts/partial_50_grpo.yaml):
#   base OpenMath-Nemotron-1.5B, full-parameter FSDP bf16 + gradient checkpointing,
#   max_response_length 24000, max_prompt_length 4096, sampling temperature 1.0,
#   AdamW lr 2e-5 CONSTANT, weight_decay 0.05, grad_clip 1.0, clip_ratio 0.2,
#   NO KL (kl_ctl 0.0), no entropy bonus, GRPO, 1 PPO epoch per rollout batch.
#
# DELIBERATE DEVIATIONS (all forced by having 2 GPUs instead of 64) — keep this list honest,
# every number we report is under these, not QuestA's:
#   1. 200 steps, not 2000.        QuestA's stage-1 is exactly 100 steps, so 200 = their full
#                                  stage 1 plus 100 steps of stage 2.
#   2. batch 32 x 8 rollouts, not 128 x 16.
#   3. verl + FSDP + vLLM (sync), not AReaL + sglang (async, off-policyness 4).
#   4. NO DAPO success-rate filtering (0.05/0.95): `algorithm.filter_groups` is declared in
#      this verl fork's yaml but NOT implemented. Degenerate groups still zero out via GRPO's
#      std normalization, but the batch is NOT resampled to refill, so the effective batch
#      shrinks when groups are all-correct/all-wrong.
#   5. AdamW betas stay verl's default (0.9, 0.999); QuestA uses (0.9, 0.95).
#   6. QuestA's reward_scaling 5.0 has no verl equivalent; GRPO normalizes by group std, so a
#      constant factor cancels.
set -x
ARM=${1:-bare}
TO_STEP=${2:-200}
shift 2 || true

VENV=/mnt/data1/zha00175/miniconda/envs/verl
export CUDA_VISIBLE_DEVICES=${QM_GPUS:-6,7}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PATH=$VENV/bin:$PATH
export CUDA_HOME=$VENV
export TORCH_CUDA_ARCH_LIST=9.0
export PYTHONUNBUFFERED=1
export HF_HOME=/mnt/data1/zha00175/hf_home
# NFS-cold-cache guard (Ray agents fate-share with the raylet on a 30s default deadline).
export RAY_agent_register_timeout_ms=${RAY_AGENT_TIMEOUT_MS:-300000}

PY=$VENV/bin/python
MODEL=${QM_MODEL:-/mnt/data1/zha00175/models/OpenMath-Nemotron-1.5B}
DATA=${QM_DATA:-/mnt/data1/zha00175/questa_scaffold_data}
EXP=${EXP:-questa_${ARM}}
CKPT_DIR=${QM_CKPT:-/mnt/data1/zha00175/questa_ckpts/$EXP}
RAY_TMP=${QM_RAY_TMP:-/dev/shm/zray_$EXP}
mkdir -p "$RAY_TMP" "$CKPT_DIR"
cd /mnt/data1/zha00175/verl-agent

# --- which prompts this arm trains on -------------------------------------------------
# QM_POOL=union (default): every arm draws the SAME 8,402 problems and only the hint ratio
#   differs. This is what makes an arm comparison about the SCHEDULE rather than the data.
# QM_POOL=faithful: the released per-ratio pools (1,752 at p50 / 10,126 at p25), i.e. exactly
#   what QuestA trains on. Use this to reproduce them, not to compare arms — the two pools are
#   difficulty-filtered at different ratios, so the problem sets are not the same.
POOL=${QM_POOL:-union}
case "$POOL" in
  union)    F50=$DATA/union_p50.parquet;  F25=$DATA/union_p25.parquet ;;
  faithful) F50=$DATA/train_p50.parquet;  F25=$DATA/train_p25.parquet ;;
  *) echo "unknown QM_POOL '$POOL' (union|faithful)"; exit 2 ;;
esac
# `bare` and `scaf` always read the unhinted union: the alpha layer rebuilds each prompt at
# __getitem__ time from extra_info.solution, so no ratio is frozen on disk for our arm.
case "$ARM" in
  bare) TRAIN=$DATA/union_bare.parquet ;;
  p50)  TRAIN=$F50 ;;
  p25)  TRAIN=$F25 ;;
  scaf) TRAIN=$DATA/union_bare.parquet
        export QUESTA_SCAFFOLD_PATH=${SCAFFOLD_PATH:-/mnt/data1/zha00175/exp_questa/scaf/scaffold.json} ;;
  eval) TRAIN=$DATA/union_bare.parquet ;;
  *) echo "unknown arm '$ARM' (bare|p50|p25|scaf|eval)"; exit 2 ;;
esac
VAL=$DATA/val.parquet          # ALWAYS unhinted — the only number comparable across arms

# The scaf arm re-renders each prompt at the Teacher's per-problem alpha; every other arm
# uses verl's stock dataset so nothing can silently differ between them.
CUSTOM_CLS=""
[ "$ARM" = "scaf" ] && CUSTOM_CLS="data.custom_cls.path=agent_system/skill_opt/mathscaffold/dataset.py data.custom_cls.name=AlphaRLHFDataset"

VAL_ONLY_FLAG=False; VAL_BEFORE_FLAG=${VAL_BEFORE:-True}
[ "$ARM" = "eval" ] && { VAL_ONLY_FLAG=True; VAL_BEFORE_FLAG=True; }

$PY -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.use_kl_in_reward=False \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  data.train_files=$TRAIN \
  data.val_files=$VAL \
  data.train_batch_size=${TRAIN_BS:-32} \
  data.val_batch_size=${VAL_BS:-442} \
  data.max_prompt_length=${MAX_PROMPT:-4096} \
  data.max_response_length=${MAX_RESP:-24000} \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  data.shuffle=True \
  custom_reward_function.path=agent_system/skill_opt/mathscaffold/reward_fn.py \
  custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path=$MODEL \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=${LR:-2e-5} \
  actor_rollout_ref.actor.optim.weight_decay=${WD:-0.05} \
  actor_rollout_ref.actor.optim.warmup_style=constant \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BS:-32} \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAXTOK:-32768} \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=${ROLLOUT_N:-8} \
  actor_rollout_ref.rollout.temperature=${TEMP:-1.0} \
  actor_rollout_ref.rollout.top_p=${TOPP:-1.0} \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${TP:-2} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM:-0.65} \
  actor_rollout_ref.rollout.max_model_len=${MAXLEN:-28672} \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAXTOK:-32768} \
  actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMP:-0.7} \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=${VAL_N_SAMPLES:-8} \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAXTOK:-32768} \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.critic_warmup=0 \
  "trainer.logger=[console]" \
  trainer.project_name=questa_math \
  trainer.experiment_name=$EXP \
  trainer.default_local_dir=$CKPT_DIR \
  +ray_init._temp_dir=$RAY_TMP \
  +ray_init.include_dashboard=False \
  trainer.n_gpus_per_node=${N_GPUS:-2} \
  trainer.nnodes=1 \
  trainer.save_freq=${SAVE_FREQ:-20} \
  trainer.max_actor_ckpt_to_keep=${MAX_CKPT:-20} \
  trainer.test_freq=${TEST_FREQ:-20} \
  trainer.total_training_steps=$TO_STEP \
  trainer.total_epochs=${TOTAL_EPOCHS:-100} \
  trainer.resume_mode=auto \
  trainer.val_only=$VAL_ONLY_FLAG \
  trainer.val_before_train=$VAL_BEFORE_FLAG \
  $CUSTOM_CLS \
  "$@"
