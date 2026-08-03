#!/usr/bin/env bash
# Math auto-scaffold arm. Hyperparameters mirror OC-GRPO (arXiv 2607.19313, App. G) so our
# numbers stay comparable to their Tables 2-4, with ONE deliberate difference:
#
#   NO importance-sampling correction.  We keep vanilla GRPO's ratio and let the Teacher's
#   scaffold + per-scope injection probability p be the only mechanism. In their Table 1 that
#   places us in the "guided sampling, no off-context correction" row. Their 1.5B/3B results
#   say that row can fall BELOW vanilla GRPO, so vanilla GRPO is the baseline that matters.
#
# Usage:  bash run_math_scaffold.sh <phase> <to_step> [extra hydra overrides...]
#   phase = weighted  -> scaffold injected per scaffold.json (training)
#   phase = none      -> nothing injected (standalone eval; the only comparable number)
set -x
PHASE=${1:-weighted}
TO_STEP=${2:-10}
shift 2 || true

VENV=/mnt/data1/zha00175/miniconda/envs/verl
export CUDA_VISIBLE_DEVICES=${MATH_GPUS:-2,6}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PATH=$VENV/bin:$PATH
export CUDA_HOME=$VENV
export TORCH_CUDA_ARCH_LIST=9.0
export PYTHONUNBUFFERED=1
export HF_HOME=/mnt/data1/zha00175/hf_home
# NFS-cold-cache guard: Ray's agents fate-share with the raylet on a 30s default deadline.
export RAY_agent_register_timeout_ms=${RAY_AGENT_TIMEOUT_MS:-300000}

export GIGPO_SKILL_PATH=${SKILL_PATH:-/mnt/data1/zha00175/exp_mathscaffold/math_arm/scaffold.json}
case "$PHASE" in
  weighted) export GIGPO_SKILL_MODE=full; export GIGPO_SKILL_FORCE=0; unset GIGPO_SKILL_P ;;
  none)     export GIGPO_SKILL_MODE=none ;;
  *) echo "unknown phase '$PHASE'"; exit 2 ;;
esac

PY=$VENV/bin/python
MODEL=${MATH_MODEL:-/mnt/data1/zha00175/models/Qwen2.5-1.5B-Instruct}
EXP=${EXP:-math_scaffold_1p5b}
DATA_DIR=${MATH_DATA:-/mnt/data1/zha00175/math_scaffold_data}
CKPT_DIR=/mnt/data1/zha00175/math_scaffold_ckpts/$EXP
RAY_TMP=${MATH_RAY_TMP:-/dev/shm/zray_$EXP}
mkdir -p "$RAY_TMP" "$CKPT_DIR"
cd /mnt/data1/zha00175/verl-agent

# --- paper-matched knobs -------------------------------------------------------------
TRAIN_BS=${TRAIN_BS:-32}          # 32 prompts per step
ROLLOUT_N=${ROLLOUT_N:-16}        # N = 16 rollouts per prompt
MINI_BS=${MINI_BS:-32}            # PPO mini-batch 32, 1 PPO epoch per iteration
LR=${LR:-1e-5}                    # AdamW 1e-5 (LoRA, so higher than a full-FT 1e-6)
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
TEMP=${TEMP:-0.7}                 # rollout temperature 0.7, top-p 0.95
TOPP=${TOPP:-0.95}
MAX_RESP=${MAX_RESP:-2048}

$PY -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files=$DATA_DIR/train_scaffold.parquet \
  data.val_files=$DATA_DIR/val.parquet \
  data.train_batch_size=$TRAIN_BS \
  data.val_batch_size=${VAL_BS:-256} \
  data.max_prompt_length=${MAX_PROMPT:-2048} \
  data.max_response_length=$MAX_RESP \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  custom_reward_function.path=agent_system/skill_opt/mathscaffold/reward_fn.py \
  custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path=$MODEL \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora_rank=$LORA_RANK \
  actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BS \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BS:-8} \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=$ROLLOUT_N \
  actor_rollout_ref.rollout.temperature=$TEMP \
  actor_rollout_ref.rollout.top_p=$TOPP \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${TP:-1} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM:-0.6} \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BS:-8} \
  actor_rollout_ref.rollout.val_kwargs.temperature=$TEMP \
  actor_rollout_ref.rollout.val_kwargs.top_p=$TOPP \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=${VAL_N_SAMPLES:-16} \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BS:-8} \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.critic_warmup=0 \
  "trainer.logger=[console]" \
  trainer.project_name=math_scaffold \
  trainer.experiment_name=$EXP \
  trainer.default_local_dir=$CKPT_DIR \
  +ray_init._temp_dir=$RAY_TMP \
  +ray_init.include_dashboard=False \
  trainer.n_gpus_per_node=${N_GPUS:-2} \
  trainer.nnodes=1 \
  trainer.save_freq=${SAVE_FREQ:-10} \
  trainer.max_actor_ckpt_to_keep=${MAX_CKPT:-60} \
  trainer.test_freq=${TEST_FREQ:-99999} \
  trainer.total_training_steps=$TO_STEP \
  trainer.total_epochs=${TOTAL_EPOCHS:-4} \
  trainer.resume_mode=auto \
  trainer.val_only=${VAL_ONLY:-False} \
  trainer.val_before_train=${VAL_BEFORE:-False} \
  "$@"
