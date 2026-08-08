#!/usr/bin/env bash
# The training subprocess for one autoscaffold cycle (or the plain-RL baseline).
#
#   bash autoscaffold/train_alfworld.sh <scaffold|none> <total_steps> [extra hydra args...]
#
# Adapted from examples/gigpo_trainer/run_alfworld.sh (which is NOT modified). Every
# machine-specific value resolves through autoscaffold/env.sh; nothing here names a
# path that exists on only one machine.
set -eo pipefail

MODE="${1:?mode: scaffold|none}"
TOTAL_STEPS="${2:?absolute step to train to}"
shift 2 || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ARM_ROOT="${ARM_ROOT:-$(cd "$HERE/.." && pwd)}"
source "$ARM_ROOT/autoscaffold/env.sh"
cd "$ARM_ROOT"

PY="${ARM_PYTHON:?}"
EXP="${ARM_EXP:?}"
STATE_DIR="$ARM_EXP_ROOT/$EXP"
mkdir -p "$STATE_DIR"

train_data_size=${TRAIN_DATA_SIZE:-32}
val_data_size=${VAL_BS:-128}
group_size=${ARM_ROLLOUT_N:-8}
# One PPO update per step over the whole rollout batch, derived so changing the prompt
# count cannot silently change the algorithm.
ppo_mini_bs=${PPO_MINI_BS:-$(( train_data_size * group_size ))}
micro_bs=${MICRO_BS:-32}

# The scaffold hooks. mode=none leaves every flag unset: the run is byte-identical to
# upstream (the baseline the experiment compares against).
SCAFFOLD_ARGS=()
if [[ "$MODE" == "scaffold" ]]; then
  export AUTOSCAFFOLD_ALFWORLD=1
  export AUTOSCAFFOLD_SCAFFOLD="$STATE_DIR/scaffold.json"
  export AUTOSCAFFOLD_ROLLOUT_LOG="${AUTOSCAFFOLD_ROLLOUT_LOG:-$STATE_DIR/rollouts.jsonl}"
  export AUTOSCAFFOLD_SEED="${ENV_SEED:-0}"
  SCAFFOLD_ARGS+=(
    # OFF by default (decision 2026-08-06): the loss conditions on the SAME prompt the
    # rollout used. ARM_BARE_LOSS=True re-enables the bare-prompt swap as an ablation.
    "algorithm.bare_prompt_loss.enable=${ARM_BARE_LOSS:-False}"
    "algorithm.bare_prompt_loss.mode=both"
    # make_envs runs inside a Ray actor; runtime_env carries the flags even when a
    # pre-existing cluster would not inherit shell exports
    # hydra parses unquoted numerics as ints and Ray requires env_vars values to be
    # STRINGS — '1' failed runtime_env validation on the first scaffold smoke
    "+ray_init.runtime_env.env_vars.AUTOSCAFFOLD_ALFWORLD='1'"
    "+ray_init.runtime_env.env_vars.AUTOSCAFFOLD_SCAFFOLD='$AUTOSCAFFOLD_SCAFFOLD'"
    "+ray_init.runtime_env.env_vars.AUTOSCAFFOLD_ROLLOUT_LOG='$AUTOSCAFFOLD_ROLLOUT_LOG'"
    "+ray_init.runtime_env.env_vars.AUTOSCAFFOLD_SEED='$AUTOSCAFFOLD_SEED'"
  )
fi

KL_ARGS=()
if [[ "${ARM_KL:-0.01}" == "0" ]]; then
  KL_ARGS+=("actor_rollout_ref.actor.use_kl_loss=False")
else
  KL_ARGS+=("actor_rollout_ref.actor.use_kl_loss=True"
            "actor_rollout_ref.actor.kl_loss_coef=${ARM_KL:-0.01}"
            "actor_rollout_ref.actor.kl_loss_type=low_var_kl")
fi

# Only prepare data when the parquet is missing or the size changed; the file only
# indicates modality and row count.
# Overridable: $HOME is a shared mount on many clusters.
DATA_DIR="${ARM_DATA_DIR:-$HOME/data/verl-agent/text}"
if [[ ! -f "$DATA_DIR/train.parquet" || "$(cat "$DATA_DIR/.size" 2>/dev/null)" != "$train_data_size/$val_data_size" ]]; then
  "$PY" -m examples.data_preprocess.prepare --mode 'text' \
      --train_data_size "$train_data_size" --val_data_size "$val_data_size"
  echo "$train_data_size/$val_data_size" > "$DATA_DIR/.size"
fi

export RAY_TMPDIR="${ARM_RAY_TMP:?}"
mkdir -p "$RAY_TMPDIR"
unset RAY_ADDRESS   # never silently attach to a stale cluster

"$PY" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.dataloader_num_workers="${DATALOADER_WORKERS:-0}" \
    actor_rollout_ref.model.path="$ARM_MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_bs \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_bs \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_bs \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ARM_TP:-2}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ARM_GPU_MEM:-0.6}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_bs \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.actor.checkpoint.contents="['model','optimizer','extra']" \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=mean_std_norm \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed="${ENV_SEED:-0}" \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.alfworld.eval_dataset="${ARM_EVAL_SPLIT:-eval_in_distribution}" \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger="$ARM_TRAINER_LOGGER" \
    trainer.project_name="${ARM_WANDB_PROJECT:-verl_agent_alfworld_inspect}" \
    trainer.experiment_name="$EXP" \
    trainer.n_gpus_per_node="${ARM_N_GPUS:-2}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ:-10}" \
    trainer.test_freq="${TEST_FREQ:-99999}" \
    trainer.val_before_train="${VAL_BEFORE:-False}" \
    trainer.val_only="${VAL_ONLY:-False}" \
    trainer.total_epochs=99999 \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.default_local_dir="$ARM_CKPT_ROOT/$EXP" \
    trainer.resume_mode=auto \
    "${KL_ARGS[@]}" "${SCAFFOLD_ARGS[@]}" "$@"
