#!/usr/bin/env bash
# Baseline arm (a): NO-SKILL GiGPO baseline, Qwen2.5-3B-Instruct on ALFWorld.
# Eval on valid_unseen (OOD) = the project's primary metric (none-mode success).
# wandb offline so per-task val success_rate is readable + syncable to mhong later.
# HARD CONSTRAINT: GPUs 0,1 ONLY. Never 6,7.
set -x

EXP=baseline_a_qwen2.5_3b_unseen

# --- env / hardware ---
export CUDA_VISIBLE_DEVICES=0,1
export ALFWORLD_DATA=/mnt/data1/zha00175/skillzero-env/alfworld_data
# vllm 0.11 V1: XFORMERS paged-flash needs block_size%256==0 (default 16) -> crash; FLASH_ATTN is fine.
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# conda env bin first so flashinfer JIT finds ninja + nvcc 12.8 (matches torch cu128); cicc relative to nvcc.
VENV=/mnt/data1/zha00175/miniconda/envs/verl
export PATH=$VENV/bin:$PATH
export CUDA_HOME=$VENV
export TORCH_CUDA_ARCH_LIST=9.0  # H200 = Hopper sm_90

# wandb offline (no login needed for autonomous run); sync later: wandb sync -e mhong-university-of-minnesota
export WANDB_MODE=offline
export WANDB_DIR=/mnt/data1/zha00175/gigpo_helper_wandb

PY=$VENV/bin/python
REPO=/mnt/data1/zha00175/verl-agent
MODEL=/mnt/data1/zha00175/skillzero-env/models/Qwen2.5-3B-Instruct
CKPT_DIR=/mnt/data1/zha00175/gigpo_helper_ckpts

cd "$REPO" || exit 1

ENGINE=${1:-vllm}
num_cpus_per_env_worker=0.1

train_data_size=16
val_data_size=128
group_size=8
mode="mean_std_norm"

$PY -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

$PY -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.alfworld.eval_dataset=eval_out_of_distribution \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='gigpo_helper' \
    trainer.experiment_name=$EXP \
    trainer.default_local_dir=$CKPT_DIR/$EXP \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
