#!/usr/bin/env bash
# From-BASE ALFWorld GiGPO — parameterized for the pure-RL vs RL+scaffold experiment.
# Phase $1: none|full|weighted (scaffold injection). $2: cumulative step target.
# Env knobs: ALF_GPUS (default 0,1), ALF_MODEL (default base 1.5B-Instruct),
#            ALF_RAY_TMP (default /dev/shm/zray_alf), EXP, TRAIN_DATA_SIZE, TEST_FREQ,
#            SAVE_FREQ, EVAL_SPLIT, VAL_ONLY, VAL_BEFORE, VAL_N, ENV_SEED, SKILL_PATH.
set -x
PHASE=${1:-none}
TOTAL_EPOCHS=${2:-300}
# tmpfs copy of the env when it is staged (see .verl_env_done), else the NFS original.
if [ -f /dev/shm/.verl_env_done ]; then VENV=/dev/shm/verl_env; else VENV=/mnt/data1/zha00175/miniconda/envs/verl; fi
export CUDA_VISIBLE_DEVICES=${ALF_GPUS:-0,1}
# 18412 small game files. Reading them from NFS with ~140 concurrent Ray env workers is
# what pinned this box (load 567, 324 processes stuck in uninterruptible I/O) and made
# worker registration time out. Point this at a tmpfs copy to keep the eval off NFS; the
# files are identical, so which games run and in what order does not change.
export ALFWORLD_DATA=${ALFWORLD_DATA:-/mnt/data1/zha00175/skillzero-env/alfworld_data}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PATH=$VENV/bin:$PATH
export CUDA_HOME=$VENV
export TORCH_CUDA_ARCH_LIST=9.0
export PYTHONUNBUFFERED=1
export HF_HOME=/mnt/data1/zha00175/hf_home

export GIGPO_SKILL_PATH=${SKILL_PATH:-/mnt/data1/zha00175/gigpo_helper_skillopt/skills_general.json}
case "$PHASE" in
  full)     export GIGPO_SKILL_MODE=full;  export GIGPO_SKILL_FORCE=1; export GIGPO_SKILL_P=1.0 ;;
  half)     export GIGPO_SKILL_MODE=full;  export GIGPO_SKILL_FORCE=0; export GIGPO_SKILL_P=0.5 ;;
  none)     export GIGPO_SKILL_MODE=none;  export GIGPO_SKILL_FORCE=0; export GIGPO_SKILL_P=0.0 ;;
  weighted) export GIGPO_SKILL_MODE=full;  export GIGPO_SKILL_FORCE=0; unset GIGPO_SKILL_P ;;
esac

PY=$VENV/bin/python
MODEL=${ALF_MODEL:-/mnt/data1/zha00175/models/Qwen2.5-1.5B-Instruct}
EXP=${EXP:-alf_frombase}
CKPT_DIR=/mnt/data1/zha00175/gigpo_helper_ckpts/$EXP
RAY_TMP=${ALF_RAY_TMP:-/dev/shm/zray_alf}
mkdir -p "$RAY_TMP"
cd /mnt/data1/zha00175/verl-agent || exit 1

train_data_size=${TRAIN_DATA_SIZE:-32}; val_data_size=${VAL_BS:-128}; group_size=8
$PY -m examples.data_preprocess.prepare --mode 'text' --train_data_size $train_data_size --val_data_size $val_data_size

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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BS:-32} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BS:-32} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE:-2} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM:-0.6} \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N:-1} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BS:-32} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_OFFLOAD:-True} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPT_OFFLOAD:-True} \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=mean_std_norm \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=${ENV_SEED:-0} \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.alfworld.eval_dataset=${EVAL_SPLIT:-eval_in_distribution} \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='gigpo_frombase' \
    trainer.experiment_name=$EXP \
    trainer.default_local_dir=$CKPT_DIR \
    +ray_init._temp_dir=$RAY_TMP \
    +ray_init.include_dashboard=False \
    trainer.n_gpus_per_node=${N_GPUS:-2} \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ:-10} \
    trainer.max_actor_ckpt_to_keep=${MAX_CKPT:-40} \
    trainer.test_freq=${TEST_FREQ:-10} \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.resume_mode=auto \
    trainer.val_only=${VAL_ONLY:-False} \
    trainer.val_before_train=${VAL_BEFORE:-True} "${@:3}"
