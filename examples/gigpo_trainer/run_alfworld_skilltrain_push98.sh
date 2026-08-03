#!/usr/bin/env bash
# Phase 3: GiGPO continued-training of the GiGPO-TRAINED 1.5B, WITH the diagnosed
# general skill injected, for curriculum withdrawal. Phase via $1: full|half|none.
# Resume-chaining: same default_local_dir + resume_mode=auto -> phases continue the
# same model/optimizer, only the skill-injection probability changes (1 -> 0.5 -> 0).
# GPUs 0,1 only.
set -x
PHASE=${1:-full}
TOTAL_EPOCHS=${2:-150}          # cumulative step target (1 game-batch/epoch here)

VENV=/mnt/data1/zha00175/miniconda/envs/verl
export CUDA_VISIBLE_DEVICES=0,1
export ALFWORLD_DATA=/mnt/data1/zha00175/skillzero-env/alfworld_data
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PATH=$VENV/bin:$PATH
export CUDA_HOME=$VENV
export TORCH_CUDA_ARCH_LIST=9.0
export PYTHONUNBUFFERED=1

# --- skill injection (P1 in env_manager) ---
export GIGPO_SKILL_PATH=${SKILL_PATH:-/mnt/data1/zha00175/gigpo_helper_skillopt/skills_push98.json}
case "$PHASE" in
  full) export GIGPO_SKILL_MODE=full;  export GIGPO_SKILL_FORCE=1; export GIGPO_SKILL_P=1.0 ;;   # p=1 always inject
  half) export GIGPO_SKILL_MODE=full;  export GIGPO_SKILL_FORCE=0; export GIGPO_SKILL_P=0.5 ;;   # group-level Bernoulli(0.5)
  none) export GIGPO_SKILL_MODE=none;  export GIGPO_SKILL_FORCE=0; export GIGPO_SKILL_P=0.0 ;;   # withdrawn
  weighted) export GIGPO_SKILL_MODE=full; export GIGPO_SKILL_FORCE=0; unset GIGPO_SKILL_P ;;     # per-task p_task from skill json (oversample weak tasks)
esac

PY=$VENV/bin/python
MODEL=/mnt/data1/zha00175/ckpts_alfworld/verl_agent_alfworld/0621_1310_gigpo_Qwen2.5-1.5B-Instruct_full_g8_b16_lr1e-6/global_step_150/actor/huggingface
EXP=${EXP:-skilltrain_push98}
CKPT_DIR=/mnt/data1/zha00175/gigpo_helper_ckpts/$EXP
cd /mnt/data1/zha00175/verl-agent || exit 1

train_data_size=${TRAIN_DATA_SIZE:-16}; val_data_size=128; group_size=8
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N:-1} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
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
    env.alfworld.eval_dataset=${EVAL_SPLIT:-eval_out_of_distribution} \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='gigpo_helper' \
    trainer.experiment_name=$EXP \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ:-10} \
    trainer.max_actor_ckpt_to_keep=${MAX_CKPT:-8} \
    trainer.test_freq=${TEST_FREQ:-5} \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.resume_mode=auto \
    trainer.val_only=${VAL_ONLY:-False} \
    trainer.val_before_train=${VAL_BEFORE:-True} "${@:3}"
