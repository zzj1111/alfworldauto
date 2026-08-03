#!/usr/bin/env bash
# GAP check (pivotal): does the skill open a full-vs-none gap on 3B?
# Pure validation (val_only) on valid_unseen. Run twice: MODE=full and MODE=none.
#   bash run_alfworld_gap_eval.sh full   [MODEL_PATH]
#   bash run_alfworld_gap_eval.sh none   [MODEL_PATH]
# Default MODEL = base Qwen2.5-3B-Instruct (untrained zero-shot gap).
# HARD CONSTRAINT: GPUs 0,1 ONLY.
set -x

MODE=${1:-full}            # full | general_only | none
MODEL=${2:-/mnt/data1/zha00175/skillzero-env/models/Qwen2.5-3B-Instruct}
TAG=${3:-base3b}

VENV=/mnt/data1/zha00175/miniconda/envs/verl
export CUDA_VISIBLE_DEVICES=0,1
export ALFWORLD_DATA=/mnt/data1/zha00175/skillzero-env/alfworld_data
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PATH=$VENV/bin:$PATH
export CUDA_HOME=$VENV
export TORCH_CUDA_ARCH_LIST=9.0
export PYTHONUNBUFFERED=1   # so 'Initial validation metrics: {...}' flushes to the log

# --- skill scaffold control ---
export GIGPO_SKILL_PATH=/mnt/data1/zha00175/verl-agent/agent_system/skill/skills_v0.json
export GIGPO_SKILL_MODE=$MODE
export GIGPO_SKILL_FORCE=1   # deterministic injection for eval (none-mode renders empty anyway)

export WANDB_MODE=offline
export WANDB_DIR=/mnt/data1/zha00175/gigpo_helper_wandb

PY=$VENV/bin/python
cd /mnt/data1/zha00175/verl-agent || exit 1

EXP=gap_${TAG}_${MODE}
$PY -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    algorithm.gamma=0.95 \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=8 \
    env.alfworld.eval_dataset=eval_out_of_distribution \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='gigpo_helper' \
    trainer.experiment_name=$EXP \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.val_only=True $@
