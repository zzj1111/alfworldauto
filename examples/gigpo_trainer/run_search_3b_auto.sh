set -x
# Auto-scaffold experiment: SAME base 3B as the baseline (42.9), so the final standalone
# is a clean apples-to-apples vs baseline and vs the hand-tuned iter2 (45.9). The scaffold
# JSON is owned by the GPT-5.5 controller (buckets form); NO hand-coded withdrawal here.
export CUDA_VISIBLE_DEVICES=${AUTO_GPUS:-1,6}
export GIGPO_SKILL_PATH=/mnt/data1/zha00175/searchR1_data/search_skills_auto.json
export GIGPO_SKILL_MODE=full
export GIGPO_SKILL_DEBUG=1
export WANDB_ENTITY=mhong-university-of-minnesota
export WANDB_RUN_ID=auto_gigpo3b
export WANDB_RESUME=allow
export TMPDIR=/dev/shm/ztmp_auto
export RAY_TMPDIR=/dev/shm/zray_auto
export PATH=/mnt/data1/zha00175/miniconda/envs/verl/bin:$PATH
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

ENGINE=${1:-vllm}

train_data_size=256
val_data_size=512
group_size=5

# GiGPO config (identical to baseline/iter2 -- only the scaffold harness differs)
mode="mean_std_norm"
enable_similarity=True
similarity_thresh=0.9

TRAIN_DATA="$HOME/data/searchR1_processed_direct/train.parquet"

# Revert hook: on a controller revert, the supervisor bumps to a NEW generation dir and
# sets AUTO_CKPT_DIR=<gen dir> + AUTO_RESUME_MODEL=<earlier ckpt hf> + AUTO_RESUME_MODE=disable.
# Separate gen dirs mean revert never overwrites the pre-revert checkpoints. Default =
# clean start from base 3B into gen0 with normal auto-resume.
RESUME_MODEL=${AUTO_RESUME_MODEL:-/mnt/data1/zha00175/skillzero-env/models/Qwen2.5-3B-Instruct}
RESUME_MODE=${AUTO_RESUME_MODE:-auto}
CKPT_DIR=${AUTO_CKPT_DIR:-/mnt/data1/zha00175/ckpts_search_3b_auto/gen0}

cd /mnt/data1/zha00175/verl-agent || exit 1
/mnt/data1/zha00175/miniconda/envs/verl/bin/python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=/mnt/data1/zha00175/searchR1_data/test_subset.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$RESUME_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    algorithm.gigpo.enable_similarity=$enable_similarity \
    algorithm.gigpo.similarity_thresh=$similarity_thresh \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=$group_size \
    env.history_length=4 \
    env.search.search_url='http://127.0.0.1:8010/retrieve' \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_search' \
    trainer.experiment_name='gigpo_3b_auto' \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.resume_mode=$RESUME_MODE \
    actor_rollout_ref.actor.checkpoint.contents=['model','optimizer','extra','hf_model'] \
    +ray_init._temp_dir=/dev/shm/zray_auto \
    +ray_init.include_dashboard=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${AUTO_TOTAL_STEPS:-400} \
    trainer.val_before_train=False $@
