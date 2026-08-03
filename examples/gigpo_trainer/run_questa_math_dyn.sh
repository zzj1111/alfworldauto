set -x
# QuestA-style math RL on Nemotron-1.5B, with OUR dynamic scaffold (DynamicHintDataset).
# Entry = recipe.dapo.main_dapo (standalone DAPO math trainer, NO agent env). The hint =
# first f% of the reference solution, f read live from MATH_SCAFFOLD_PATH (hot-reload).
export HF_HOME=/mnt/data1/zha00175/hf_home
export PATH=/mnt/data1/zha00175/miniconda/envs/verl/bin:$PATH
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TOKENIZERS_PARALLELISM=true
export MATH_SCAFFOLD_PATH=${MATH_SCAFFOLD_PATH:-/mnt/data1/zha00175/data/questa_math/math_scaffold.json}
export MATH_HINT_FRACTION=${MATH_HINT_FRACTION:-0.5}
export MATH_HINT_MAX_CHARS=${MATH_HINT_MAX_CHARS:-6000}

MODEL=/mnt/data1/zha00175/models/OpenMath-Nemotron-1.5B
DATA=/mnt/data1/zha00175/data/questa_math
HINT_DS=/mnt/data1/zha00175/verl-agent/agent_system/skill_opt/math_hint_dataset.py

N_GPUS=${AUTO_GPUS_N:-2}
group_size=${GROUP:-8}
train_bsz=${BSZ:-64}
mini_bsz=${MINI:-64}
max_prompt=${MAXP:-4096}
max_resp=${MAXR:-8192}
steps=${STEPS:-400}

cd /mnt/data1/zha00175/verl-agent || exit 1
python3 -m recipe.dapo.main_dapo \
    data.train_files=$DATA/train_main.parquet \
    data.val_files=$DATA/val_heldout.parquet \
    data.train_batch_size=$train_bsz \
    data.max_prompt_length=$max_prompt \
    data.max_response_length=$max_resp \
    data.truncation='right' \
    data.reward_fn_key=data_source \
    data.custom_cls.path=$HINT_DS \
    data.custom_cls.name=DynamicHintDataset \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=2e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.05 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$mini_bsz \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.loss_agg_mode='token-mean' \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=$group_size \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt + max_resp)) \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=acc \
    algorithm.filter_groups.max_num_gen_batches=10 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name='questa_math' \
    trainer.experiment_name='nemotron_dyn_scaffold' \
    trainer.default_local_dir=/mnt/data1/zha00175/ckpts_questa_math \
    trainer.resume_mode=auto \
    actor_rollout_ref.actor.checkpoint.contents=['model','optimizer','extra','hf_model'] \
    +ray_init._temp_dir=/dev/shm/zray_math \
    +ray_init.include_dashboard=False \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.total_training_steps=$steps \
    trainer.val_before_train=True "$@"
