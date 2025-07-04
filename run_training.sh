set -x

# HQQ configuration
use_hqq_rollout=True
hqq_weight_bits=4
use_hqq_qat=True
update_hqq_qat_metadata=rollout

model="Qwen/Qwen2.5-1.5B-Instruct"

# Validate HQQ configuration
if [ "$use_hqq_rollout" != "True" ] && [ "$use_hqq_rollout" != "False" ]; then
    echo "Error: use_hqq_rollout must be either True or False" && exit 1
fi
if [ "$use_hqq_qat" != "True" ] && [ "$use_hqq_qat" != "False" ]; then
    echo "Error: use_hqq_qat must be either True or False" && exit 1
fi
if [ "$use_hqq_rollout" = "True" ]; then
    rollout_name="vllm_hqq"
else
    rollout_name="vllm"
fi

# Set experiment name
experiment_name=$(echo "${model#*/}" | sed 's/[.-]/_/g' | tr '[:upper:]' '[:lower:]')
if [ "$use_hqq_qat" = "True" ]; then
    if [ "$optimize_hqq_qat" = "True" ]; then
        experiment_name="${experiment_name}_optimized"
    fi
    experiment_name="${experiment_name}_qat"
fi
if [ "$use_hqq_rollout" = "True" ]; then
    experiment_name="${experiment_name}_qapo"
fi
if [ "$use_hqq_rollout" = "True" ] || [ "$use_hqq_qat" = "True" ]; then
    experiment_name="${experiment_name}_${hqq_weight_bits}bit"
else
    experiment_name="${experiment_name}_grpo"
fi

# Data paths
gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.hqq_config.weight_bits=$hqq_weight_bits \
    actor_rollout_ref.actor.fsdp_config.use_hqq_qat=$use_hqq_qat \
    actor_rollout_ref.actor.fsdp_config.hqq_qat_config.nbits=$hqq_weight_bits \
    actor_rollout_ref.actor.fsdp_config.hqq_qat_config.update_metadata=$update_hqq_qat_metadata \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='qapo_gsm8k_math' \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=8 $@