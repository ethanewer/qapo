import glob
import os

import hydra  # type: ignore
import ray  # type: ignore
from omegaconf import OmegaConf

from verl.trainer.main_ppo import create_rl_dataset
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.tracking import Tracking


@hydra.main(config_path="../verl/trainer/config", config_name="ppo_trainer")
def main(config):
    OmegaConf.resolve(config)

    checkpoint_dir = config.checkpoint_dir

    if not ray.is_initialized():
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()  # type: ignore
    ray.get(runner.run.remote(config, checkpoint_dir))

    ray.shutdown()


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config, checkpoint_dir):
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local

        print(config.rollout.hqq_config)
        print(f"{config.rollout.hqq_config.weight_bits=}")

        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))
        tokenizer = hf_tokenizer(local_path, trust_remote_code=config.data.get("trust_remote_code", False))
        processor = hf_processor(local_path, use_fast=True)

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))

        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,  # type: ignore
            val_reward_fn=val_reward_fn,
            val_dataset=val_dataset,
            device_name=config.trainer.device,
        )
        trainer.init_workers()

        logger = Tracking(
            project_name=config.trainer.project_name,
            experiment_name=config.trainer.experiment_name,
            default_backend=config.trainer.logger,
            config=OmegaConf.to_container(config, resolve=True),
        )

        checkpoint_paths = sorted(
            glob.glob(os.path.join(checkpoint_dir, "global_step_*")),
            key=lambda path: int(path.split("_")[-1]),
        )

        print(f"Found {len(checkpoint_paths)} checkpoints in {checkpoint_dir}")

        all_metrics = {}

        for checkpoint_path in checkpoint_paths:
            print(f"--- Evaluating checkpoint: {checkpoint_path} ---")

            actor_checkpoint_path = os.path.join(checkpoint_path, "actor")

            trainer.actor_rollout_wg.load_checkpoint(actor_checkpoint_path)

            print("Starting validation")
            val_metrics = trainer._validate()

            global_step = int(os.path.basename(checkpoint_path).split("_")[-1])
            all_metrics[global_step] = val_metrics

            logger.log(data=val_metrics, step=global_step)

            print(f"Metrics for global_step {global_step}:")
            for k, v in val_metrics.items():
                print(f"  {k}: {v}")
            print("---" * 10)

        print("\n\n--- All Checkpoint Metrics ---")
        for step, metrics in all_metrics.items():
            print(f"Global Step: {step}")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
        print("---" * 10)


if __name__ == "__main__":
    main()  # type: ignore
