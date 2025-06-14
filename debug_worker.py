import os

import torch
from omegaconf import OmegaConf
from torch.distributed import init_process_group
from transformers import AutoTokenizer  # type: ignore

from verl import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker

# Example: run with torchrun:
#   torchrun --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 29500 test.py


prompt1 = """
Question: Mary had 3 boxes of pencils. Each box had 12 pencils. She gave 15 pencils to her friend. How many pencils does she have left?

Answer:"""

prompt2 = """
Question: Bob had 3 boxes of pencils. Each box had 2 pencils. She gave 8 pencils to her friend. How many pencils does she have left?

Answer:"""


def get_inputs(prompt, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs["position_ids"] = torch.cumsum(inputs["attention_mask"], dim=-1)

    meta_info = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "recompute_log_prob": False,
        "do_sample": True,
        "validate": False,
    }

    return DataProto.from_single_dict(data=inputs, meta_info=meta_info)


def main():
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    init_process_group(backend="nccl", init_method="env://")

    config = OmegaConf.load("qapo/verl/trainer/config/ppo_trainer.yaml")
    config.actor_rollout_ref.model.path = "Qwen/Qwen2.5-0.5B-Instruct"
    config.actor_rollout_ref.rollout.name = "vllm_hqq"
    print(config.keys())

    worker = ActorRolloutRefWorker(config.actor_rollout_ref, "actor_rollout")
    worker.init_model()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    outputs = worker.generate_sequences(get_inputs(prompt1, tokenizer))

    print("TEXT:")
    print(tokenizer.decode(outputs.batch["responses"][0]).replace("<|endoftext|>", ""))
    print("END")

    print("ROUND 2")

    outputs = worker.generate_sequences(get_inputs(prompt2, tokenizer))

    print("TEXT:")
    print(tokenizer.decode(outputs.batch["responses"][0]).replace("<|endoftext|>", ""))
    print("END")


if __name__ == "__main__":
    main()
