import os

import torch
from omegaconf import OmegaConf
from torch.distributed import init_process_group
from transformers import AutoConfig, AutoTokenizer  # type: ignore

from verl import DataProto
from verl.workers.rollout.vllm_rollout.hqq_vllm_rollout import HQQvLLMRollout

# Example: run with torchrun:
#   torchrun --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 29500 test.py

prompt = """
Question: Mary had 3 boxes of pencils. Each box had 12 pencils. She gave 15 pencils to her friend. How many pencils does she have left?

Answer:"""


def main():
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    init_process_group(backend="nccl", init_method="env://")

    full_config = OmegaConf.load("qapo/verl/trainer/config/ppo_trainer.yaml")
    config = full_config.actor_rollout_ref.rollout
    config.tensor_model_parallel_size = 1

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_cfg = AutoConfig.from_pretrained(model_name)

    rollout = HQQvLLMRollout(
        model_name,
        config,
        tokenizer,
        model_cfg,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.cuda(local_rank) for k, v in inputs.items()}
    inputs["position_ids"] = torch.cumsum(inputs["attention_mask"], dim=-1)

    meta_info = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "recompute_log_prob": False,
        "do_sample": True,
        "validate": False,
    }

    data_proto = DataProto.from_single_dict(data=inputs, meta_info=meta_info)

    outputs = rollout.generate_sequences(data_proto)

    print("TEXT:")
    print(tokenizer.decode(outputs.batch["responses"][0]))
    print("END")


if __name__ == "__main__":
    main()
