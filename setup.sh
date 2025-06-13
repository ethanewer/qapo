conda create -n verl python==3.10 -y
conda activate verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .