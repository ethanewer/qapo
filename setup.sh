USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
pip install tensorboard

PYTHONPATH=qapo python examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k
PYTHONPATH=qapo python examples/data_preprocess/math_dataset.py --local_dir ~/data/math