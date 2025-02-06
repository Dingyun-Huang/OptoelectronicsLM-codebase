#!/bin/bash -l

source /path/to/venv/bin/activate

echo Using Python `which python`

HF_HOME=/path/to/HF_HOME
export HF_HOME=$HF_HOME
# NCCL_COLLNET_ENABLE=1
# NCCL_NET_GDR_LEVEL=PHB

# export MPICH_GPU_SUPPORT_ENABLED=1

accelerate launch --config_file configs/accelerate_config.yaml finetune_sroberta_mean.py
