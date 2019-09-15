#!/bin/bash

gpu_count=$(nvidia-smi --query-gpu=gpu_name --format=csv | grep '[^name]' | wc -l)
mkdir -p /storage/boe_lm_out
python3 -m torch.distributed.launch --nproc_per_node=$gpu_count scripts/train.py /storage/boe/lm_data.pkl /storage/boe_lm_out /storage/boe/encM2.pth /storage/boe/itos.pkl --head-only
cp -r /storage/boe_lm_out /artifacts