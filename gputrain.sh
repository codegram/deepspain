#!/bin/bash

gpu_count=$(nvidia-smi --query-gpu=gpu_name --format=csv | grep '^name' | wc -l)
python3 -m torch.distributed.launch --nproc_per_node=$gpu_count train.py /storage/boe/lm_data.pkl /artifacts /storage/boe/encM2.pth /storage/boe/itos.pkl