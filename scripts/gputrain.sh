#!/bin/bash

gpu_count=$(nvidia-smi --query-gpu=gpu_name --format=csv | grep '[^name]' | wc -l)
mkdir -p /storage/boe/output
python3 -m torch.distributed.launch \
    --nproc_per_node=$gpu_count \
    scripts/train.py \
    data/sample_databunch.pkl \
    models/ \
    pretrained/encoder.pth \
    pretrained/itos.pkl\
    --head-only
cp -r /storage/boe/output /artifacts