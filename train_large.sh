#!/bin/bash

gpu_count=$(nvidia-smi --query-gpu=gpu_name --format=csv | grep '[^name]' | wc -l)
dvc run -f train_large.dvc \
    -d train.py -d data/lm_data.pkl \
	-d pretrained/encoder.pth \
	-d pretrained/itos.pkl \
	-o models/large_empty_data \
	-o models/encoder_large_head.pth \
	-o models/model_large_head.pth \
	-o models/learner_large_head.pkl \
	-o logs/large \
    -M models/large_accuracy.metric \
    python3 -m torch.distributed.launch \
        --nproc_per_node=$gpu_count \
        train.py \
        data/lm_data.pkl \
        models/ \
        pretrained/encoder.pth \
        pretrained/itos.pkl\
        --label large \
        --head-epochs 4 \
        --backbone-epochs 10 \
        --gpus $gpu_count
