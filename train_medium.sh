#!/bin/bash

gpu_count=$(nvidia-smi --query-gpu=gpu_name --format=csv | grep '[^name]' | wc -l)
dvc run -f train_medium.dvc \
    -d train.py -d data/lm_data.pkl \
	-d pretrained/encoder.pth \
	-d pretrained/itos.pkl \
	-o models/medium_empty_data \
	-o models/encoder_medium_head.pth \
	-o models/model_medium_head.pth \
	-o models/learner_medium_head.pkl \
    -M models/medium_accuracy.metric \
    python3 -m torch.distributed.launch \
        --nproc_per_node=$gpu_count \
        train.py \
        data/lm_data.pkl \
        models/ \
        pretrained/encoder.pth \
        pretrained/itos.pkl\
        --label medium \
        --epochs 2
