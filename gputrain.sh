#!/bin/bash

gpu_count=$(nvidia-smi --query-gpu=gpu_name --format=csv | grep '[^name]' | wc -l)
dvc run -f small_training.dvc \
    -d train.py -d data/sample_databunch.pkl \
	-d pretrained/encoder.pth \
	-d pretrained/itos.pkl \
	-o models/empty_data \
	-o models/encoder_head.pth \
	-o models/model_head.pth \
	-o models/learner_head.pkl \
    -M models/accuracy.metric \
    python3 -m torch.distributed.launch \
        --nproc_per_node=$gpu_count \
        train.py \
        data/sample_databunch.pkl \
        models/ \
        pretrained/encoder.pth \
        pretrained/itos.pkl\
        --head-only
