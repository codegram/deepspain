#!/bin/bash

gpu_count=$(nvidia-smi --query-gpu=gpu_name --format=csv | grep '[^name]' | wc -l)
dvc run -f train_small.dvc \
    -d train.py -d data/sample_databunch.pkl \
	-d pretrained/encoder.pth \
	-d pretrained/itos.pkl \
	-o models/small_empty_data \
	-o models/encoder_small_head.pth \
	-o models/model_small_head.pth \
	-o models/learner_small_head.pkl \
    -M models/small_accuracy.metric \
    python3 -m torch.distributed.launch \
        --nproc_per_node=$gpu_count \
        train.py \
        data/sample_databunch.pkl \
        models/ \
        pretrained/encoder.pth \
        pretrained/itos.pkl\
        --label small \
        --head-epochs 2 \
	    --gpus $gpu_count \
        --head-only
