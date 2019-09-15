deps: environment.yaml
	conda env create -f environment.yaml
	conda activate deepspain

lint:
	python -m pylama

/storage/boe:
	mkdir -p /storage/boe

/storage/boe/lm_data.pkl: /storage/boe
	curl "https://s3-eu-west-1.amazonaws.com/datascience.codegram.com/lm_data.pkl" > /storage/boe/lm_data.pkl

/storage/boe/itos.pkl: /storage/boe
	curl "https://s3-eu-west-1.amazonaws.com/datascience.codegram.com/itos.pkl" > /storage/boe/itos.pkl

/storage/boe/encM2.pth: /storage/boe
	curl "https://s3-eu-west-1.amazonaws.com/datascience.codegram.com/encM2.pth" > /storage/boe/encM2.pth

databunch: /storage/boe/lm_data.pkl
pretrained_model: /storage/boe/itos.pkl /storage/boe/encM2.pth

gputrain: deps databunch pretrained_model
	bash scripts/gputrain.sh

train:
	gradient experiments run singlenode \
		--name boe_language_model \
		--projectId przjwc38i \
		--container paperspace/fastai:1.0-CUDA9.2-base-3.0-v1.0.6 \
		--machineType P5000 \
		--command 'make gputrain'

.PHONY: lint deps
