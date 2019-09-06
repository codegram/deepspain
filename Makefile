deps: requirements.txt
	pip install --upgrade pip && pip install -r requirements.txt -U

lint:
	flake8

/storage/boe/lm_data.pkl:
	curl "https://s3-eu-west-1.amazonaws.com/datascience.codegram.com/lm_data.pkl" > /storage/boe/lm_data.pkl

/storage/boe/itos.pkl:
	curl "https://s3-eu-west-1.amazonaws.com/datascience.codegram.com/itos.pkl" > /storage/boe/itos.pkl

/storage/boe/encM2.pth:
	curl "https://s3-eu-west-1.amazonaws.com/datascience.codegram.com/encM2.pth" > /storage/boe/encM2.pth

databunch: /storage/boe/lm_data.pkl
pretrained_model: /storage/boe/itos.pkl /storage/boe/encM2.pth

gputrain: deps databunch pretrained_model
	bash gputrain.sh

train:
	gradient experiments run singlenode \
		--name boe_language_model \
		--projectId pr064oj2f \
		--container paperspace/fastai:1.0-CUDA9.2-base-3.0-v1.0.6 \
		--machineType GV100x8 \
		--command 'make gputrain'

.PHONY: lint deps
