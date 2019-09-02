# DeepSpain

A fine-tuned language model on the Spanish BOE (Official State Bulletin), using Fast.ai.

It uses a [pre-trained Spanish TransformerXL language model](https://github.com/mmcctt00/SpanishTransformerXL).

The idea is to facilitate:

- Training classifiers
- Similarity search between items via their word embeddings
- Semantic search

## Setup

Make sure you have Python 3.7 and access to a powerful GPU. Language models need a lot of memory and take a long time to train.

    make deps

## Creating the BOE dataset from scratcho

Pick a starting date from which you wish to fetch the BOE documents, up until yesterday. For example, to start at 2018-11-05:

    python make_dataset.py BOE-S-20181105 output.jsonlines

## Processing the data for language modeling

    python prepare_databunch.py output.jsonlines lm_data

This will output a Pickle file called `lm_data.pkl`.

## Training the model

    mkdir models/
    python train.py lm_data.pkl models/

This will output a pickled Language Learner under `models/boe_lm`, which you can use to encode strings, extract their embeddings, etcetera.
