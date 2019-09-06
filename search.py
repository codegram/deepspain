import hnswlib
from pathlib import Path
from flask import Flask
import pandas
from typing import List
import numpy as np
import pandas as pd
from fastai.text import LanguageLearner, TransformerXL, language_model_learner
from fastai.basics import load_data
import os
import pickle


def load_pretrained(lm_databunch: str, models_path: Path) -> LanguageLearner:
    p = Path(lm_databunch)
    folder = p.parent
    filename = p.name
    data = load_data(folder, filename)

    learn = language_model_learner(
        data, TransformerXL, pretrained=False).to_fp16()
    learn.load(models_path / "bestmodel")
    learn.model.eval()
    learn.to_fp32().freeze()
    return learn


def get_full_embeddings(learner: LanguageLearner, s: str):
    xs, _ = learner.data.one_item(s)
    learner.model.reset()
    outputs = learner.model(xs)
    return outputs[-1][-1]


def get_summarized_embeddings(learner: LanguageLearner, s: str):
    embeddings = get_full_embeddings(learner, s)
    mean = embeddings.mean(1).data.cpu().numpy()
    maxm = embeddings.max(1)[0].data.cpu().numpy()
    last = embeddings[-1][-1].unsqueeze(0).data.cpu().numpy()
    return np.concatenate([mean, maxm, last], axis=1).reshape(-1)


def create_search_index():
    s = hnswlib.Index(space='l2', dim=1200)
    s.init_index(max_elements=40000, ef_construction=200, M=16)
    s.set_ef(50)
    return s


def save_search_index(index: hnswlib.Index):
    index.save_index("search_index.pkl")


def load_search_index():
    if Path("search_index.pkl").exists():
        s = hnswlib.Index(space='l2', dim=1200)
        s.load_index("search_index.pkl")
        return s
    else:
        return None


def index_dataframe(learner: LanguageLearner, index: hnswlib.Index,
                    df: pd.DataFrame):
    for idx, row in df.iterrows():
        s = row['title'] + row['content']
        print(idx)
        index.add_items([get_summarized_embeddings(learner, s)], [idx])


def search(learner: LanguageLearner,
           df: pd.DataFrame,
           index: hnswlib.Index, query: str, n=3) -> (List[int], List[float]):
    labels, distances = index.knn_query(
        np.array([get_summarized_embeddings(learner, query)]), k=n)
    return [{'row': df[label], 'distance': distance}
            for (label, distance) in labels.zip(distances)]


# app=Flask(__name__)
# app.config['DEBUG']=True
index = load_search_index() or create_search_index()
print(index)
print("Loading model....")
learn = load_pretrained("hey.pkl", Path("."))
# print(learn.validate())
print("Loading dataset...")
p = Path("hey.pkl")
folder = p.parent
filename = p.name
data = load_data(folder, filename)
df = data.train_ds.inner_df
print(df)
print(df.head())
index_dataframe(learn, index, df)
save_search_index(index)
