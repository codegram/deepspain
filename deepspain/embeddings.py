"""Use a Language Model to extract embeddings for text documents."""
from fastai.text import LanguageLearner
from deepspain.utils import measure
import torch
from torch import Tensor
from typing import Sequence, cast, Any


def word_embeddings(learner: LanguageLearner, s: str, debug: bool = False) -> Tensor:
    tokens, _ = measure("tokenizing", lambda: learner.data.one_item(s), debug)
    measure("resetting model", lambda: learner.model.reset(), debug)
    encoder = learner.model[0]
    outputs = measure("predicting", lambda: encoder(tokens), debug)
    embeddings = outputs[-1][-1]
    return embeddings


def doc2vec(
    learner: LanguageLearner, s: str, debug: bool = False, max_dim: int = 1024
) -> Sequence[Tensor]:
    with torch.no_grad():
        embeddings = measure(
            "get_full_embeddings", lambda: word_embeddings(learner, s, debug), debug
        )
        avg_pool = embeddings.mean(dim=1)
        max_pool = embeddings.max(dim=1)[0]
        last = cast(Tensor, cast(Any, embeddings)[:, -1])  # workaround pyright issue
        return (
            torch.cat([last, max_pool, avg_pool], 1).to("cpu").squeeze().split(max_dim)
        )
