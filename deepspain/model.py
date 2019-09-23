from pathlib import Path

from fastai.text import (
    LanguageLearner,
    TransformerXL,
    language_model_learner,
    DataBunch,
    TextList,
)


def from_encoder(model_path: Path, encoder_name: str) -> LanguageLearner:
    """Loads a trained language model for inference."""
    print("Loading model for inference....")
    data = DataBunch.load_empty(model_path, "data/empty_data")
    learn = language_model_learner(data, TransformerXL, pretrained=False)
    learn.load_encoder(encoder_name)
    learn.freeze()
    learn.model.eval()
    return learn


def from_model(model_path: Path, model_name: str) -> LanguageLearner:
    """Loads a trained language model for inference."""
    print("Loading model for inference....")
    data = DataBunch.load_empty(model_path, "data/empty_data")
    learn = language_model_learner(data, TransformerXL, pretrained=False)
    learn.load(model_name)
    learn.freeze()
    learn.model.eval()
    return learn
