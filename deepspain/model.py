from pathlib import Path

from fastai.text import (
    LanguageLearner,
    TransformerXL,
    language_model_learner,
    DataBunch,
)


def load_for_inference(model_path: Path) -> LanguageLearner:
    """Loads a trained language model for inference."""
    print("Loading model for inference....")
    data = DataBunch.load_empty(model_path, "empty_data")
    learn = language_model_learner(data, TransformerXL, pretrained=False)
    learn.model_dir = model_path
    learn.load("model")
    learn.load_encoder("encoder")
    learn.freeze()
    learn.model.eval()
    return learn
