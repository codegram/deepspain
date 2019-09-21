import json
import warnings
from pathlib import Path
from typing import Any

import click
import torch
from fastai.basic_train import Learner
from fastai.callback import MetricsList
from fastai.callbacks import LearnerCallback, SaveModelCallback
from fastai.distributed import ifnone
from fastai.text import (
    TransformerXL,
    language_model_learner,
    LanguageLearner,
    TextLMDataBunch,
)

from deepspain.dataset import load_databunch


def save(
    data: TextLMDataBunch, learn: LanguageLearner, label: str, suffix: str, accuracy: int
):
    f = open("models/" + label + "_accuracy.metric", "w")
    f.write(str(accuracy))
    f.close()
    click.echo("Saving...")
    learn.save("model_" + label + "_" + suffix)
    learn.save_encoder("encoder_" + label + "_" + suffix)
    click.echo("Exporting...")
    data.export("models/" + label + "_empty_data")
    learn.export("models/learner_" + label + "_" + suffix + ".pkl")


@click.command()
@click.argument(
    "databunch", metavar="<databunch.pkl>", type=click.Path(exists=True, dir_okay=False)
)
@click.argument(
    "output_dir", metavar="<output-dir>", type=click.Path(exists=True, file_okay=False)
)
@click.argument(
    "pretrained_encoder",
    metavar="<pretrained_encoder.pth>",
    type=click.Path(exists=True, dir_okay=False),
)
@click.argument(
    "pretrained_itos",
    metavar="<pretrained_itos.pkl>",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--label",
    default="standard",
    type=str,
    help="Label to distinguish the trained model",
)
@click.option(
    "--head-only",
    is_flag=True,
    default=False,
    help="Train only the model's head (without fine-tuning the backbone)",
)
@click.option(
    "--head-epochs", type=int, default=4, help="Number of epochs to train the head"
)
@click.option(
    "--backbone-epochs",
    type=int,
    default=8,
    help="Number of epochs to train the backbone",
)
@click.option(
    "--local_rank", type=int, help="Node number (set by pyTorch's distributed trainer)"
)
def main(
    databunch: str,
    output_dir: str,
    pretrained_encoder: str,
    pretrained_itos: str,
    label: str,
    head_only: bool,
    head_epochs: int,
    backbone_epochs: int,
    local_rank: int,
):
    """Trains a Language Model, starting from pretrained weights, with data from <databunch.pkl>.
    Saves the best model and encoder to <output-dir>/{model,encoder}.pth respectively.
    """
    output_path = Path(output_dir)

    click.echo("Setting up distributed training...")
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    click.echo("Loading LM databunch...")
    data = load_databunch(Path(databunch))

    data.path = Path(".")
    click.echo("Training language model...")
    learn = language_model_learner(
        data,
        TransformerXL,
        pretrained_fnames=[
            "./../" + pretrained_encoder.replace(".pth", ""),
            "./../" + pretrained_itos.replace(".pkl", ""),
        ],
        drop_mult=0.1,
    ).to_distributed(local_rank)
    learn.freeze()
    click.echo("Training model head...")
    learn.fit_one_cycle(
        head_epochs,
        1e-3,
        moms=(0.8, 0.7),
        callbacks=[
          # SaveModelCallback(learn, name="bestmodel_" + label + "_head")
        ]
    )
    # learn.load("bestmodel_" + label + "_head")
    click.echo("Validating...")
    accuracy = learn.validate()[1].item()

    if local_rank == 0 and head_only:
        save(data, learn, label, "head", accuracy)

    if not head_only:
        click.echo("Unfreezing and fine-tuning earlier layers...")

        learn.unfreeze()
        learn.fit_one_cycle(backbone_epochs, 1e-3, moms=(0.8, 0.7))
	
    	click.echo("Validating...")
    	accuracy = learn.validate()[1].item()
        if local_rank == 0:
            save(data, learn, label, "finetuned", accuracy)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        main()  # pylint: disable=no-value-for-parameter

