import warnings
from pathlib import Path

from functools import partial

import click
import torch
from fastai.text import (
    TransformerXL,
    language_model_learner,
    LanguageLearner,
    TextLMDataBunch,
)
from fastai.distributed import setup_distrib

from deepspain.dataset import load_databunch
from deepspain.tensorboard import LearnerTensorboardWriter

def save(
    data: TextLMDataBunch,
    learn: LanguageLearner,
    label: str,
    suffix: str,
    accuracy: int,
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


def clean(word: str):
    w = (
        word.replace("\n", "(nl)")
        .replace("\t", "(tab)")
        .replace("\u2002", "(u2002)")
        .replace(" ", "(sp)")
    )
    return "".join([c if ord(c) < 1000 else "_" for c in w])


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
@click.option("--gpus", type=int, help="Total number of GPUs")
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
    gpus: int,
    local_rank: int,
):
    """Trains a Language Model, starting from pretrained weights, with data from <databunch.pkl>.
    Saves the best model and encoder to <output-dir>/{model,encoder}.pth respectively.
    """
    output_path = Path(output_dir)

    if gpus > 1:
        click.echo("Setting up distributed training...")
        setup_distrib(local_rank)

    click.echo("Loading LM databunch...")
    data = load_databunch(Path(databunch))

    # Do a bit a hack to save time
    pretrained_special = ["_unk_", "_pad_", "xbos", "xfld"]
    actual_special = ["xxunk", "xxpad", "xxbos", "xxfld"]

    print(len(list(filter(lambda x: "?" in x, data.vocab.itos))))
    itos = [
        pretrained_special[actual_special.index(word)]
        if (word in actual_special)
        else clean(word)
        for word in data.vocab.itos
    ]
    data.vocab.itos = itos

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
    )

    tboard_path = Path("logs/" + label)
    node_name = "gpu-" + str(local_rank) + "-head" 
    learn.callback_fns.append(
        partial(
            LearnerTensorboardWriter, base_dir=tboard_path, gpus=gpus, name=node_name
        )
    )

    if gpus > 1:
        learn.to_distributed(local_rank)

    learn.freeze()
    click.echo("Training model head...")
    learn.fit_one_cycle(head_epochs, 1e-3, moms=(0.8, 0.7))
    click.echo("Validating...")
    accuracy = learn.validate()[1].item()

    if local_rank == 0:
        save(data, learn, label, "head", accuracy)

    if not head_only:
        click.echo("Unfreezing and fine-tuning earlier layers...")

        learn = language_model_learner(
            data,
            TransformerXL,
            pretrained_fnames=[
                "./../" + pretrained_encoder.replace(".pth", ""),
                "./../" + pretrained_itos.replace(".pkl", ""),
            ],
            drop_mult=0.1,
        )

        tboard_path = Path("logs/" + label)
        node_name = "gpu-" + str(local_rank) + "-finetuned"
        learn.callback_fns.append(
            partial(
                LearnerTensorboardWriter, base_dir=tboard_path, gpus=gpus, name=node_name
            )
        )
        if gpus > 1:
            learn.to_distributed(local_rank)

        learn.load("model_" + label + "_head")

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
