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
from fastai.text import TransformerXL, language_model_learner

from deepspain.dataset import load_databunch


class PaperspaceLoggingCallback(LearnerCallback):
    def __init__(self, learn: Learner, node: int):
        super().__init__(learn)
        self.node = node

    def on_train_begin(self, **kwargs: Any) -> None:
        for name in self.learn.recorder.names:
            print(json.dumps(dict(chart=name, axis=name)))

    def on_epoch_end(
        self,
        epoch: int,
        smooth_loss: torch.Tensor,
        last_metrics: MetricsList,
        **kwargs: Any,
    ) -> bool:
        "Add a line with `epoch` number, `smooth_loss` and `last_metrics`."
        last_metrics = ifnone(last_metrics, [])
        for name, stat in zip(
            self.learn.recorder.names, [epoch, smooth_loss] + last_metrics
        ):
            value = (
                str(stat)
                if isinstance(stat, int)
                else "#na#"
                if stat is None
                else f"{stat:.6f}"
            )
            print(json.dumps(dict(chart=str(self.node) + "_" + name, x=epoch, y=value)))
        return True


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
    "--head-only",
    is_flag=True,
    default=False,
    help="Train only the model's head (without fine-tuning the backbone)",
)
@click.option(
    "--local_rank", type=int, help="Node number (set by pyTorch's distributed trainer)"
)
def main(
    databunch: str,
    output_dir: str,
    pretrained_encoder: str,
    pretrained_itos: str,
    head_only: bool,
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

    click.echo("Training language model...")
    learn = language_model_learner(
        data,
        TransformerXL,
        pretrained_fnames=[
            pretrained_encoder.replace(".pth", ""),
            pretrained_itos.replace(".pkl", ""),
        ],
        drop_mult=0.3,
    )
    learn = learn.to_distributed(local_rank)
    learn.freeze()
    click.echo("Training model head...")
    learn.fit_one_cycle(
        4,
        1e-3,
        callbacks=[
            PaperspaceLoggingCallback(learn, node=local_rank),
            SaveModelCallback(learn),
        ],
    )
    learn.load("bestmodel")

    if local_rank == 0:
        click.echo("Saving...")
        learn.save(output_path / "model")
        learn.save_encoder(output_path / "encoder")
        click.echo("Exporting...")
        data.export(output_path / "empty_data")
        learn.export(output_path / "export.pkl")

    if not head_only:
        click.echo("Unfreezing and fine-tuning earlier layers...")

        learn = learn.to_distributed(local_rank)
        learn.unfreeze()
        learn.fit_one_cycle(
            4,
            1e-3,
            callbacks=[
                PaperspaceLoggingCallback(learn, node=local_rank),
                SaveModelCallback(learn),
            ],
        )
        learn.load("bestmodel")
        click.echo("Saving...")
        learn.save(output_path / "model_ft")
        learn.save_encoder(output_path / "/encoder_ft")
        click.echo("Exporting...")
        learn.export(output_path / "/export_ft.pkl")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        main()  # pylint: disable=no-value-for-parameter
