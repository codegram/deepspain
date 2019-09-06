from pathlib import Path
import click
from fastai.text import language_model_learner, TransformerXL
from fastai.basics import load_data
from fastai.callbacks import SaveModelCallback
from fastai.distributed import *
import torch
import warnings


@click.command()
@click.argument("lm_databunch", type=click.Path(exists=True, dir_okay=False))
@click.argument("model_path", type=click.Path(exists=True, file_okay=False))
@click.argument("encoder", type=click.Path(exists=True, dir_okay=False))
@click.argument("itos", type=click.Path(exists=True, dir_okay=False))
@click.option("--local_rank", type=int)
def main(lm_databunch: str, model_path: str, encoder: str, itos: str,
         local_rank: int):
    print(local_rank)
    click.echo("Setting up distributed training...")
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    click.echo("Loading LM databunch...")
    p = Path(lm_databunch)
    folder = p.parent
    filename = p.name
    data = load_data(folder, filename)

    click.echo("Training language model...")
    learn = language_model_learner(
        data, TransformerXL, pretrained_fnames=[
            encoder.replace('.pth', ''),
            itos.replace('.pkl', '')], drop_mult=0.3
    ).to_fp16()
    learn = learn.to_distributed(local_rank)
    learn.freeze()
    click.echo("Training model head...")
    learn.fit_one_cycle(4, slice(1e-3), callbacks=[SaveModelCallback(learn)])
    learn.load("bestmodel")
    learn.unfreeze()
    click.echo("Unfreezing and fine-tuning earlier layers...")
    learn.fit_one_cycle(4, slice(1e-3), callbacks=[SaveModelCallback(learn)])
    learn.load("bestmodel")
    click.echo("Validating...")
    click.echo(learn.validate())
    learn.eval()
    click.echo("Exporting...")
    learn.export(model_path + "/boe_lm")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        main()  # pylint: disable=no-value-for-parameter
