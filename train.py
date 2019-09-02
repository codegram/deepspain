from pathlib import Path
import click
from fastai.text import language_model_learner, TransformerXL
from fastai.basics import load_data
from fastai.callbacks import SaveModelCallback


@click.command()
@click.argument("lm_databunch", type=click.Path(exists=True, dir_okay=False))
@click.argument("model_path", type=click.Path(exists=True, file_okay=False))
@click.argument("encoder", type=click.Path(exists=True, dir_okay=False))
@click.argument("itos", type=click.Path(exists=True, dir_okay=False))
def main(lm_databunch: str, model_path: str, encoder: str, itos: str):
    click.echo("Loading LM databunch...")
    p = Path(lm_databunch)
    folder = p.parent
    filename = p.name
    data = load_data(folder, filename)

    click.echo("Training language model...")
    learn = language_model_learner(
        data, TransformerXL, pretrained_fnames=[encoder, itos], drop_mult=0.3
    ).to_fp16()
    learn.freeze()
    learn.fit_one_cycle(4, slice(1e-3), callbacks=[SaveModelCallback(learn)])
    learn.load("bestmodel")
    learn.unfreeze()
    learn.fit_one_cycle(4, slice(1e-3), callbacks=[SaveModelCallback(learn)])
    learn.load("bestmodel")
    learn.validate()
    learn.eval()
    learn.export(model_path + "/boe_lm")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
