import warnings
from pathlib import Path
import click
import jsonlines
import pandas as pd

from fastai.text import TextList

from deepspain.utils import measure
from deepspain.model import from_model


@click.command()
@click.argument(
    "models_path", type=click.Path(exists=True, file_okay=False), metavar="<models>"
)
@click.argument(
    "test_data_json",
    type=click.Path(exists=True, dir_okay=False),
    metavar="<test_data.jsonlines>",
)
@click.option("--debug", is_flag=True, default=False)
def main(models_path: Path, test_data_json: Path, debug: bool):
    """Evaluates a language model against a test data set."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        print(f"Loading test data from {test_data_json}...")
        rows = []
        with jsonlines.open(test_data_json) as reader:
            for obj in reader.iter(type=dict, skip_invalid=True):
                rows.append(obj)
        df = pd.DataFrame(rows)
        test_databunch = (
            TextList.from_df(df, path=models_path, cols=["title", "content"])
            .split_none()
            .label_for_lm()
            .databunch(bs=4)
        )

        learner = measure(
            "model loading",
            lambda: from_model(models_path, model_name="model_large_finetuned"),
            debug,
        )

        print(learner.validate(dl=test_databunch.train_dl))


if __name__ == "__main__":
    main()
