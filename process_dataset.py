import click
import jsonlines
import pandas as pd

from deepspain.dataset import df_to_lm_databunch


@click.command()
@click.argument(
    "jsonlines_path",
    metavar="<output.jsonlines>",
    type=click.Path(exists=True, dir_okay=False),
)
@click.argument(
    "output_file",
    metavar="<databunch.pkl>",
    type=click.Path(exists=False, dir_okay=False),
)
@click.option(
    "--sample", is_flag=True, default=False, help="Use only 1% of the data, to test."
)
def main(dataset_path: str, output_file: str, sample: bool):
    """Turn the data in <output.jsonlines> into a DataBunch suitable for Language Modeling,
    saving it to <databunch.pkl>."""

    click.echo("Turning raw data into a Databunch suitable for language modeling")
    rows = []
    with jsonlines.open(dataset_path) as reader:
        for obj in reader.iter(type=dict, skip_invalid=True):
            rows.append(obj)
    df = pd.DataFrame(rows)
    click.echo("Created dataframe with shape " + str(df.shape))
    databunch = df_to_lm_databunch(df, columns=["title", "content"], sample=sample)
    databunch.save(output_file)


if __name__ == "__main__":
    main()
