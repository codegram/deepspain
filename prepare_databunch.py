import click
import jsonlines
from fastai.text import pd, np, TextList


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(exists=False, dir_okay=False))
def main(dataset_path: str, output: str):
    click.echo("Turning raw data into a Databunch suitable for language model")
    rows = []
    with jsonlines.open(dataset_path) as reader:
        for obj in reader.iter(type=dict, skip_invalid=True):
            rows.append(obj)
    df = pd.DataFrame(rows)
    click.echo("Created dataframe with shape " + str(df.shape))
    cols = ["title", "content"]
    bs = 48
    np.random.seed(42)

    data = (
        TextList.from_df(df, cols=cols)
        .split_by_rand_pct(0.2, seed=42)
        .label_for_lm()
        .databunch(bs=bs)
    )

    data.save(output + ".pkl")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
