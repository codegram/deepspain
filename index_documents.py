import warnings
from pathlib import Path
import click

from elasticsearch import Elasticsearch

from deepspain.utils import measure
from deepspain.model import from_encoder
from deepspain.dataset import load_databunch
from deepspain.search import recreate_index, index_document


@click.command()
@click.argument(
    "models_path", type=click.Path(exists=True, file_okay=False), metavar="<models>"
)
@click.argument(
    "data_path", type=click.Path(exists=True, dir_okay=False), metavar="<databunch.pkl>"
)
@click.option(
    "--drop-index",
    is_flag=True,
    default=False,
    help="Whether to drop and recreate the index before indexing",
)
@click.option(
    "--index_name",
    type=str,
    default="boe",
    help='ElasticSearch index name (default "boe")',
)
@click.option(
    "--host",
    type=str,
    default="localhost",
    help='ElasticSearch hostname (default "localhost")',
)
@click.option(
    "--port", type=int, default=9200, help="ElasticSearch port (default 9200)"
)
@click.option(
    "--limit-bytes",
    type=int,
    default=5000,
    help="The bytes to keep from the text before indexing",
)
@click.option("--debug", is_flag=True, default=False)
def main(
    models_path: Path,
    data_path: Path,
    drop_index: bool,
    index_name: str,
    host: str,
    port: int,
    limit_bytes: int,
    debug: bool,
):
    """Index all the training rows in <databunch.pkl> into ElasticSearch."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        learner = measure(
            "encoder loading",
            lambda: from_encoder(models_path, encoder_name="encoder_large_finetuned"),
            debug,
        )

        es = Elasticsearch(hosts=[{"host": host, "port": port}])
        if drop_index:
            print("Recreating index...")
            recreate_index(es, learner, index_name, debug)
        print("Loading data...")
        df = load_databunch(Path(data_path), debug).train_ds.inner_df
        total = df.shape[0]
        print(f"Indexing {total} rows...")
        for idx, row in df.iterrows():
            measure(
                f"{idx}/{total}",
                lambda: index_document(
                    es, learner, index_name, row.to_dict(), limit_bytes, debug
                ),
                debug,
            )


if __name__ == "__main__":
    main()
