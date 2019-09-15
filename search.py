import warnings
from pathlib import Path

from elasticsearch import Elasticsearch
from fire import Fire

from deepspain.utils import measure
from deepspain.model import load_for_inference
from deepspain.search import search


def main(
    models_path: Path,
    query: str,
    index_name: str = "boe",
    host: str = "localhost",
    port: int = 9200,
    debug: bool = False,
):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        learner = measure(
            "model loading", lambda: load_for_inference(models_path), debug
        )
        es = Elasticsearch(hosts=[{"host": host, "port": port}])
        for result in search(es, learner, index_name, query, debug):
            print("\n")
            print(result)


if __name__ == "__main__":
    Fire(main)
