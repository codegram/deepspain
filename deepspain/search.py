from elasticsearch import Elasticsearch
from fastai.text import LanguageLearner

from deepspain.embeddings import doc2vec
from deepspain.utils import measure


def recreate_index(
    es: Elasticsearch, learn: LanguageLearner, index_name: str, debug=False
):
    if es.indices.exists(index_name):
        print("deleting '%s' index..." % (index_name))
        res = es.indices.delete(index=index_name)
        print(" response: '%s'" % (res))
    # since we are running locally, use one shard and no replicas
    shapes = list(map(lambda x: x.shape[0], doc2vec(learn, "test", debug)))
    mappings = {
        "embeddings_" + str(idx): {"type": "dense_vector", "dims": dims}
        for idx, dims in zip(range(len(shapes)), shapes)
    }
    request_body = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {"properties": mappings},
    }
    print("creating '%s' index..." % (index_name))
    res = es.indices.create(index=index_name, body=request_body)
    print(" response: '%s'" % (res))


def index_document(
    es: Elasticsearch,
    learner: LanguageLearner,
    index_name: str,
    document: dict,
    limit_bytes: int,
    debug=False,
):
    content = (document["title"] + "\n" + document["content"])[:limit_bytes]
    embeddings = doc2vec(learner, content, debug)
    for idx, e in zip(range(len(embeddings)), embeddings):
        document["embeddings_" + str(idx)] = e.tolist()
    res = es.index(index=index_name, id=document["id"], body=document)
    return res


def search(
    es: Elasticsearch,
    learner: LanguageLearner,
    index_name: str,
    query: str,
    debug=False,
):
    embeddings = doc2vec(learner, query, debug)
    # embeddings = [embeddings[0]]
    indices = range(len(embeddings))
    with_index = zip(indices, embeddings)
    params = {"queryVector" + str(idx): e.tolist() for idx, e in with_index}
    queries = [
        "cosineSimilarity(params.queryVector"
        + str(idx)
        + ", doc['embeddings_"
        + str(idx)
        + "'])"
        for idx in indices
    ]
    q = {
        "size": 1,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {"source": "+".join(queries) + "+0.0", "params": params},
            }
        },
    }
    result = measure("search", lambda: es.search(index=index_name, body=q), debug)
    return result["hits"]["hits"][0]["_source"]["title"]
