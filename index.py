
# A demo showing hybrid semantic search with dense and sparse vectors using Milvus.
#
# You can optionally choose to use the BGE-M3 model to embed the text as dense
# and sparse vectors, or simply use random generated vectors as an example.
#
# You can also use the BGE CrossEncoder model to rerank the search results.
#
# Note that the sparse vector search feature is only available in Milvus 2.4.0 or
# higher version. Make sure you follow https://milvus.io/docs/install_standalone-docker.md
# to set up the latest version of Milvus in your local environment.

# To connect to Milvus server, you need the python client library called pymilvus.
# To use BGE-M3 model, you need to install the optional `model` module in pymilvus.
# You can get them by simply running the following commands:
#
# pip install pymilvus
# pip install pymilvus[model]

# If true, use BGE-M3 model to generate dense and sparse vectors.
# If false, use random numbers to compose dense and sparse vectors.
use_bge_m3 = True
# If true, the search result will be reranked using BGE CrossEncoder model.
use_reranker = True

# The overall steps are as follows:
# 1. embed the text as dense and sparse vectors
# 2. setup a Milvus collection to store the dense and sparse vectors
# 3. insert the data to Milvus
# 4. search and inspect the result!
import random
import string
import numpy as np
import pandas as pd

from pymilvus import (
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection, AnnSearchRequest, RRFRanker, connections,
)

# 1. prepare a small corpus to search
#docs = [
#    "Artificial intelligence was founded as an academic discipline in 1956.",
#    "Alan Turing was the first person to conduct substantial research in AI.",
#    "Born in Maida Vale, London, Turing was raised in southern England.",
#]


file_path = 'quora_duplicate_questions.tsv'
df = pd.read_csv(file_path, sep='\t')
questions = set()
for index, row in df.iterrows():
    obj = row.to_dict()
    #print(obj['question1'], obj['question2'])
    questions.add(obj['question1'][:512])
    questions.add(obj['question2'][:512])
    if len(questions) > 10000:
        break

docs = list(questions)

# add some randomly generated texts
#docs.extend([' '.join(''.join(random.choice(string.ascii_lowercase) for _ in range(random.randint(1, 8))) for _ in range(10)) for _ in range(1000)])
query = "Who started AI research?"

def random_embedding(texts):
    rng = np.random.default_rng()
    return {
        "dense": np.random.rand(len(texts), 768),
        "sparse": [{d: rng.random() for d in random.sample(range(1000), random.randint(20, 30))} for _ in texts],
    }

dense_dim = 768
ef = random_embedding

if use_bge_m3:
    # BGE-M3 model can embed texts as dense and sparse vectors.
    # It is included in the optional `model` module in pymilvus, to install it,
    # simply run "pip install pymilvus[model]".
    from pymilvus.model.hybrid import BGEM3EmbeddingFunction
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cuda")
    dense_dim = ef.dim["dense"]

docs_embeddings = ef(docs)
query_embeddings = ef([query])

# 2. setup Milvus collection and index
#connections.connect("default", host="localhost", port="19530")
connections.connect("default", uri="milvus.db")

# Specify the data schema for the new Collection.
fields = [
    # Use auto generated id as primary key
    FieldSchema(name="pk", dtype=DataType.VARCHAR,
                is_primary=True, auto_id=True, max_length=100),
    # Store the original text to retrieve based on semantically distance
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
    # Milvus now supports both sparse and dense vectors, we can store each in
    # a separate field to conduct hybrid search on both vectors.
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR,
                dim=dense_dim),
]
schema = CollectionSchema(fields, "")
col_name = 'hybrid_demo'
# Now we can create the new collection with above name and schema.
col = Collection(col_name, schema, consistency_level="Strong")

# We need to create indices for the vector fields. The indices will be loaded
# into memory for efficient search.
sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
col.create_index("sparse_vector", sparse_index)
dense_index = {"index_type": "FLAT", "metric_type": "IP"}
col.create_index("dense_vector", dense_index)
col.load()

# 3. insert text and sparse/dense vector representations into the collection
entities = [docs, docs_embeddings["sparse"], docs_embeddings["dense"]]
for i in range(0, len(docs), 50):
    print(i)
    batched_entities= [docs[i: i + 50], docs_embeddings["sparse"][i: i + 50], docs_embeddings["dense"][i: i+50]]
    col.insert(batched_entities)
col.flush()

