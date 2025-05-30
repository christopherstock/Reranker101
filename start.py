# acclaim
print("invoke start.py ", end='')
COLOR_OK = '\033[92m' # light green
COLOR_DEFAULT = '\033[0m'
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# read config
print("importing config.ini ", end='')
import configparser
config = configparser.ConfigParser()
config.read("config.ini")
OPEN_API_KEY = config.get("OpenAPI", "OPEN_API_KEY")
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# disable warnings
print("disable warnings ", end='')
import warnings
warnings.filterwarnings("ignore", message=r"^urllib3 v2 only supports OpenSSL")
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# import libs
print("import LlamaIndex lib ", end='')
import os
import sys
from llama_index.core import ( VectorStoreIndex, SimpleDirectoryReader)
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# assign OpenAPI key
print("set OpenAPI key ", end='')
os.environ["OPENAI_API_KEY"] = OPEN_API_KEY
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# load documents
docRoot ="./data/paul_graham/"
print("load documents [" + docRoot + "] ", end='')
documents = SimpleDirectoryReader(docRoot).load_data()
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# build search index
print("build search index ", end='')
index = VectorStoreIndex.from_documents(documents=documents)
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# magic!
from llama_index.postprocessor.colbert_rerank import ColbertRerank
colbert_reranker = ColbertRerank(
    top_n=5,
    model="colbert-ir/colbertv2.0",
    tokenizer="colbert-ir/colbertv2.0",
    keep_retrieval_score=True,
)
query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[colbert_reranker],
)
response = query_engine.query(
    "What did Sam Altman do in this essay?",
)

# original response
for node in response.source_nodes:
    print(node.id_)
    print(node.node.get_content()[:120])
    print("reranking score: ", node.score)
    print("retrieval score: ", node.node.metadata["retrieval_score"])
    print("**********")

# best response
print(response)

# next query
response = query_engine.query(
    "Which schools did Paul attend?",
)

# all responses
for node in response.source_nodes:
    print(node.id_)
    print(node.node.get_content()[:120])
    print("reranking score: ", node.score)
    print("retrieval score: ", node.node.metadata["retrieval_score"])
    print("**********")

# best response
print(response)
