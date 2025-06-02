# acclaim
print("invoke start.py ", end='')
COLOR_OK = '\033[92m' # light green
COLOR_DEFAULT = '\033[0m'
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# disable warnings
print("disable warnings ", end='')
import warnings
warnings.filterwarnings("ignore", message=r"urllib3 v2 only supports")
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# read config
print("importing config/config.ini ", end='')
import configparser
config = configparser.ConfigParser()
config.read("config/config.ini")
OPEN_AI_KEY = config.get("OpenAI", "OPEN_AI_KEY")
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# check OpenAI key validity
import openai
client = openai.OpenAI(api_key=OPEN_AI_KEY)
try:
    models = client.models.list()
    print("OpenAI Key is valid ", end='')
    print(COLOR_OK + "OK" + COLOR_DEFAULT)
except Exception as e:
    print("OpenAI Key is NOT valid")
    print(e)

# import libs
print("import LlamaIndex lib ", end='')
import os
from llama_index.core import ( VectorStoreIndex, SimpleDirectoryReader)
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# assign OpenAI key
print("set OpenAI key ", end='')
os.environ["OPENAI_API_KEY"] = OPEN_AI_KEY
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# load documents
docRoot ="./data/"
print("load documents [" + docRoot + "] ", end='')
documents = SimpleDirectoryReader(docRoot).load_data()
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# build search index
print("build search index ", end='')
index = VectorStoreIndex.from_documents(documents=documents)
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# specify question
QUESTION = "What did Sam Altman do in this essay?"

# -------------------------------------------------------------

# query without reranker
query_engine = index.as_query_engine(
    similarity_top_k=10,
)
response = query_engine.query(QUESTION)

"""
# original responses
print(">> Original Responses (no Reranker):")
for node in response.source_nodes:
    print(node.id_)
    print(node.node.get_content()[:120])
    print("reranking score: ", node.score)
    print("**********")
"""

# best response
print(">> Best Response (no Reranker):")
print(response)

# -------------------------------------------------------------

# query with reranker
from llama_index.postprocessor.colbert_rerank import ColbertRerank
colbert_reranker = ColbertRerank(
    top_n=5,
    model="colbert-ir/colbertv2.0",
    tokenizer="colbert-ir/colbertv2.0",
    keep_retrieval_score=True,
)
query_engine_rr = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[colbert_reranker],
)
response_rr = query_engine_rr.query(QUESTION)

"""
# original responses
print(">> Original Responses (with Reranker):")
for node_rr in response_rr.source_nodes:
    print(node_rr.id_)
    print(node_rr.node.get_content()[:120])
    print("reranking score: ", node_rr.score)
    print("retrieval score: ", node_rr.node.metadata["retrieval_score"])
    print("**********")
"""

# best response
print(">> Best Response (with Reranker):")
print(response_rr)
