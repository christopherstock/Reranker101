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

# assign OpenAI key
import os
print("set OpenAI key ", end='')
os.environ["OPENAI_API_KEY"] = OPEN_AI_KEY
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
print("import LangChain lib ", end='')
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# load documents
docRoot ="./data/"
print("load documents [" + docRoot + "] ", end='')
documents = DirectoryLoader(docRoot, glob="**/*.txt", loader_cls=TextLoader).load()
print(COLOR_OK + "OK" + COLOR_DEFAULT)
print("[" + str(len(documents)) + "] docs loaded")

# try LangChain OpenAI prompt
from langchain_openai import OpenAI
llm = OpenAI()
print("Answer: " + llm.invoke("Hello how are you?"))

# read text file
print("text File read ", end='')
with open("./data/paul_graham_essay.txt") as f:
    text_file = f.read()
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# split text into chunks
print("split text file ", end='')
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
chunks = text_splitter.split_text(text_file)
print(COLOR_OK + "OK" + COLOR_DEFAULT)
print("split to [" + str(len(chunks)) + "] chunks ", end='')
print(COLOR_OK + "OK" + COLOR_DEFAULT)



exit(0)



# build search index
print("build search index ", end='')
search_index = VectorStoreIndex.from_documents(documents=documents)
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# specify question
QUESTION = "What did Sam Altman do in this essay?"

# -------------------------------------------------------------

# query without reranker
query_engine = search_index.as_query_engine(
    similarity_top_k=10,
)
response = query_engine.query(QUESTION)

"""
# original responses
print()
print(">> Original Responses (no Reranker):")
for node in response.source_nodes:
    print(node.id_)
    print(node.node.get_content()[:120])
    print("ranking score: ", node.score)
    print("**********")
"""

# best response
print()
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
query_engine_rr = search_index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[colbert_reranker],
)
response_rr = query_engine_rr.query(QUESTION)

"""
# original responses
print()
print(">> Original Responses (with Reranker):")
for node_rr in response_rr.source_nodes:
    print(node_rr.id_)
    print(node_rr.node.get_content()[:120])
    print("reranking score: ", node_rr.score)
    print("retrieval score: ", node_rr.node.metadata["retrieval_score"])
    print("**********")
"""

# best response
print()
print(">> Best Response (with Reranker):")
print(response_rr)
