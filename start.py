# acclaim
print("invoke start.py ", end='')
COLOR_OK = '\033[92m' # light green
COLOR_DEFAULT = '\033[0m'
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
os.environ["OPENAI_API_KEY"] = "sk-"
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
