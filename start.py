# acclaim
print("invoke start.py ", end='')
print("OK")

# disable warnings
print("disable warnings ", end='')
import warnings
warnings.filterwarnings("ignore", message=r"^urllib3 v2 only supports OpenSSL")
print("OK")

# import libs
print("import LlamaIndex lib ", end='')
import os
import sys
from llama_index.core import ( VectorStoreIndex, SimpleDirectoryReader)
print("OK")

# assign OpenAPI key
print("set OpenAPI key ", end='')
os.environ["OPENAI_API_KEY"] = "sk-"
print("OK")

# print("OK Sys Lib Path:")
# print(sys.path)
