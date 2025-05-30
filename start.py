print("OK start.py invoked")

import warnings
warnings.filterwarnings("ignore", message=r"^urllib3 v2 only supports OpenSSL")
print("OK disabled warnings")

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
)
print("OK completed imports")

import os
os.environ["OPENAI_API_KEY"] = "sk-"
print("OK set OpenAPI key")

print("OK Sys Lib Path:")
import sys
print(sys.path)
