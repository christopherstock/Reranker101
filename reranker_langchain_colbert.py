# acclaim
from llama_index.core.node_parser import SentenceSplitter

print("invoke start.py ", end='')
COLOR_OK = '\033[92m' # light green
COLOR_DEFAULT = '\033[0m'
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# disable warnings
print("disable warnings ", end='')
import warnings
warnings.filterwarnings("ignore", message=r"urllib3 v2 only supports")
warnings.filterwarnings("ignore", message=r"FutureWarning")
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
from langchain_community.document_loaders import TextLoader, DirectoryLoader, UnstructuredHTMLLoader
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# try LangChain OpenAI prompt
from langchain_openai import OpenAI
llm = OpenAI()
print("Answer: " + llm.invoke("Hello how are you?"))

# load documents
docRoot ="./data/"
print("load documents [" + docRoot + "] ", end='')
documents = DirectoryLoader(docRoot, glob="**/*.txt", loader_cls=TextLoader).load()
print(COLOR_OK + "OK" + COLOR_DEFAULT)
print("loaded [" + str(len(documents)) + "] docs ", end='')
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# split documents using recursive character text splitter
print("split documents ", end='')
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    chunk_size = 750,
    chunk_overlap = 150,
    separator="\n\n\n",
)
splits = text_splitter.split_documents(documents)
print("(" + str(len(splits)) + " splits) " + COLOR_OK + "OK" + COLOR_DEFAULT)

# create embeddings
print("create embeddings ", end='')
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
embedding = OpenAIEmbeddings()
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# create vector store
persist_directory = '.vstore/chroma/'
print("create vector store ", end='')
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
print(COLOR_OK + "OK" + COLOR_DEFAULT)
print("created [" + str(vectordb._collection.count()) + "] vectorDB entries")

# -------------------------------------------------------------
# launch query (no reranker)

print("launch query ", end='')
QUESTION = "Which emotions does the heart chakra relate to?"
docs = vectordb.similarity_search(QUESTION, k=10)
print(COLOR_OK + "OK" + COLOR_DEFAULT)
# Check the number of results
print("found [" + str(len(docs)) + "] results ", end='')
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# best response
print()
print(">> Best Response (no Reranker, VectorDB & embeddings, similarity):")
print(docs[0].page_content)
print()

# -------------------------------------------------------------
# use LangChain Lib 'RAGatouille' + ColBERT 2.0 Reranker

print("import Lib 'RAGatouille' ", end='')
from ragatouille import RAGPretrainedModel
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# create RAG index from document
print("create 'RAGatouille' RAG index ", end='')
with open("./data/chakras.txt","r") as file:
    content = file.read()

"""
print(">> Content:")
print("")
print(content)
"""

RAG.index(
    collection=[content],
    index_name="rag-with-reranker",
    max_document_length=180,
    split_documents=True,
    # document_splitter_fn=(SentenceSplitter.from_defaults().split_text),
)
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# launch question as query -- with reranker
print("query question using RAG Index + ColBERT reranker ", end='')
results = RAG.search(query=QUESTION, k=10)
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# best response
print()
print(">> Best Response (Ratatouille RAG + ColBERT Reranker):")
print(results[0]['content'])
print()
