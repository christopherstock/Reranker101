# acclaim
from pydantic import SecretStr

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

from langchain_community.document_loaders import TextLoader, DirectoryLoader

# load documents
documents = DirectoryLoader("./data/", glob="**/*.txt", loader_cls=TextLoader).load()
print("load [" + str(len(documents)) + "] documents OK")

# split documents using recursive character text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 150)
splits = text_splitter.split_documents(documents)
print("split documents OK")

# create embeddings
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
embedding = OpenAIEmbeddings()
print("create embeddings OK")

# create vector store
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory='.vstore/chroma/'
)
print("create [" + str(vectordb._collection.count()) + "] vector store DB entries OK")

# launch query
docs = vectordb.similarity_search("What did Sam Altman do in this essay?", k=10)

print("best response (no reranker):")
print(docs[0].page_content)

from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereEmbeddings, ChatCohere, CohereRerank, CohereRagRetriever
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

# create Cohere LLM
llm = ChatCohere(cohere_api_key=SecretStr("your-Cohere-API-key"), model="command-a-03-2025")
print("create Cohere LLM OK")

# create embedding model
embeddings = CohereEmbeddings(
    cohere_api_key=SecretStr("your-Cohere-API-key"),
    model="embed-english-light-v3.0"
)
print("create embedding model OK")

# load documents
documents = DirectoryLoader("./data/", glob="**/*.txt", loader_cls=TextLoader).load()
print("load [" + str(len(documents)) + "] documents OK")

# split documents
text_splitter = CharacterTextSplitter(chunk_size = 750, chunk_overlap = 150, separator="\n")
splits = text_splitter.split_documents(documents)
print("split text files OK")

# create vector store from documents
db = Chroma.from_documents(splits, embeddings)
print("create vecor store DB OK")

# create Cohere Reranker
reranker = CohereRerank(
    cohere_api_key=SecretStr("your-Cohere-API-key"), model="rerank-english-v3.0"
)
compression_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=db.as_retriever())
compressed_docs = compression_retriever.get_relevant_documents("What ")
print("create Cohere Reranker OK' ", end='')

# create Cohere RAG
rag = CohereRagRetriever(llm=llm, connectors=[])
docs = rag.get_relevant_documents("What did Sam Altman do in this essay?", documents=compressed_docs)
print("create Cohere RAG instance OK")

# Print the final generation
answer = docs[-1].page_content
print("best response (with Cohere reranker):")
print(answer)
