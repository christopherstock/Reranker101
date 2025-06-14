# acclaim
from llama_index.core.node_parser import SentenceSplitter
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
COHERE_API_KEY = config.get("CohereAPI", "COHERE_API_KEY")
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# assign OpenAI & Cohere key
import os
print("set OpenAI key ", end='')
os.environ["OPENAI_API_KEY"] = OPEN_AI_KEY
os.environ["COHERE_API_KEY"] = COHERE_API_KEY
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
    # separator="\n\n\n",
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
QUESTION_1 = "Which emotions does the heart chakra relate to?"
QUESTION_2 = "What did Sam Altman do in this essay?"
QUESTION = QUESTION_2
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
# use LangChain with Reranker 'Cohere'

from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereEmbeddings
from langchain_cohere import ChatCohere
from langchain_cohere import CohereRerank, CohereRagRetriever
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

# create Cohere LLM
print("create Cohere LLM' ", end='')
llm = ChatCohere(
    cohere_api_key=SecretStr(COHERE_API_KEY),
    model="command-a-03-2025"
)
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# create embedding model
print("create embedding model' ", end='')
embeddings = CohereEmbeddings(
    cohere_api_key=SecretStr(COHERE_API_KEY),
    model="embed-english-light-v3.0"
)
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# Load text files and split into chunks, you can also use data gathered elsewhere in your application
print("load and split text files' ", end='')
text_splitter = CharacterTextSplitter(
    chunk_size = 750,
    chunk_overlap = 150,
    separator="\n",
)
splits = text_splitter.split_documents(documents)
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# Create a vector store from the documents
print("create vecor store DB' ", end='')
db = Chroma.from_documents(splits, embeddings)
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# Create Cohere's reranker with the vector DB using Cohere's embeddings as the base retriever
print("create Cohere Rerank' ", end='')
reranker = CohereRerank(
    cohere_api_key=SecretStr(COHERE_API_KEY), model="rerank-english-v3.0"
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=db.as_retriever()
)
compressed_docs = compression_retriever.get_relevant_documents(
    QUESTION
)
print(COLOR_OK + "OK" + COLOR_DEFAULT)
"""
# Print the relevant documents from using the embeddings and reranker
print(compressed_docs)
"""
# Create the cohere rag retriever using the chat model
print("create Cohere RAG instance' ", end='')
rag = CohereRagRetriever(llm=llm, connectors=[])
docs = rag.get_relevant_documents(
    QUESTION,
    documents=compressed_docs,
)
print(COLOR_OK + "OK" + COLOR_DEFAULT)

# Print the documents
"""
print("Documents:")
for doc in docs[:-1]:
    print(doc.metadata)
    print("\n\n" + doc.page_content)
    print("\n\n" + "-" * 30 + "\n\n")
"""
# Print the final generation
answer = docs[-1].page_content
print("")
print(">>> Best Answer:")
print(answer)

# Print the final citations
"""
citations = docs[-1].metadata["citations"]
print("Citations:")
print(citations)
"""
