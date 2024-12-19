
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    Document
)
import chardet
from llama_index.embeddings.openai import OpenAIEmbedding
#from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
#from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from dotenv import load_dotenv
import os
import toml
import qdrant_client
from llama_index.llms.ollama import Ollama
import openai
from qdrant_client import QdrantClient
from pathlib import Path
from llama_index.core import StorageContext
from llama_index.core.node_parser import SentenceSplitter





load_dotenv()
llm_deployment = os.getenv("AZURE_LLM_DEPLOYMENT")
embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.getenv("API_VERSION")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_port = os.getenv("QDRANT_PORT")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

config = toml.load("config.toml")
SIMILARITY_TOP_K = config["retrieval"]["similarity_top_k"]
QDRANT_COLLECTION_NAME = config["vectordb"]["collection_name"]
QDRANT_COLLECTION_NAME = "hcl_shoba_test1"


qdrant_url = "https://54770dc4-ce17-4c66-85d6-353c1952085a.us-east4-0.gcp.cloud.qdrant.io/"
qdrant_port = os.getenv("QDRANT_PORT")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_api_key = "gy6bWoyvkQS9aOaudxOrAjCPsoUiUeKnTozHHpPL6JdUpXfevzql2A"

client = qdrant_client.QdrantClient(
    url=qdrant_url,
    port=qdrant_port,
    api_key=qdrant_api_key,
)


splitter = SentenceSplitter(
    chunk_size=256,
    chunk_overlap=20,
)
openai.api_key = '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'
documents = []
# Initialize Qdrant client and vector store
client = QdrantClient(url="https://54770dc4-ce17-4c66-85d6-353c1952085a.us-east4-0.gcp.cloud.qdrant.io/", port=6333,api_key="gy6bWoyvkQS9aOaudxOrAjCPsoUiUeKnTozHHpPL6JdUpXfevzql2A")
directory = Path("data")  # Specify your directory path here
# Get all files in the directory
files = [f.name for f in directory.iterdir() if f.is_file()]
print(files)
# Print the list of files
for file in files:
    print(file)
    file_path = os.path.join(directory, file)  # Get the full path
    with open(file_path, 'rb') as file:
        raw_data = file.read()  # Read the file as bytes
        result = chardet.detect(raw_data)  # Detect encoding
        encoding = result['encoding']  # Get the detected encoding
    
        with open(file_path, "r", encoding=encoding) as f:
        
            file_content = f.read()
            print(f"Content of {file}:\n{file_content}\n")
    
            document = Document(text=file_content, metadata={"filename": str(file)})
            documents.append(document)

# 4. Process files to create documents


vector_store = QdrantVectorStore(client=client, collection_name="ollama", enable_hybrid=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# Create the vector store index with Ollama
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    transformations=[splitter],
    
)
print(documents)
input()
query = "What do you think about Harry Potter?"
#Ollama.pull(gpt4.0 Mini)
# Define your query as a dictionary, not as a string
#query = {"query": query}
#llm = Ollama(model="llama3.2:latest", request_timeout=120.0)
llm = Ollama(base_url="http://localhost:11434", model="llama3.2:latest", request_timeout=120.0)
#vector_store = QdrantVectorStore(client=client, collection_name="pusthagam1", enable_hybrid=True)
index = VectorStoreIndex.from_vector_store(vector_store)
# Create the query engine
query_engine = index.as_query_engine(llm=llm)

# Your query
#query = "What do you know about Shoba?"

# Now query the engine
response = query_engine.query(query)
#response  = index.as_query_engine(llm=llm, query = query)
print(response)

input()

