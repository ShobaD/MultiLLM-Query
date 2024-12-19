# Function to process each PDF


from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Document,
    Settings,
)
#from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from dotenv import load_dotenv
import os
import toml
import qdrant_client
import pandas as pd
from qdrant_client.http.models import Distance
import pymupdf4llm
import xml.etree.ElementTree as ET
import json
import os
import shutil
import openai
import numpy as np

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    Document
)
import chardet
from llama_index.embeddings.openai import OpenAIEmbedding
import openai
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
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()
splitter = SentenceSplitter(
    #chunk_size=CHUNK_SIZE,
    #chunk_overlap=CHUNK_OVERLAP,
    chunk_size=2000,
    chunk_overlap=100,
)

client = qdrant_client.QdrantClient(
    #url=qdrant_url,
    #port=qdrant_port,
    #api_key=qdrant_api_key,
    
)

qdrant_url = os.getenv("VECTORQUADRANT_URL")
qdrant_port = os.getenv("QDRANT_PORT")
qdrant_api_key = os.getenv("VECTORQUADRANT_KEY")

client = qdrant_client.QdrantClient(
    url=qdrant_url,
    port=6333,
    api_key=qdrant_api_key,
)


splitter = SentenceSplitter(
    chunk_size=256,
    chunk_overlap=20,
)
openai.api_key = os.getenv("OPENAI_API_KEY")
documents = []
# Initialize Qdrant client and vector store
client = QdrantClient(url=os.getenv("VECTORQUANDRANT_URL"), port=6333,api_key=os.getenv("VECTORQUADRANT_KEY"))
#directory = Path("data")  # Specify your directory path here
# Example directory path
openai.api_key = os.getenv("OPENAI_API_KEY")


directory_path = "Books/"

LLMclient = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

llm = LLMclient.models._client.models.retrieve('gpt-4o-mini')

vector_store = QdrantVectorStore(client=client, collection_name="Mukkiya_Pusthagangal", enable_hybrid=True, llm=llm)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

#vector_store = QdrantVectorStore(client=client, collection_name="pusthagam", enable_hybrid=True, Distance = Distance.COSINE)
#storage_context = StorageContext.from_defaults(vector_store=vector_store)
def process_pdf(file_path):
    reader = pymupdf4llm.to_markdown(file_path)
    print(reader)
    print(type(reader))
    #document = reader.load_and_split()
    # Create a Document instance
    document = Document(text=reader, metadata={"filename": str(file_path)})

    # Ensure the document is in a list format as expected by VectorStoreIndex
    document = [document]  # Wrap in a list
    
    index = VectorStoreIndex.from_documents(
        document,
        storage_context=storage_context,
        transformations=[splitter],
        
    )    
    return reader

# Function to process each PDF
def processdirpdf(file_path):
# Loop through all files in the directory
    for filename in os.listdir(file_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join("Books", filename)
            #print(f"Processing: {filename}")
            
            try:
                pdf_text = process_pdf(pdf_path)
                #print(f"Extracted text from {filename}:\n")
                #print(pdf_text[:1000])  # Print the first 1000 characters
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")



# Load documents and metadata

processdirpdf(directory_path)