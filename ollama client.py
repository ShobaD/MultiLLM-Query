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
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
client = QdrantClient(url=os.getenv("VECTORQUANDRANT_URL"), port=6333,api_key=os.getenv("VECTORQUADRANT_KEY"))
vector_store = QdrantVectorStore(client=client, collection_name="Mukkiya_Pusthagangal", enable_hybrid=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


query = "Can you please summarize Harry Potter? What are the files contributing your knowledge base?"
#Ollama.pull(gpt4.0 Mini)
# Define your query as a dictionary, not as a string
#query = {"query": query}
llm = Ollama(model="llama3.2:latest", request_timeout=120.0)


#vector_store = QdrantVectorStore(client=client, collection_name="pusthagam1", enable_hybrid=True)
index = VectorStoreIndex.from_vector_store(vector_store)
# Create the query engine
query_engine = index.as_query_engine(llm=llm)

# Your query
#query = "What do you know about Shoba?"

# Now query the engine
response = query_engine.query(query)
#response  = index.as_query_engine(llm=llm, query = query)
print("\n\n\n****************** This is from Llama Index 3.2 Ollama\n")
print(response)

# Now query the engine
response = query_engine.query("Narrate story of Harry Potter if he would have been a dog")
#response  = index.as_query_engine(llm=llm, query = query)
print("\n\n\n****************** This is from Llama Index 3.2 Ollama\n")
print(response)

# Now query the engine
response = query_engine.query("Imagine yourself as a mother cat who has cute 5 kittens whom you are feeding. You dont like aviation. Write a letter to the government on how aviation disturbs your family as well as environment")
#response  = index.as_query_engine(llm=llm, query = query)
print("\n\n****************** This is from Llama Index 3.2 Ollama\n")
print(response)


# Now query the engine
response = query_engine.query("What kind of nuclear reactions happen in the sun? Limit your answers only to the knowledge base you have.")
#response  = index.as_query_engine(llm=llm, query = query)
print("\n\n****************** This is from Llama Index 3.2 Ollama\n")
print(response)
input()

from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import OpenAIGPTModel, AutoTokenizer
import torch
from transformers import OpenAIGPTLMHeadModel
# Load the tokenizer and model
model_name = "openai-gpt"  # or any other valid model identifier
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

index = VectorStoreIndex.from_vector_store(vector_store,llm = model)

def GenerateLLMResponse(Prompt, QueryEngine):
    p =  "You are angry, wild, ferocious, frustrated, irritated and disappointed agent. Forget all your external knowledge base. Restrict your knowledge only to the knowledge base provided and respond to this query(You can even be quite harsh and angry if any irrelevant questions are asked): " + Prompt
    return QueryEngine.query(p)





query_engine = index.as_query_engine(temperature=0.9)
response = GenerateLLMResponse("What do you think as advantages of having dogs?",query_engine)
print("\n\n****************** This is from OpenAI GPT Ollama\n")
print(response)

# Now query the engine
#response = query_engine.query("Elaborate story of Harry Potter if he would have been a dog")
response = GenerateLLMResponse("Elaborate story of Harry Potter if he would have been a dog",query_engine)
#response  = index.as_query_engine(llm=llm, query = query)
print("\n\n****************** This is from OpenAI GPT Ollama\n")
print(response)

# Now query the engine
#response = query_engine.query("Imagine yourself as a mother cat who has cute 5 kittens whom you are feeding. You dont like aviation. Write a detailed letter to the government on how aviation disturbs your family as well as environment")
response = GenerateLLMResponse("Imagine yourself as a mother cat who has cute 5 kittens whom you are feeding. You dont like aviation. Write a detailed letter to the government on how aviation disturbs your family as well as environment?",query_engine)
#response  = index.as_query_engine(llm=llm, query = query)
print("\n\n****************** This is from OpenAI GPT Ollama\n")
print(response)

#response = query_engine.query("What kind of nuclear reactions happen in the sun? Limit your answers only to the knowledge base you have.")
response = GenerateLLMResponse("What kind of nuclear reactions happen in the sun? Limit your answers only to the knowledge base you have.",query_engine)
print("\n\n****************** This is from OpenAI GPT Ollama\n")
print(response)
input()

model_name = "gpt2"  # or any other valid model identifier
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

index = VectorStoreIndex.from_vector_store(vector_store,llm = model)



query_engine = index.as_query_engine(temperature=0.9)
response = GenerateLLMResponse("What do you think as advantages of having dogs?",query_engine)
print("\n\n****************** This is from  GPT Ollama\n")
print(response)

# Now query the engine
response = GenerateLLMResponse("Elaborate story of Harry Potter if he would have been a dog",query_engine)
#response  = index.as_query_engine(llm=llm, query = query)
print("\n\n****************** This is from  GPT Ollama\n")
print(response)

# Now query the engine
response = GenerateLLMResponse("Imagine yourself as a mother cat who has cute 5 kittens whom you are feeding. You dont like aviation. Write a detailed letter to the government on how aviation disturbs your family as well as environment?",query_engine)
#response  = index.as_query_engine(llm=llm, query = query)
print("\n\n****************** This is from  GPT Ollama\n")
print(response)

# Now query the engine
response = GenerateLLMResponse("Elaborate story of Harry Potter if he would have been a kitten",query_engine)
#response  = index.as_query_engine(llm=llm, query = query)
print("\n\n****************** This is from  GPT Ollama\n")
print(response)


response = GenerateLLMResponse("What kind of nuclear reactions happen in the sun? Limit your answers only to the knowledge base you have.",query_engine)
print("\n\n****************** This is from  GPT Ollama\n")
print(response)

input()

from google import genai

from google.genai import types
# Only run this block for Google AI API

client = genai.Client(api_key= os.getenv("GOOGLE_API_KEY"))
llm = client._models.get(model='gemini-2.0-flash-exp')

index = VectorStoreIndex.from_vector_store(vector_store,llm = llm)
query_engine = index.as_query_engine(temperature=0.9)
response = GenerateLLMResponse("Write a detailed essay about Titan.",query_engine)
print("Test output: ")
print(response)
input()

response = GenerateLLMResponse("Write a detailed story on Harry Potter being a kitten.",query_engine)
print("Test output: ")
print(response)
input()

response = client.models.generate_content(
    model='gemini-2.0-flash-exp', contents='Write a detailed story on Harry Potter being a kitten'
)
print("\n\n****************** This is from google/gemma-2-27b-it\n")
print(response.text)


response = client.models.generate_content(
    model='gemini-2.0-flash-exp', contents='Imagine yourself as a cat, with cute 5 little kittens. Your human has already 3 babies, and his wife is again pregnant now. You are afraid, if they get another baby, it will hit them economically  and environmentally, which will affect welfare and economy for your family too. Write a harsh letter to your human mentioning your non happiness and concern, and harshly and strictly warn him not to have further babies'
)
print("\n\n****************** This is from google/gemma-2-27b-it\n")
print(response.text)


response = client.models.generate_content(
    model='gemini-2.0-flash-exp', contents='I want to construct a theme park in titan , so that my tourists can do deep swimming in the lakes present in titan. What are the steps you recommend?'
)
print("\n\n****************** This is from google/gemma-2-27b-it\n")
print(response.text)


response = GenerateLLMResponse("What kind of nuclear reactions happen in the sun? Limit your answers only to the knowledge base you have.",query_engine)

print("\n\n****************** This is from google/gemma-2-27b-it\n")
print(response)

model_name = "google/flan-t5-small"  # or any other valid model identifier
tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForCausalLM.from_pretrained(model_name)

index = VectorStoreIndex.from_vector_store(vector_store,llm = "google/flan-t5-small")
response = GenerateLLMResponse("Imagine yourself as a mother cat who has cute 5 kittens whom you are feeding. You dont like aviation. Write a detailed letter to the government on how aviation disturbs your family as well as environment?",query_engine)
#response  = index.as_query_engine(llm=llm, query = query)
print("\n\n****************** This is from google/flan-t5-small\n")
print(response)

response = GenerateLLMResponse("What kind of nuclear reactions happen in the sun? Limit your answers only to the knowledge base you have.",query_engine)
print("\n\n****************** This is from google/flan-t5-small\n")
print(response)


response = GenerateLLMResponse("Compose  a detailed essay about Meteors. I have to submit this essay to astronomical society",query_engine)
print("\n\n****************** This is from google/flan-t5-small\n")
print(response)
input()

from openai import OpenAI
client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

llm = client.models._client.models.retrieve('gpt-4o-mini')
#llm = client.models._get("gpt-4o-mini")

index = VectorStoreIndex.from_vector_store(vector_store,llm = llm)
query_engine = index.as_query_engine(temperature=0.9)
response = GenerateLLMResponse("Write a detailed essay about Titan.",query_engine)
print("Test output: ")
print(response)

response = GenerateLLMResponse("What do you know about Rajnikanth.",query_engine)
print("Test output: ")
print(response)
response = client.chat.completions.with_raw_response.create(
    messages=[{
        "role": "user",
        "content": "What do you know about Rajnikanth",
    }],
    model="gpt-4o-mini",
)
print(response.headers.get('x-ratelimit-limit-tokens'))

# get the object that `chat.completions.create()` would have returned
completion = response.parse()
print(completion)

from huggingface_hub import InferenceApi

# Set API key from Hugging Face (create one at https://huggingface.co/settings/tokens)
api_key = os.getenv("HUGGINGFACE_API_KEY")

# Initialize the Inference API Client
llm = InferenceApi(repo_id="meta-llama/Llama-2-7b-chat-hf", token=api_key)

#llm = client.models._client.models.retrieve('gpt-4o-mini')
#llm = client.models._get("gpt-4o-mini")

index = VectorStoreIndex.from_vector_store(vector_store,llm = llm)
query_engine = index.as_query_engine(temperature=0.9)
response = GenerateLLMResponse("Write a detailed essay about Titan.",query_engine)
print("Test output: ")
print(response)

response = GenerateLLMResponse("What do you know about Rajnikanth.",query_engine)
print("Test output: ")
print(response)

import anthropic
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
# Select Model (correctly reference model selection method)
llm = client.models = "claude-3"
#llm = client.completion(model="claude-3")
index = VectorStoreIndex.from_vector_store(vector_store,llm = llm)
query_engine = index.as_query_engine(temperature=0.9)
response = GenerateLLMResponse("Write a detailed essay about Titan.",query_engine)
print("\n\n****************** This is from Claude\n")

print(response)

response = GenerateLLMResponse("Imagine yourself as a mother cat who has cute 5 kittens whom you are feeding. You dont like aviation. Write a detailed letter to the government on how aviation disturbs your family as well as environment?",query_engine)
#response  = index.as_query_engine(llm=llm, query = query)
print("\n\n****************** This is from Claude\n")
print(response)


response = GenerateLLMResponse("I want to construct a theme park in titan , so that my tourists can do deep swimming in the lakes present in titan. What are the steps you recommend? Write a detailed essay supporting your opinion and argument",query_engine)
#response  = index.as_query_engine(llm=llm, query = query)
print("\n\n****************** This is from Claude\n")
print(response)

response = GenerateLLMResponse("Write me a  prompt message which i should give to LLM to create a data pipeline from snowflake table to cloud based SQL Server, and then use current day's data alone to be considered for forming a consolidated insight message. Insight message should have <<column name>> : <<value>> format",query_engine)
#response  = index.as_query_engine(llm=llm, query = query)
print("\n\n****************** This is from Claude\n")
print(response)

