import asyncio
from utils import load_pdf, prettify_docs
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import chromadb
import os

embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

pages = asyncio.run(load_pdf("./linux-note-for-rag.pdf"))

client = chromadb.PersistentClient(path="./chroma_langchain_db")

collection = client.get_or_create_collection("example_collection")


vector_store = Chroma(
    client=client,
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

base_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

docs = base_retriever.invoke("Explain Linux file permissions")

llm = init_chat_model(
    "gpt-4o-mini",
    model_provider="openai",
    base_url=os.environ["OPENAI_BASE_URI"],
    api_key=os.environ["OPENAI_API_KEY"]
)  

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)

print("Enter your Linux question:")
raw_prompt = input()
docs = compression_retriever.invoke(raw_prompt)
docs = prettify_docs(docs)

prompt_template = ChatPromptTemplate.from_template("""
You are a DevOps Engineer.
Use the following context to answer the question.

Context:
{context}

Question: {question}
""")

chain = prompt_template | llm

response = chain.invoke({"context": docs, "question" : raw_prompt})

print(response.content)

