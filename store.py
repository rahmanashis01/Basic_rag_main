import asyncio
from utils import load_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from uuid import uuid4
import chromadb

embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

pages = asyncio.run(load_pdf("./linux-note-for-rag.pdf"))

client = chromadb.PersistentClient(path="./chroma_langchain_db")

collection = client.get_or_create_collection("example_collection")


vector_store = Chroma(
    client=client,
    collection_name="example_collection",
    embedding_function=embeddings,
)

# print(f"{pages[0].metadata}\n")
# print(pages[0].page_content)
# print("total pages = ", len(pages))

page_contents = ""
for page in pages:
    page_contents = page_contents + "\n" + page.page_content

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
texts = text_splitter.split_text(page_contents)

documents = []

for index,text in enumerate(texts):
    documents.append(
        Document(
            page_content=text,
            metadata={"chunk_num": index},
            id=index
        )
    )

uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)
