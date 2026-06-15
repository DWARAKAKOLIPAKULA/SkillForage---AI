from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os
from app.core.config import settings

FAISS_PATH = "faiss_index"

def get_embeddings():
    return OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=settings.OLLAMA_BASE_URL
    )

def ingest_texts(texts: list[str], metadatas: list[dict] = None) -> int:
    # Step 1 — Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )

    documents = []
    for i, text in enumerate(texts):
        chunks = splitter.create_documents(
            [text],
            metadatas=[metadatas[i] if metadatas else {"source": f"doc_{i}"}]
        )
        documents.extend(chunks)

    # Step 2 — Embed and store
    embeddings = get_embeddings()

    if os.path.exists(FAISS_PATH):
        vectorstore = FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(documents)
    else:
        vectorstore = FAISS.from_documents(documents, embeddings)

    # Step 3 — Save
    vectorstore.save_local(FAISS_PATH)
    return len(documents)