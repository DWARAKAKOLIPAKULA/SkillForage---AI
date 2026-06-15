from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import os
from app.core.config import settings

FAISS_PATH = "faiss_index"

def get_embeddings():
    return OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=settings.OLLAMA_BASE_URL
    )

def get_llm():
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model="llama-3.1-8b-instant",   # free and fast on Groq
        temperature=0.2,
        request_timeout=30
    )

def search_similar(query: str, top_k: int = 5) -> list[dict]:
    """Pure vector search — returns top-K similar chunks"""
    if not os.path.exists(FAISS_PATH):
        return []

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    results = vectorstore.similarity_search_with_score(query, k=top_k)
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": round(float(score), 4)
        }
        for doc, score in results
    ]

def search_and_answer(query: str, top_k: int = 3) -> dict:
    """
    Full RAG — retrieve relevant chunks then generate answer with Groq.
    This is the complete RAG pipeline in one function.
    """
    if not os.path.exists(FAISS_PATH):
        return {"answer": "No knowledge base found. Please ingest documents first.", "sources": []}

    # Step 1 — Retrieve
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    results = vectorstore.similarity_search(query, k=top_k)

    # Step 2 — Build context from retrieved chunks
    context = "\n\n".join([
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in results
    ])

    # Step 3 — Augment prompt with context and send to Groq
    llm = get_llm()
    messages = [
        SystemMessage(content="""You are a learning assistant for SkillForge AI. 
Answer the user's question using ONLY the provided context. 
If the context doesn't contain the answer, say 'I don't have enough information on this topic.'
Be concise and practical."""),
        HumanMessage(content=f"""Context:
{context}

Question: {query}

Answer:""")
    ]

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": [doc.metadata.get("source", "unknown") for doc in results]
    }