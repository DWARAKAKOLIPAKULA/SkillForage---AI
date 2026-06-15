from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.services.ingestion import ingest_texts
from app.services.retrieval import search_similar, search_and_answer
from app.models.user import User
from app.core.dependencies import get_current_user

router = APIRouter(prefix="/rag", tags=["RAG"])

class IngestRequest(BaseModel):
    texts: list[str]
    sources: list[str] = []

@router.post("/ingest")
async def ingest(
    request: IngestRequest,
    current_user: User = Depends(get_current_user)
):
    metadatas = [
        {"source": src, "user_id": current_user.id}
        for src in (request.sources or [f"doc_{i}" for i in range(len(request.texts))])
    ]
    count = ingest_texts(request.texts, metadatas)
    return {"message": f"Ingested {count} chunks successfully"}

@router.get("/search")
async def search(
    query: str,
    top_k: int = 5,
    current_user: User = Depends(get_current_user)
):
    """Pure vector search — returns similar chunks"""
    results = search_similar(query, top_k)
    if not results:
        raise HTTPException(status_code=404, detail="No documents ingested yet")
    return {"query": query, "results": results}

@router.get("/ask")
async def ask(
    query: str,
    current_user: User = Depends(get_current_user)
):
    """Full RAG — retrieves context and generates answer with Groq"""
    result = search_and_answer(query)
    return result