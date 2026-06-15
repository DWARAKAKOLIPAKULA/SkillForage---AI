from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.services.claude_service import generate_learning_path_with_cache, answer_with_cache
from app.models.user import User
from app.core.dependencies import get_current_user

router = APIRouter(prefix="/claude", tags=["Claude AI"])


class LearningPathRequest(BaseModel):
    goal: str
    top_k: int = 3


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3


@router.post("/learning-path")
async def generate_learning_path(
    request: LearningPathRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Generates a structured JSON learning path using Claude with prompt caching.
    Check token_usage in response to see cache hits vs misses.
    Call twice with same goal — second call will show cache_hit: true.
    """
    result = generate_learning_path_with_cache(
        user_goal=request.goal,
        top_k=request.top_k
    )
    return result


@router.post("/ask")
async def ask_claude(
    request: QuestionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Answers a question using Claude + RAG with prompt caching.
    Call twice — second call shows cache_hit: true and tokens_saved.
    """
    result = answer_with_cache(
        query=request.question,
        top_k=request.top_k
    )
    return result