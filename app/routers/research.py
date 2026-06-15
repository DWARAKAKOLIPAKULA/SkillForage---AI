from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.agents.research_crew import create_research_crew
from app.models.user import User
from app.core.dependencies import get_current_user

router = APIRouter(prefix="/research", tags=["Research"])

class ResearchRequest(BaseModel):
    topics: list[str]

@router.post("/web")
async def web_research(
    request: ResearchRequest,
    current_user: User = Depends(get_current_user)
):
    result = await create_research_crew(request.topics)   # ← await
    return {
        "topics": request.topics,
        "research": result
    }