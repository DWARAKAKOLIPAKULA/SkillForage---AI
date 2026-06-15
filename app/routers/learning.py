from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.agents.graph import skillforge_graph
from app.models.user import User
from app.core.dependencies import get_current_user

router = APIRouter(prefix="/learn", tags=["Learning Path"])

class LearningRequest(BaseModel):
    goal: str

@router.post("/generate")
async def generate_learning_path(
    request: LearningRequest,
    current_user: User = Depends(get_current_user)
):
    # Run the full agent graph
    result = await skillforge_graph.ainvoke({
        "user_goal": request.goal,
        "topics": [],
        "resources": [],
        "quiz_questions": [],
        "final_path": "",
        "messages": []
    })

    return {
        "goal": request.goal,
        "topics": result["topics"],
        "resources": result["resources"],
        "quiz_questions": result["quiz_questions"],
        "learning_path": result["final_path"]
    }