from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.models.goal import Goal
from app.models.user import User
from app.schemas.goal import GoalCreate, GoalResponse
from app.core.dependencies import get_current_user

router = APIRouter(prefix="/goals", tags=["Goals"])

@router.post("", status_code=status.HTTP_201_CREATED, response_model=GoalResponse)
async def create_goal(
    goal_data: GoalCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    goal = Goal(
        title=goal_data.title,
        description=goal_data.description,
        user_id=current_user.id
    )
    db.add(goal)
    await db.flush()   # assigns the id before commit
    return goal

@router.get("", response_model=list[GoalResponse])
async def get_goals(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(Goal).where(Goal.user_id == current_user.id)
    )
    return result.scalars().all()