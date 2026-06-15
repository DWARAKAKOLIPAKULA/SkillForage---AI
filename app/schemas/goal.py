from pydantic import BaseModel
from typing import Optional

class GoalCreate(BaseModel):
    title: str
    description: Optional[str] = None

class GoalResponse(BaseModel):
    id: str
    title: str
    description: Optional[str]
    user_id: str

    class Config:
        from_attributes = True