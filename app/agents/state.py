from typing import TypedDict, Annotated
import operator

class SkillForgeState(TypedDict):
    user_goal: str                          # input from user
    topics: list[str]                       # filled by Planner
    resources: list[dict]                   # filled by Resource agent
    quiz_questions: list[str]               # filled by Quiz agent
    final_path: str                         # filled at end
    messages: Annotated[list, operator.add] # conversation history — auto-appended