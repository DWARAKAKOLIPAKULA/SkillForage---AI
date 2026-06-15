from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.state import SkillForgeState
from app.core.config import settings
import json

def planner_agent(state: SkillForgeState) -> dict:
    """
    Reads user_goal from state.
    Returns topics list — subtopics the user needs to learn.
    """
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0.3,
        request_timeout=30
    )

    messages = [
        SystemMessage(content="""You are a learning path planner.
Given a user's learning goal, break it down into 3-5 specific subtopics they need to learn.
Respond ONLY with a JSON array of strings. No explanation, no markdown, just the array.
Example: ["topic 1", "topic 2", "topic 3"]"""),
        HumanMessage(content=f"Learning goal: {state['user_goal']}")
    ]

    response = llm.invoke(messages)

    try:
        topics = json.loads(response.content)
        if not isinstance(topics, list):
            topics = [state['user_goal']]
    except json.JSONDecodeError:
        # fallback if LLM doesn't return clean JSON
        topics = [state['user_goal']]

    return {"topics": topics}