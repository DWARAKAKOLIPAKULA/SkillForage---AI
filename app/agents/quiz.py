from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.state import SkillForgeState
from app.core.config import settings
import json

def quiz_agent(state: SkillForgeState) -> dict:
    """
    Reads topics + resources from state.
    Generates quiz questions and final learning path summary.
    """
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0.4,
        request_timeout=30

    )

    # Build context from resources
    context = "\n".join([
        f"- {r['topic']}: {r['content']}"
        for r in state['resources']
    ]) if state['resources'] else "\n".join(state['topics'])

    # Generate quiz questions
    quiz_messages = [
        SystemMessage(content="""You are a learning assessment expert.
Generate 3 quiz questions based on the topics and resources provided.
Respond ONLY with a JSON array of question strings. No explanation.
Example: ["Question 1?", "Question 2?", "Question 3?"]"""),
        HumanMessage(content=f"Topics: {state['topics']}\n\nResources:\n{context}")
    ]

    quiz_response = llm.invoke(quiz_messages)
    try:
        questions = json.loads(quiz_response.content)
        if not isinstance(questions, list):
            questions = ["What did you learn from this topic?"]
    except json.JSONDecodeError:
        questions = ["What did you learn from this topic?"]

    # Generate final learning path
    path_messages = [
        SystemMessage(content="""You are a learning coach.
Create a structured 1-week learning path based on the topics provided.
Be concise — max 150 words."""),
        HumanMessage(content=f"Goal: {state['user_goal']}\nTopics: {state['topics']}")
    ]

    path_response = llm.invoke(path_messages)

    return {
        "quiz_questions": questions,
        "final_path": path_response.content
    }