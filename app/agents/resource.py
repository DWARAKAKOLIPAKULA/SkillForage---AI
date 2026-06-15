from app.agents.state import SkillForgeState
from app.services.retrieval import search_similar
from app.agents.research_crew import create_research_crew
import os

def resource_agent(state: SkillForgeState) -> dict:
    """
    First tries RAG (FAISS) for each topic.
    If RAG results are insufficient, falls back to CrewAI web research.
    """
    all_resources = []
    topics_needing_research = []

    # Step 1 — Try RAG first
    for topic in state['topics']:
        results = search_similar(query=topic, top_k=2)

        if results and results[0]["score"] < 0.8:
            # Good RAG results found (low score = high similarity in FAISS)
            for r in results:
                all_resources.append({
                    "topic": topic,
                    "content": r["content"],
                    "source": r["source"],
                    "relevance_score": r["score"],
                    "from": "rag"
                })
        else:
            # RAG results are weak — mark for web research
            topics_needing_research.append(topic)

    # Step 2 — Use CrewAI for topics RAG couldn't answer well
    if topics_needing_research:
        web_results = create_research_crew(topics_needing_research)
        all_resources.append({
            "topic": "web_research",
            "content": web_results,
            "source": "crewai_web_search",
            "relevance_score": 0.0,
            "from": "crewai"
        })

    return {"resources": all_resources}