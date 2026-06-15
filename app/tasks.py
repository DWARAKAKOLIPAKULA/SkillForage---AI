from app.celery_app import celery_app
from celery import current_task
import redis
import json
import asyncio
from app.core.config import settings

redis_client = redis.from_url(settings.REDIS_URL)


def _update_status(job_id: str, status: str, progress: int, message: str, result: dict = None):
    """Helper — stores job status in Redis"""
    data = {
        "job_id": job_id,
        "status": status,
        "progress": progress,
        "message": message,
        "result": result
    }
    redis_client.setex(
        f"job:{job_id}",
        3600,
        json.dumps(data)
    )


@celery_app.task(bind=True, name="generate_learning_path")
def generate_learning_path_task(self, user_goal: str, user_id: str):
    """
    Background task — runs LangGraph agents synchronously.
    Each agent is called directly since Celery is sync.
    """
    job_id = self.request.id

    try:
        # Step 1 — Planner agent
        _update_status(job_id, "planning", 10, "Planner agent analyzing your goal...")

        from app.agents.planner import planner_agent
        initial_state = {
            "user_goal": user_goal,
            "topics": [],
            "resources": [],
            "quiz_questions": [],
            "final_path": "",
            "messages": []
        }
        planner_result = planner_agent(initial_state)
        topics = planner_result.get("topics", [])

        # Step 2 — Resource agent (uses RAG search — sync, no issue)
        _update_status(job_id, "finding_resources", 40, f"Found {len(topics)} topics. Searching knowledge base...")

        from app.services.retrieval import search_similar
        all_resources = []
        for topic in topics:
            results = search_similar(query=topic, top_k=2)
            for r in results:
                all_resources.append({
                    "topic": topic,
                    "content": r["content"],
                    "source": r["source"],
                    "relevance_score": r["score"],
                    "from": "rag"
                })

        # Step 3 — Quiz agent
        _update_status(job_id, "generating_quiz", 75, "Generating quiz questions and learning path...")

        from app.agents.quiz import quiz_agent
        quiz_state = {
            "user_goal": user_goal,
            "topics": topics,
            "resources": all_resources,
            "quiz_questions": [],
            "final_path": "",
            "messages": []
        }
        quiz_result = quiz_agent(quiz_state)

        # Step 4 — Store final result
        final_result = {
            "goal": user_goal,
            "topics": topics,
            "resources": all_resources,
            "quiz_questions": quiz_result.get("quiz_questions", []),
            "learning_path": quiz_result.get("final_path", "")
        }

        _update_status(job_id, "complete", 100, "Learning path ready!", result=final_result)
        return final_result

    except Exception as e:
        _update_status(job_id, "failed", 0, f"Error: {str(e)}")
        raise


@celery_app.task(bind=True, name="web_research_task")
def web_research_task(self, topics: list, user_id: str):
    """
    Background task — runs CrewAI web research.
    Uses asyncio.run() to handle async CrewAI function.
    """
    job_id = self.request.id

    try:
        _update_status(job_id, "researching", 20, f"Searching web for {len(topics)} topics...")

        from app.agents.research_crew import create_research_crew

        # asyncio.run() creates a new event loop in this thread — safe in Celery
        result = asyncio.run(create_research_crew(topics))

        _update_status(job_id, "complete", 100, "Research complete!", result={"research": result})
        return {"research": result}

    except Exception as e:
        _update_status(job_id, "failed", 0, f"Error: {str(e)}")
        raise


@celery_app.task(bind=True, name="evaluate_rag_task")
def evaluate_rag_task(self, query: str, model_name: str, user_id: str):
    """
    Background task — runs DeepEval evaluation.
    """
    job_id = self.request.id

    try:
        _update_status(job_id, "evaluating", 20, "Retrieving context from knowledge base...")

        from app.services.retrieval import search_similar, search_and_answer
        retrieved = search_similar(query, top_k=3)
        contexts = [r["content"] for r in retrieved]
        rag_result = search_and_answer(query)
        answer = rag_result["answer"]

        _update_status(job_id, "scoring", 60, "Running LLM-as-a-Judge evaluation...")

        from app.evaluation.evaluator import evaluate_rag_response
        scores = evaluate_rag_response(
            query=query,
            retrieved_contexts=contexts,
            actual_output=answer,
            model_name=model_name
        )

        final_result = {
            "query": query,
            "answer": answer,
            "evaluation": scores
        }
        _update_status(job_id, "complete", 100, "Evaluation complete!", result=final_result)
        return final_result

    except Exception as e:
        _update_status(job_id, "failed", 0, f"Error: {str(e)}")
        raise