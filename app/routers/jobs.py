from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
import redis
import json
from app.celery_app import celery_app
from app.tasks import generate_learning_path_task, web_research_task, evaluate_rag_task
from app.models.user import User
from app.core.dependencies import get_current_user
from app.core.config import settings

router = APIRouter(prefix="/jobs", tags=["Background Jobs"])
redis_client = redis.from_url(settings.REDIS_URL)


class LearningPathJobRequest(BaseModel):
    goal: str


class ResearchJobRequest(BaseModel):
    topics: list[str]


class EvaluationJobRequest(BaseModel):
    query: str
    model_name: str = "groq/llama-3.1-8b-instant"


@router.post("/learning-path")
async def submit_learning_path_job(
    request: LearningPathJobRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Submits learning path generation as background job.
    Returns job_id immediately — poll /jobs/status/{job_id} for results.
    """
    task = generate_learning_path_task.delay(
        user_goal=request.goal,
        user_id=current_user.id
    )
    return {
        "job_id": task.id,
        "status": "queued",
        "message": "Learning path generation started",
        "poll_url": f"/jobs/status/{task.id}"
    }


@router.post("/research")
async def submit_research_job(
    request: ResearchJobRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Submits web research as background job.
    Returns job_id immediately.
    """
    task = web_research_task.delay(
        topics=request.topics,
        user_id=current_user.id
    )
    return {
        "job_id": task.id,
        "status": "queued",
        "message": "Web research started",
        "poll_url": f"/jobs/status/{task.id}"
    }


@router.post("/evaluate")
async def submit_evaluation_job(
    request: EvaluationJobRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Submits RAG evaluation as background job.
    Returns job_id immediately.
    """
    task = evaluate_rag_task.delay(
        query=request.query,
        model_name=request.model_name,
        user_id=current_user.id
    )
    return {
        "job_id": task.id,
        "status": "queued",
        "message": "Evaluation started",
        "poll_url": f"/jobs/status/{task.id}"
    }


@router.get("/status/{job_id}")
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Returns current status of a background job.
    Poll this every 2 seconds until status is 'complete' or 'failed'.
    """
    # Check our custom Redis status first
    raw = redis_client.get(f"job:{job_id}")
    if raw:
        return json.loads(raw)

    # Fallback — check Celery's own status
    task = celery_app.AsyncResult(job_id)
    return {
        "job_id": job_id,
        "status": task.status.lower(),
        "progress": 0,
        "message": "Job is queued, waiting for worker...",
        "result": None
    }


@router.get("/all")
async def get_all_jobs(
    current_user: User = Depends(get_current_user)
):
    """Returns all active jobs in Redis"""
    keys = redis_client.keys("job:*")
    jobs = []
    for key in keys:
        raw = redis_client.get(key)
        if raw:
            jobs.append(json.loads(raw))
    return {"total": len(jobs), "jobs": jobs}