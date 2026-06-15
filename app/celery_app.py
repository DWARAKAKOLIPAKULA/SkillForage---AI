from celery import Celery
from app.core.config import settings

# Create Celery instance
# broker = Redis queue (where tasks are sent)
# backend = Redis result store (where results are saved)
celery_app = Celery(
    "skillforge",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.tasks"]      # tells Celery where to find tasks
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,       # results expire after 1 hour
    task_track_started=True,   # shows "started" status, not just "pending"
    worker_prefetch_multiplier=1,  # one task at a time per worker
)