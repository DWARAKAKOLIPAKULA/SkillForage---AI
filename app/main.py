from fastapi import FastAPI
from app.database import engine, Base
from app.routers import auth, goals, rag, learning, research, evaluation, claude, jobs

app = FastAPI(title="SkillForge AI")

@app.on_event("startup")
async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

app.include_router(auth.router)
app.include_router(goals.router)
app.include_router(rag.router)
app.include_router(learning.router)
app.include_router(research.router)
app.include_router(evaluation.router)
app.include_router(claude.router)
app.include_router(jobs.router)

@app.get("/")
async def root():
    return {"message": "SkillForge AI is running"}