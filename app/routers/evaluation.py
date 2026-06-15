# from fastapi import APIRouter, Depends
# from pydantic import BaseModel
# from app.evaluation.evaluator import evaluate_rag_response
# from app.services.retrieval import search_similar, search_and_answer
# from app.models.user import User
# from app.core.dependencies import get_current_user

# router = APIRouter(prefix="/evaluate", tags=["Evaluation"])

# class EvaluationRequest(BaseModel):
#     query: str
#     expected_output: str = None
#     model_name: str = "llama-3.3-70b-versatile"

# @router.post("/rag")
# async def evaluate_rag(
#     request: EvaluationRequest,
#     current_user: User = Depends(get_current_user)
# ):
#     """
#     Runs a RAG query, gets the answer, then evaluates it with DeepEval.
#     Logs results to MLflow.
#     """
#     # Step 1 — Get retrieved contexts
#     retrieved = search_similar(request.query, top_k=3)
#     contexts = [r["content"] for r in retrieved]

#     # Step 2 — Get LLM answer
#     rag_result = search_and_answer(request.query)
#     answer = rag_result["answer"]

#     # Step 3 — Evaluate
#     scores = evaluate_rag_response(
#         query=request.query,
#         retrieved_contexts=contexts,
#         actual_output=answer,
#         expected_output=request.expected_output,
#         model_name=request.model_name
#     )

#     return {
#         "query": request.query,
#         "answer": answer,
#         "evaluation": scores
#     }


from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.evaluation.evaluator import evaluate_rag_response
from app.services.retrieval import search_similar, search_and_answer
from app.models.user import User
from app.core.dependencies import get_current_user

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])

SUPPORTED_PROVIDERS = ["groq/", "ollama/", "hf-api/", "hf-local/"]


class EvaluationRequest(BaseModel):
    query: str
    expected_output: Optional[str] = None
    model_name: str = "groq/llama-3.1-8b-instant"


class JudgeQualityRequest(BaseModel):
    model_name: str = "groq/llama-3.1-8b-instant"


@router.post("/rag")
async def evaluate_rag(
    request: EvaluationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Runs a RAG query, generates answer, evaluates with DeepEval LLM-as-a-Judge.
    Logs faithfulness, answer relevancy, contextual precision to MLflow.

    model_name accepts:
      groq/llama-3.1-8b-instant
      groq/llama-3.3-70b-versatile
      ollama/llama3
      hf-api/mistralai/Mistral-7B-Instruct-v0.3
      hf-local/microsoft/Phi-3-mini-4k-instruct
    """
    # Validate provider prefix
    if not any(request.model_name.startswith(p) for p in SUPPORTED_PROVIDERS):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model prefix. Supported: {SUPPORTED_PROVIDERS}"
        )

    # Step 1 — retrieve context from FAISS
    retrieved = search_similar(request.query, top_k=3)
    contexts = [r["content"] for r in retrieved]

    if not contexts:
        raise HTTPException(
            status_code=404,
            detail="No documents in knowledge base. Ingest documents first via POST /rag/ingest"
        )

    # Step 2 — generate answer using RAG
    rag_result = search_and_answer(request.query)
    answer = rag_result["answer"]

    # Step 3 — evaluate with LLM-as-a-Judge
    scores = evaluate_rag_response(
        query=request.query,
        retrieved_contexts=contexts,
        actual_output=answer,
        expected_output=request.expected_output,
        model_name=request.model_name
    )

    return {
        "query": request.query,
        "answer": answer,
        "retrieved_chunks": len(contexts),
        "evaluation": scores
    }


@router.post("/judge-quality")
async def test_judge_quality(
    request: JudgeQualityRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Tests if the judge model is reliable by running known good and bad answers.
    A reliable judge scores correct answers high and wrong answers low.
    Use this to validate any judge model before using it in production.
    """
    if not any(request.model_name.startswith(p) for p in SUPPORTED_PROVIDERS):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model prefix. Supported: {SUPPORTED_PROVIDERS}"
        )

    context = ["FastAPI is a modern async Python framework for building APIs with automatic validation via Pydantic"]

    test_cases = [
        {
            "label": "correct_answer",
            "answer": "FastAPI is a modern Python framework for building APIs with automatic validation",
            "expected_faithfulness": "high"
        },
        {
            "label": "hallucinated_answer",
            "answer": "FastAPI was created in 2010 by Guido van Rossum and is used for mobile apps",
            "expected_faithfulness": "low"
        },
        {
            "label": "irrelevant_answer",
            "answer": "Python is a great language for data science and machine learning projects",
            "expected_faithfulness": "low"
        }
    ]

    results = []
    for case in test_cases:
        scores = evaluate_rag_response(
            query="what is FastAPI",
            retrieved_contexts=context,
            actual_output=case["answer"],
            model_name=request.model_name
        )
        is_high = case["expected_faithfulness"] == "high"
        judge_correct = (
            scores["faithfulness"] >= 0.7 if is_high
            else scores["faithfulness"] < 0.7
        )
        results.append({
            "label": case["label"],
            "expected_faithfulness": case["expected_faithfulness"],
            "actual_faithfulness_score": scores["faithfulness"],
            "judge_correct": judge_correct,
            "reason": scores["faithfulness_reason"]
        })

    all_correct = all(r["judge_correct"] for r in results)

    return {
        "judge_model": request.model_name,
        "reliable": all_correct,
        "summary": "Judge is reliable ✅" if all_correct else "Judge is unreliable ❌ — try a stronger model",
        "test_results": results
    }


@router.get("/supported-models")
async def get_supported_models(
    current_user: User = Depends(get_current_user)
):
    """Returns all supported model providers and example model names"""
    return {
        "providers": {
            "groq": {
                "prefix": "groq/",
                "examples": [
                    "groq/llama-3.1-8b-instant",
                    "groq/llama-3.3-70b-versatile",
                    "groq/llama3-8b-8192"
                ],
                "cost": "free tier"
            },
            "ollama": {
                "prefix": "ollama/",
                "examples": [
                    "ollama/llama3",
                    "ollama/mistral",
                    "ollama/phi3"
                ],
                "cost": "free, runs locally"
            },
            "huggingface_api": {
                "prefix": "hf-api/",
                "examples": [
                    "hf-api/mistralai/Mistral-7B-Instruct-v0.3",
                    "hf-api/microsoft/Phi-3-mini-4k-instruct",
                    "hf-api/google/gemma-2-2b-it",
                    "hf-api/HuggingFaceH4/zephyr-7b-beta"
                ],
                "cost": "free tier"
            },
            "huggingface_local": {
                "prefix": "hf-local/",
                "examples": [
                    "hf-local/microsoft/Phi-3-mini-4k-instruct",
                    "hf-local/google/gemma-2-2b-it",
                    "hf-local/TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                ],
                "cost": "free, runs on your machine, needs 4-8GB RAM"
            }
        }
    }