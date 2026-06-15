# from deepeval.metrics import (
#     FaithfulnessMetric,
#     AnswerRelevancyMetric,
#     ContextualPrecisionMetric,
# )
# from deepeval.test_case import LLMTestCase
# from deepeval.models.base_model import DeepEvalBaseLLM
# from langchain_groq import ChatGroq
# import mlflow
# import time
# from app.core.config import settings


# class GroqEvaluationModel(DeepEvalBaseLLM):
#     def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
#         self.model_name = model_name
#         self.llm = ChatGroq(
#             api_key=settings.GROQ_API_KEY,
#             model=model_name,
#             temperature=0
#         )
#         super().__init__()

#     def load_model(self):
#         return self.llm

#     def generate(self, prompt: str) -> str:
#         response = self.llm.invoke(prompt)
#         return response.content

#     async def a_generate(self, prompt: str) -> str:
#         response = await self.llm.ainvoke(prompt)
#         return response.content

#     def get_model_name(self) -> str:
#         return f"groq/{self.model_name}"


# def _log_to_mlflow(query: str, scores: dict, model_name: str):
#     mlflow.set_experiment("skillforge_rag_evaluation")
#     with mlflow.start_run():
#         mlflow.log_param("query", query[:100])
#         mlflow.log_param("model", f"groq/{model_name}")
#         mlflow.log_param("passed", scores["passed"])
#         mlflow.log_metric("faithfulness", scores["faithfulness"])
#         mlflow.log_metric("answer_relevancy", scores["answer_relevancy"])
#         mlflow.log_metric("contextual_precision", scores["contextual_precision"])
#         mlflow.log_metric("evaluation_latency", scores["evaluation_latency_seconds"])


# def evaluate_rag_response(
#     query: str,
#     retrieved_contexts: list[str],
#     actual_output: str,
#     expected_output: str = None,
#     model_name: str = "llama-3.3-70b-versatile"
# ) -> dict:
#     eval_model = GroqEvaluationModel(model_name=model_name)

#     test_case = LLMTestCase(
#         input=query,
#         actual_output=actual_output,
#         retrieval_context=retrieved_contexts,
#         expected_output=expected_output or actual_output
#     )

#     faithfulness = FaithfulnessMetric(
#         threshold=0.7,
#         model=eval_model,
#         include_reason=True
#     )
#     relevancy = AnswerRelevancyMetric(
#         threshold=0.7,
#         model=eval_model,
#         include_reason=True
#     )
#     contextual_precision = ContextualPrecisionMetric(
#         threshold=0.7,
#         model=eval_model,
#         include_reason=True
#     )

#     start_time = time.time()
#     faithfulness.measure(test_case)
#     relevancy.measure(test_case)
#     contextual_precision.measure(test_case)
#     latency = round(time.time() - start_time, 3)

#     scores = {
#         "model_used": eval_model.get_model_name(),
#         "faithfulness": round(faithfulness.score, 3),
#         "faithfulness_reason": faithfulness.reason,
#         "answer_relevancy": round(relevancy.score, 3),
#         "answer_relevancy_reason": relevancy.reason,
#         "contextual_precision": round(contextual_precision.score, 3),
#         "contextual_precision_reason": contextual_precision.reason,
#         "evaluation_latency_seconds": latency,
#         "passed": all([
#             faithfulness.score >= 0.7,
#             relevancy.score >= 0.7,
#             contextual_precision.score >= 0.7
#         ])
#     }

#     _log_to_mlflow(query, scores, model_name)
#     return scores


from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric
)

from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import mlflow
import time
from app.core.config import settings

class UniversalEvaluationModel(DeepEvalBaseLLM):
    
    def __init__(self, model_name: str = "groq/llama-3.1-8b-instant"):
        self.model_name = model_name
        self.llm = self._load_llm(model_name)
        super().__init__()

    def _load_llm(self, model_name: str):
        if model_name.startswith("groq/"):
            actual_model = model_name.replace("groq/", "")
            return ChatGroq(
                api_key=settings.GROQ_API_KEY,
                model=actual_model,
                temperature=0
            )

        elif model_name.startswith("ollama/"):
            actual_model = model_name.replace("ollama/", "")
            return ChatOllama(
                model=actual_model,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0
            )

        elif model_name.startswith("hf-api/"):
            actual_model = model_name.replace("hf-api/", "")
            from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
            import os
            os.environ["HUGGINGFACEHIB_API_TOKEN"] = settings.HUGGINGFACE_API_KEY
            endpoint = HuggingFaceEndpoint(
                repo_id=actual_model,
                task="conversational",
                huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
                temperature=0.01,
                max_new_tokens=512
            )
            return ChatHuggingFace(llm=endpoint)

        elif model_name.startswith("hf-local/"):
            actual_model = model_name.replace("hf-local/", "")
            from langchain_huggingface import HuggingFacePipeline
            return HuggingFacePipeline.from_model_id(
                model_id=actual_model,
                task="text-generation",
                pipeline_kwargs={
                    "temperature": 0.01,
                    "max_new_tokens": 512,
                    "do_sample": True
                }
            )

        else:
            raise ValueError(
                f"Unknown provider '{model_name}'. "
                f"Use prefix: groq/, claude/, ollama/, hf-api/, hf-local/"
            )

    def load_model(self):
        return self.llm

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        if hasattr(response, "content"):
            return response.content
        return str(response)

    async def a_generate(self, prompt: str) -> str:
        response = await self.llm.ainvoke(prompt)
        if hasattr(response, "content"):
            return response.content
        return str(response)

    def get_model_name(self) -> str:
        return self.model_name


def _log_to_mlflow(query: str, scores: dict, model_name: str):
    """Logs evaluation results to MLflow for tracking and comparison"""
    mlflow.set_experiment("skillforge_rag_evaluation")
    with mlflow.start_run():
        mlflow.log_param("query", query[:100])
        mlflow.log_param("model", model_name)
        mlflow.log_param("passed", scores["passed"])
        mlflow.log_metric("faithfulness", scores["faithfulness"])
        mlflow.log_metric("answer_relevancy", scores["answer_relevancy"])
        mlflow.log_metric("contextual_precision", scores["contextual_precision"])
        mlflow.log_metric("evaluation_latency", scores["evaluation_latency_seconds"])


def evaluate_rag_response(
    query: str,
    retrieved_contexts: list[str],
    actual_output: str,
    expected_output: str = None,
    model_name: str = "groq/llama-3.1-8b-instant"
) -> dict:
    """
    Evaluates a RAG response using LLM-as-a-Judge via DeepEval.

    Supported model prefixes:
      groq/llama-3.1-8b-instant
      groq/llama-3.3-70b-versatile
      claude/claude-3-haiku-20240307
      ollama/llama3
      hf-api/mistralai/Mistral-7B-Instruct-v0.3
      hf-local/microsoft/Phi-3-mini-4k-instruct

    Logs all results to MLflow automatically.
    Returns scores dict with faithfulness, relevancy, precision.
    """
    eval_model = UniversalEvaluationModel(model_name=model_name)

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieved_contexts,
        expected_output=expected_output or actual_output
    )

    faithfulness = FaithfulnessMetric(
        threshold=0.7,
        model=eval_model,
        include_reason=True
    )
    relevancy = AnswerRelevancyMetric(
        threshold=0.7,
        model=eval_model,
        include_reason=True
    )
    contextual_precision = ContextualPrecisionMetric(
        threshold=0.7,
        model=eval_model,
        include_reason=True
    )

    start_time = time.time()
    faithfulness.measure(test_case)
    relevancy.measure(test_case)
    contextual_precision.measure(test_case)
    latency = round(time.time() - start_time, 3)

    scores = {
        "model_used": eval_model.get_model_name(),
        "faithfulness": round(faithfulness.score, 3),
        "faithfulness_reason": faithfulness.reason,
        "answer_relevancy": round(relevancy.score, 3),
        "answer_relevancy_reason": relevancy.reason,
        "contextual_precision": round(contextual_precision.score, 3),
        "contextual_precision_reason": contextual_precision.reason,
        "evaluation_latency_seconds": latency,
        "passed": all([
            faithfulness.score >= 0.7,
            relevancy.score >= 0.7,
            contextual_precision.score >= 0.7
        ])
    }

    _log_to_mlflow(query, scores, model_name)
    return scores