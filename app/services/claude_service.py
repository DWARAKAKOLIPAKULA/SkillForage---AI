import anthropic
from app.core.config import settings
from app.services.retrieval import search_similar

client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

# This is our system prompt — it's long and never changes
# Perfect candidate for caching
SYSTEM_PROMPT = """You are SkillForge AI, an expert learning path generator.
You help users learn technical topics by:
1. Analyzing their learning goal
2. Using the provided context from our knowledge base
3. Generating structured, actionable learning recommendations

Always base your answers on the provided context.
If the context doesn't cover the topic, say so clearly.
Never hallucinate resources or URLs.

When generating a learning path, always respond in this exact JSON format:
{
    "summary": "brief overview of the learning path",
    "difficulty": "beginner/intermediate/advanced",
    "estimated_hours": number,
    "topics": [
        {
            "name": "topic name",
            "description": "what this covers",
            "resources": ["resource 1", "resource 2"]
        }
    ],
    "next_steps": "what to do after completing this path"
}"""


def generate_learning_path_with_cache(
    user_goal: str,
    top_k: int = 3
) -> dict:
    """
    Generates a structured learning path using Claude with prompt caching.
    Returns token usage showing cache hits vs misses.
    """
    # Step 1 — retrieve context from FAISS
    retrieved = search_similar(user_goal, top_k=top_k)
    context = "\n\n".join([
        f"Source: {r['source']}\n{r['content']}"
        for r in retrieved
    ]) if retrieved else "No relevant context found in knowledge base."

    # Step 2 — call Claude with caching
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",   # fast and cheap
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}  # ← cache system prompt
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Knowledge base context:\n{context}",
                        "cache_control": {"type": "ephemeral"}  # ← cache context
                    },
                    {
                        "type": "text",
                        "text": f"Generate a learning path for: {user_goal}"
                        # ← no cache — changes every request
                    }
                ]
            }
        ]
    )

    # Step 3 — extract token usage (shows cache performance)
    usage = response.usage
    token_report = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_tokens": getattr(usage, "cache_creation_input_tokens", 0),
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0),
    }

    # Step 4 — calculate cost savings
    cache_read = token_report["cache_read_tokens"]
    cache_creation = token_report["cache_creation_tokens"]
    token_report["cache_hit"] = cache_read > 0
    token_report["tokens_saved"] = cache_read  # these were free (90% discount)

    # Step 5 — parse Claude's JSON response
    raw_text = response.content[0].text
    try:
        import json
        # Find JSON in response
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        learning_path = json.loads(raw_text[start:end])
    except (json.JSONDecodeError, ValueError):
        learning_path = {"raw_response": raw_text}

    return {
        "learning_path": learning_path,
        "token_usage": token_report,
        "model": "claude-3-5-haiku-20241022"
    }


def answer_with_cache(query: str, top_k: int = 3) -> dict:
    """
    Answers a question using Claude with RAG context and prompt caching.
    Simpler version — returns plain text answer with token usage.
    """
    retrieved = search_similar(query, top_k=top_k)
    context = "\n\n".join([
        f"Source: {r['source']}\n{r['content']}"
        for r in retrieved
    ]) if retrieved else "No relevant context found."

    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Context:\n{context}",
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "type": "text",
                        "text": query
                    }
                ]
            }
        ]
    )

    usage = response.usage
    return {
        "answer": response.content[0].text,
        "token_usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation_tokens": getattr(usage, "cache_creation_input_tokens", 0),
            "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0),
            "cache_hit": getattr(usage, "cache_read_input_tokens", 0) > 0
        },
        "model": "claude-3-5-haiku-20241022"
    }