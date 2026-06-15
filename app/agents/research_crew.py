from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from app.core.config import settings
import os
import asyncio
from functools import partial
import litellm
import httpx
import re

os.environ["SERPER_API_KEY"] = settings.SERPER_API_KEY
os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

_original_completion = litellm.completion

def _patched_completion(*args, **kwargs):
    messages = kwargs.get("messages", [])
    for msg in messages:
        if isinstance(msg, dict):
            msg.pop("cache_breakpoint", None)
    kwargs["messages"] = messages
    return _original_completion(*args, **kwargs)

litellm.completion = _patched_completion


def validate_urls_in_result(result: str) -> str:
    """Finds all URLs, validates them, marks broken ones"""
    url_pattern = r'https?://[^\s\)\]>"]+'
    urls = re.findall(url_pattern, result)

    for url in urls:
        try:
            response = httpx.get(url, timeout=5, follow_redirects=True)
            if response.status_code != 200:
                result = result.replace(url, f"{url} ⚠️UNVERIFIED")
        except Exception:
            result = result.replace(url, f"{url} ⚠️UNVERIFIED")

    return result          # ← Bug 2 fixed: must return result


def _run_crew(topics: list[str]) -> str:
    search_tool = SerperDevTool()

    researcher = Agent(
        role="Senior Learning Resource Researcher",
        goal="Find the best and most current free learning resources for technical topics",
        backstory="""You are an expert at finding high quality learning resources online.
You know where to look — official docs, YouTube, freeCodeCamp, dev.to, GitHub repos.
You always find resources that are practical, beginner friendly, and free.""",
        tools=[search_tool],
        verbose=True,
        allow_delegation=False,
        llm="groq/openai/gpt-oss-20b"
    )

    curator = Agent(
        role="Learning Path Curator",
        goal="Organize research findings into clean structured resource lists",
        backstory="""You take raw research and turn it into clean organized lists.
You group by topic, sort by difficulty, remove duplicates,
and add brief descriptions so learners know exactly what each resource covers.""",
        tools=[],
        verbose=True,
        allow_delegation=False,
        llm="groq/llama-3.3-70b-versatile"
    )

    research_task = Task(
        description=f"""Search the web and find 2-3 high quality FREE learning resources
for each of these topics: {topics}

CRITICAL RULES:
- Copy URLs EXACTLY as they appear in search results — never modify or guess URLs
- If you are not 100% certain of a URL, do not include it
- Only include resources where you found the actual URL in search results

For each resource include:
- Title (exact title from search result)
- URL (copied exactly from search result — no guessing)
- Brief description (1 sentence)
- Difficulty level (beginner/intermediate/advanced)""",
        expected_output="A list of verified resources with exact URLs from search results",
        agent=researcher
    )

    curation_task = Task(
        description="""Take the research findings and organize them into a clean
structured resource list. Group by topic, sort by difficulty within each topic,
and write a one line summary for each resource.""",
        expected_output="A clean organized resource list grouped by topic",
        agent=curator,
        context=[research_task]
    )

    crew = Crew(
        agents=[researcher, curator],
        tasks=[research_task, curation_task],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()                        # ← Bug 1&3 fixed: inside _run_crew
    validated = validate_urls_in_result(str(result))
    return validated


async def create_research_crew(topics: list[str]) -> str:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        partial(_run_crew, topics)
    )
    return result