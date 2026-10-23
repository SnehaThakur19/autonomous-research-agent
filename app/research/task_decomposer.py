import os
import re

from dotenv import load_dotenv
from groq import Groq

from app.prompts.prompts import QUERY_DECOMPOSITION_PROMPT


load_dotenv()


def decompose_query(query, max_subtasks=3):
    """Decompose a complex research question into focused subtasks.

    Returns a list of subtask strings. Falls back to the original
    query if the LLM output cannot be parsed.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")

    client = Groq(api_key=api_key)

    prompt = QUERY_DECOMPOSITION_PROMPT.format(
        query=query,
        max_subtasks=max_subtasks
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    subtasks = []

    # Parse a numbered list: "1. subtask text"
    for line in content.splitlines():
        match = re.match(r"^\s*\d+[\.\)]\s+(.+?)\s*$", line)

        if match:
            subtasks.append(match.group(1))

    # Keep only the first max_subtasks
    subtasks = subtasks[:max_subtasks]

    # Fallback: use the original query if parsing failed
    if not subtasks:
        return [query]

    return subtasks
