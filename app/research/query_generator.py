import os

from dotenv import load_dotenv
from groq import Groq

from app.prompts.prompts import QUERY_REWRITE_PROMPT


load_dotenv()


def generate_research_query(
    claim,
    critic_feedback,
    search_type="web"
):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")

    client = Groq(api_key=api_key)

    prompt = QUERY_REWRITE_PROMPT.format(
        claim=claim,
        critic_feedback=critic_feedback,
        search_type=search_type
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

    return response.choices[0].message.content.strip()
