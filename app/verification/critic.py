import os

from dotenv import load_dotenv
from groq import Groq

from app.prompts.prompts import CRITIC_PROMPT


load_dotenv()


def critique_research(
    query,
    claims,
    verification_results,
    web_verification_results=""
):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")

    client = Groq(api_key=api_key)

    prompt = CRITIC_PROMPT.format(
        query=query,
        claims=claims,
        verification_results=verification_results,
        web_verification_results=web_verification_results
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
