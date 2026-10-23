import os

from dotenv import load_dotenv
from groq import Groq

from app.prompts.prompts import VERIFICATION_PROMPT


load_dotenv()


def verify_claims(claims, evidence):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")

    client = Groq(api_key=api_key)

    evidence_text = ""

    for i, item in enumerate(evidence, start=1):
        evidence_text += f"""
Evidence {i}:
Source: {item["metadata"].get("source")}
Page: {item["metadata"].get("page")}
Chunk ID: {item["metadata"].get("chunk_id")}
Text:
{item["text"]}

"""

    prompt = VERIFICATION_PROMPT.format(
        claims=claims,
        evidence_text=evidence_text
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
