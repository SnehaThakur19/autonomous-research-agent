import os

from dotenv import load_dotenv
from groq import Groq

from app.prompts.prompts import EXTERNAL_VERIFICATION_PROMPT


load_dotenv()


def verify_claims_with_web(claims, web_results):
    """Re-verify claims against external evidence.

    Works for any external source shaped like
    {title, url, content} — web search results or
    academic paper search results.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")

    client = Groq(api_key=api_key)

    external_evidence = ""

    for i, result in enumerate(web_results, start=1):

        external_evidence += f"""
External Evidence {i}:

Title:
{result.get("title")}

URL:
{result.get("url")}

Content:
{result.get("content")}

"""

    prompt = EXTERNAL_VERIFICATION_PROMPT.format(
        claims=claims,
        external_evidence=external_evidence
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
