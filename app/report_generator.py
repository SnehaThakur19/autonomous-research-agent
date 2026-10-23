import os

from dotenv import load_dotenv
from groq import Groq

from app.prompts.prompts import REPORT_PROMPT


load_dotenv()


def generate_report(
    query,
    claims,
    verification_results,
    critic_result,
    evidence,
    web_results=None,
    citation_report=None,
    subtasks=None
):
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

    external_text = ""

    if web_results:
        for i, result in enumerate(web_results, start=1):
            external_text += f"""
External Source {i}:
Title: {result.get("title")}
URL: {result.get("url")}
Content:
{result.get("content")}

"""

    if citation_report:
        citation_text = citation_report.get("text", "")
    else:
        citation_text = "No citation validation was performed."

    if subtasks:
        subtasks_text = "\n".join(
            f"{i}. {subtask}"
            for i, subtask in enumerate(subtasks, start=1)
        )
    else:
        subtasks_text = "Not available."

    prompt = REPORT_PROMPT.format(
        query=query,
        subtasks=subtasks_text,
        claims=claims,
        verification_results=verification_results,
        critic_result=critic_result,
        citation_report=citation_text,
        evidence_text=evidence_text,
        external_text=external_text
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
