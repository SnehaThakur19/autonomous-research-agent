from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

verification_results = """
CLAIM 1:
RAG combines information retrieval with text generation.
VERDICT: SUPPORTED
REASON: Evidence directly supports the claim.

CLAIM 2:
RAG allows language models to use external information.
VERDICT: SUPPORTED
REASON: Evidence directly supports the claim.

CLAIM 3:
RAG is useful for current or domain-specific information.
VERDICT: SUPPORTED
REASON: Evidence directly supports the claim.

CLAIM 4:
RAG completely eliminates hallucinations.
VERDICT: PARTIALLY SUPPORTED
REASON: Evidence says RAG can help reduce hallucinations,
but does not say it completely eliminates them.
"""

prompt = f"""
You are a research critic agent.

Review the following evidence verification results.

Your job is to identify problems in the research.

Look for:
- Unsupported claims
- Partially supported claims
- Claims stronger than their evidence
- Missing or insufficient evidence
- Whether additional research is needed

Verification Results:
{verification_results}

Return the result in this format:

OVERALL STATUS: <GOOD / NEEDS_RESEARCH>

ISSUES:
- <issue 1>
- <issue 2>

RECOMMENDATION:
<short recommendation>
"""

completion = groq.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
)

print("\n" + "=" * 60)
print("CRITIC AGENT")
print("=" * 60)

print(completion.choices[0].message.content)