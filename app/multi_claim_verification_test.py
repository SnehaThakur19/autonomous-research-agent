from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

claims = [
    "RAG combines information retrieval with text generation.",
    "RAG allows language models to use external information.",
    "RAG is useful for current or domain-specific information.",
    "RAG completely eliminates hallucinations."
]

evidence = """
RAG is a technique that combines information retrieval with text
generation. It allows a language model to use external information
when answering questions. RAG is useful when information needs to
be current or domain-specific. It can help reduce hallucinations
by providing relevant evidence to the model.
"""

prompt = f"""
You are an evidence verification system.

Verify each claim using ONLY the provided evidence.

CLAIMS:
{claims}

EVIDENCE:
{evidence}

For each claim, classify it as exactly one of:

SUPPORTED
PARTIALLY SUPPORTED
NOT SUPPORTED

Rules:
- SUPPORTED: Evidence directly supports the complete claim.
- PARTIALLY SUPPORTED: Evidence supports only part of the claim.
- NOT SUPPORTED: Evidence does not support the claim.
- Do not add information that is not present in the evidence.

Return the result in this format:

CLAIM 1:
VERDICT: <category>
REASON: <short explanation>

CLAIM 2:
VERDICT: <category>
REASON: <short explanation>

Continue for all claims.
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
print("MULTI-CLAIM EVIDENCE VERIFICATION")
print("=" * 60)

print(completion.choices[0].message.content)