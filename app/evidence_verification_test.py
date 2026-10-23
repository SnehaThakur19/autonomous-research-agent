from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

claim = """
RAG combines information retrieval with text generation.
"""

evidence = """
Retrieval-Augmented Generation (RAG) is a technique that
combines information retrieval with text generation.
"""

prompt = f"""
You are an evidence verification system.

Your task is to determine whether the evidence supports the claim.

CLAIM:
{claim}

EVIDENCE:
{evidence}

Classify the claim into exactly one of these categories:

SUPPORTED
PARTIALLY SUPPORTED
NOT SUPPORTED

Rules:
- SUPPORTED: The evidence directly supports the claim.
- PARTIALLY SUPPORTED: The evidence supports only part of the claim.
- NOT SUPPORTED: The evidence does not support the claim.

Return the result in this format:

VERDICT: <category>
REASON: <short explanation>
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
print("EVIDENCE VERIFICATION")
print("=" * 60)

print("\nCLAIM:")
print(claim)

print("\nEVIDENCE:")
print(evidence)

print("\nVERIFICATION RESULT:")
print(completion.choices[0].message.content)