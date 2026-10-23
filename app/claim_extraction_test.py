from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

text = """
Retrieval-Augmented Generation (RAG) combines information retrieval
with text generation. It allows a language model to use external
information when answering questions. RAG is useful when information
needs to be current or domain-specific, and it can help reduce
hallucinations by providing relevant evidence to the model.
"""

prompt = f"""
You are a claim extraction system.

Extract the important factual claims from the text below.

Rules:
- Extract only factual claims.
- Do not add new information.
- Keep each claim short and clear.
- Number each claim.

Text:
{text}

Return only the claims.
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
print("EXTRACTED CLAIMS")
print("=" * 60)

print(completion.choices[0].message.content)