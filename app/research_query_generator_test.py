from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))


critic_issues = """
- The reported 40% performance gain and p-value refer to an LSTM model,
  not to Siamese-PTM.
- The F1 scores of 0.92 and 0.66 are attributed to LSTM, not Siamese-PTM.
- Direct evidence confirming Siamese-PTM's own performance metrics is missing.
"""

research_context = """
Research paper:
"Matching Patients to Clinical Trials using LLaMA 2 Embeddings
and Siamese Neural Network"

Research domain:
Clinical trial matching using machine learning and NLP.
"""

prompt = f"""
You are an autonomous research query generation agent.

Research context:
{research_context}

The critic identified these research gaps:

{critic_issues}

Generate ONE highly specific web search query that can find
the missing evidence.

The query MUST:
- Include the model name "Siamese-PTM".
- Include the clinical trial matching context.
- Focus on its F1 score or performance compared with Random Forest.
- Be specific enough to avoid unrelated machine-learning results.
- Be suitable for a web search engine.
- Return ONLY the search query.
- Do NOT answer the question.

Generate the query.
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

query = completion.choices[0].message.content.strip()

# Remove accidental duplicate query
half = len(query) // 2

if len(query) % 2 == 0 and query[:half] == query[half:]:
    query = query[:half].strip()


print("\n" + "=" * 60)
print("IMPROVED AUTOMATIC RESEARCH QUERY GENERATION")
print("=" * 60)

print("\nCRITIC ISSUES:")
print(critic_issues)

print("\nGENERATED RESEARCH QUERY:")
print(query)

print("\n" + "=" * 60)