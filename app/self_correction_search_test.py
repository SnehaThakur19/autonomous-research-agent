from groq import Groq
from tavily import TavilyClient
from dotenv import load_dotenv
import os

load_dotenv()

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# Critic output
critic_issues = """
- The reported 40% performance gain and p-value refer to an LSTM model,
  not to Siamese-PTM.
- The F1 scores of 0.92 and 0.66 are attributed to LSTM, not Siamese-PTM.
- Direct evidence confirming Siamese-PTM's own performance metrics is missing.
"""


# Research context
research_context = """
Research paper:
"Matching Patients to Clinical Trials using LLaMA 2 Embeddings
and Siamese Neural Network"

Research domain:
Clinical trial matching using machine learning and NLP.
"""


# --------------------------------------------------
# STEP 1: Generate targeted research query
# --------------------------------------------------

prompt = f"""
You are an autonomous research query generation agent.

Research context:
{research_context}

Critic issues:
{critic_issues}

Generate ONE highly specific web search query that can find
the missing evidence.

The query MUST:
- Include Siamese-PTM.
- Include the clinical trial matching context.
- Focus on F1 score or performance compared with Random Forest.
- Be specific enough to avoid unrelated machine-learning results.
- Return ONLY the search query.
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

research_query = completion.choices[0].message.content.strip()


# Remove accidental duplicate query
half = len(research_query) // 2

if (
    len(research_query) % 2 == 0
    and research_query[:half] == research_query[half:]
):
    research_query = research_query[:half].strip()


# --------------------------------------------------
# STEP 2: Search the web using generated query
# --------------------------------------------------

response = tavily.search(
    query=research_query,
    max_results=5
)


# --------------------------------------------------
# STEP 3: Display results
# --------------------------------------------------

print("\n" + "=" * 60)
print("AUTONOMOUS SELF-CORRECTION")
print("=" * 60)

print("\nCRITIC IDENTIFIED:")
print(critic_issues)

print("\nGENERATED RESEARCH QUERY:")
print(research_query)

print("\n" + "=" * 60)
print("NEW WEB EVIDENCE")
print("=" * 60)


for i, result in enumerate(response["results"], start=1):

    print(f"\nEVIDENCE {i}")
    print("-" * 60)

    print("Title:", result["title"])
    print("URL:", result["url"])
    print("Content:", result["content"])


print("\n" + "=" * 60)