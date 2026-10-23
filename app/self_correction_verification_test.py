from groq import Groq
from tavily import TavilyClient
from dotenv import load_dotenv
import os

load_dotenv()

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# --------------------------------------------------
# ORIGINAL CLAIMS
# --------------------------------------------------

claims = [
    {
        "claim_id": 1,
        "claim": "Siamese-PTM achieved a statistically significant 40% performance gain over the Random Forest baseline (p-value = 0.006)."
    },
    {
        "claim_id": 2,
        "claim": "Siamese-PTM attained an F1 score of 0.92, compared with the Random Forest baseline's F1 score of 0.66."
    },
    {
        "claim_id": 3,
        "claim": "Siamese-PTM outperformed seven out of ten counterpart single-encoder models."
    }
]


# --------------------------------------------------
# CRITIC ISSUES FROM PREVIOUS ITERATION
# --------------------------------------------------

critic_issues = """
- The reported 40% performance gain and p-value refer to an LSTM model,
  not to Siamese-PTM.
- The F1 scores of 0.92 and 0.66 are attributed to LSTM, not Siamese-PTM.
- Direct evidence confirming Siamese-PTM's own performance metrics is missing.
"""


# --------------------------------------------------
# STEP 1: GENERATE TARGETED QUERY
# --------------------------------------------------

query_prompt = f"""
You are an autonomous research query generation agent.

The research topic is clinical trial matching using Siamese-PTM.

The critic identified these issues:

{critic_issues}

Generate ONE highly specific web search query to find the missing evidence.

The query must:
- Include Siamese-PTM.
- Include clinical trial matching.
- Focus on F1 score or performance compared with Random Forest.
- Be suitable for a web search engine.
- Return ONLY the search query.
"""

query_completion = groq.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": query_prompt
        }
    ],
)

research_query = query_completion.choices[0].message.content.strip()


# --------------------------------------------------
# STEP 2: SEARCH WEB
# --------------------------------------------------

search_response = tavily.search(
    query=research_query,
    max_results=5
)


# --------------------------------------------------
# STEP 3: BUILD NEW EVIDENCE CONTEXT
# --------------------------------------------------

new_evidence = ""

for i, result in enumerate(search_response["results"], start=1):

    new_evidence += f"""
EVIDENCE {i}
Title: {result["title"]}
URL: {result["url"]}
Content: {result["content"]}
"""


# --------------------------------------------------
# STEP 4: VERIFY CLAIMS AGAINST NEW EVIDENCE
# --------------------------------------------------

verification_prompt = f"""
You are an evidence verification agent.

You must verify each claim using ONLY the evidence provided below.

Allowed verdicts:
- SUPPORTED
- PARTIALLY SUPPORTED
- NOT SUPPORTED

Important:
- Do not assume information that is not explicitly present.
- Pay close attention to which model is associated with each metric.
- If a metric belongs to LSTM rather than Siamese-PTM, do NOT treat it
  as evidence for Siamese-PTM.
- A claim is SUPPORTED only when the evidence directly supports it.
- Return a separate verdict and reason for every claim.

CLAIMS:

{claims}

NEW WEB EVIDENCE:

{new_evidence}

Return the results in this format:

CLAIM 1
VERDICT: ...
REASON: ...

CLAIM 2
VERDICT: ...
REASON: ...

CLAIM 3
VERDICT: ...
REASON: ...
"""


verification_completion = groq.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": verification_prompt
        }
    ],
)


verification_result = verification_completion.choices[0].message.content


# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

print("\n" + "=" * 60)
print("SELF-CORRECTION VERIFICATION")
print("=" * 60)

print("\nGENERATED RESEARCH QUERY:")
print(research_query)

print("\n" + "=" * 60)
print("NEW EVIDENCE USED FOR VERIFICATION")
print("=" * 60)

print(new_evidence)

print("\n" + "=" * 60)
print("VERIFICATION RESULTS")
print("=" * 60)

print(verification_result)

print("\n" + "=" * 60)