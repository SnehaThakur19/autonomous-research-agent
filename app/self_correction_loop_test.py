from groq import Groq
from tavily import TavilyClient
from dotenv import load_dotenv
import os

load_dotenv()

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

MAX_ITERATIONS = 3


# --------------------------------------------------
# INITIAL CLAIMS
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
# INITIAL CRITIC ISSUES
# --------------------------------------------------

critic_issues = """
- The reported 40% performance gain and p-value refer to an LSTM model,
  not to Siamese-PTM.
- The F1 scores of 0.92 and 0.66 are attributed to LSTM, not Siamese-PTM.
- Direct evidence confirming Siamese-PTM's own performance metrics is missing.
"""


# --------------------------------------------------
# FUNCTION 1: GENERATE RESEARCH QUERY
# --------------------------------------------------

def generate_query(issues):

    prompt = f"""
You are an autonomous research query generation agent.

Research topic:
Clinical trial matching using Siamese-PTM.

The critic identified these issues:

{issues}

Generate ONE highly specific web search query that can find
the missing evidence.

The query must:
- Include Siamese-PTM.
- Include clinical trial matching.
- Focus on F1 score or performance compared with Random Forest.
- Be specific enough to avoid unrelated results.
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

    return completion.choices[0].message.content.strip()


# --------------------------------------------------
# FUNCTION 2: SEARCH WEB
# --------------------------------------------------

def search_web(query):

    response = tavily.search(
        query=query,
        max_results=5
    )

    evidence = ""

    for i, result in enumerate(response["results"], start=1):

        evidence += f"""
EVIDENCE {i}
Title: {result["title"]}
URL: {result["url"]}
Content: {result["content"]}
"""

    return evidence


# --------------------------------------------------
# FUNCTION 3: VERIFY CLAIMS
# --------------------------------------------------

def verify_claims(evidence):

    prompt = f"""
You are an evidence verification agent.

Verify every claim using ONLY the provided evidence.

Allowed verdicts:
- SUPPORTED
- PARTIALLY SUPPORTED
- NOT SUPPORTED

Important:
- Do not assume missing information.
- Pay attention to which model produced each metric.
- If a metric belongs to LSTM and not Siamese-PTM,
  do not use it to support a Siamese-PTM claim.

CLAIMS:

{claims}

EVIDENCE:

{evidence}

Return:

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

    completion = groq.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return completion.choices[0].message.content


# --------------------------------------------------
# FUNCTION 4: CRITIC
# --------------------------------------------------

def run_critic(verification):

    prompt = f"""
You are a research critic agent.

Review the verification results below.

Your task is to decide whether additional research is required.

If any important claim is NOT SUPPORTED or PARTIALLY SUPPORTED,
return NEEDS_RESEARCH.

If all claims are adequately supported, return GOOD.

Verification results:

{verification}

Return exactly:

DECISION: GOOD

or

DECISION: NEEDS_RESEARCH

Then list the main issues that require further research.
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

    return completion.choices[0].message.content


# ==================================================
# AUTONOMOUS LOOP
# ==================================================

current_issues = critic_issues


for iteration in range(1, MAX_ITERATIONS + 1):

    print("\n" + "=" * 60)
    print(f"SELF-CORRECTION ITERATION {iteration}")
    print("=" * 60)


    # ----------------------------------------------
    # Generate targeted query
    # ----------------------------------------------

    query = generate_query(current_issues)

    print("\nRESEARCH QUERY:")
    print(query)


    # ----------------------------------------------
    # Search web
    # ----------------------------------------------

    evidence = search_web(query)

    print("\nNEW EVIDENCE:")
    print(evidence)


    # ----------------------------------------------
    # Verify claims
    # ----------------------------------------------

    verification = verify_claims(evidence)

    print("\nVERIFICATION:")
    print(verification)


    # ----------------------------------------------
    # Critic
    # ----------------------------------------------

    critic = run_critic(verification)

    print("\nCRITIC:")
    print(critic)


    # ----------------------------------------------
    # Decide whether to continue
    # ----------------------------------------------

    if "DECISION: GOOD" in critic:
        print("\nResearch is sufficient.")
        break


    if "DECISION: NEEDS_RESEARCH" in critic:

        print("\nMore research is required.")

        current_issues = critic

        if iteration == MAX_ITERATIONS:
            print("\nMaximum iterations reached.")
            print("Stopping self-correction loop.")