from groq import Groq
from dotenv import load_dotenv
import os


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 2. Connect to Groq
# --------------------------------------------------

groq = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# --------------------------------------------------
# 3. Verification results from Step 12F
# --------------------------------------------------

verification_results = [

    {
        "claim_id": 1,
        "claim": (
            "Siamese-PTM achieved a statistically significant "
            "40% performance gain over the rule-based Random "
            "Forest baseline (p-value = 0.006)."
        ),
        "verdict": "PARTIALLY SUPPORTED",
        "reason": (
            "The evidence confirms a statistically significant "
            "40% performance gain over the Random Forest baseline "
            "with p = 0.006, but it attributes this gain to an "
            "LSTM model, not to Siamese-PTM as claimed."
        ),
        "source": "data/research_paper.pdf",
        "page": 6,
        "chunk_id": "page_6_chunk_1"
    },

    {
        "claim_id": 2,
        "claim": (
            "Siamese-PTM attained an F1 score of 0.92, "
            "which is higher than the Random Forest baseline's "
            "F1 score of 0.66."
        ),
        "verdict": "NOT SUPPORTED",
        "reason": (
            "The evidence states that an LSTM model achieved "
            "an F1 score of 0.92, not a Siamese-PTM model. "
            "The numerical scores are present, but the evidence "
            "does not support the attribution to Siamese-PTM."
        ),
        "source": "data/research_paper.pdf",
        "page": 6,
        "chunk_id": "page_6_chunk_1"
    },

    {
        "claim_id": 3,
        "claim": (
            "Siamese-PTM outperformed seven out of ten "
            "counterpart single-encoder models."
        ),
        "verdict": "SUPPORTED",
        "reason": (
            "The evidence explicitly states that Siamese-PTM "
            "outperforms seven of the ten counterpart "
            "single-encoder models."
        ),
        "source": "data/research_paper.pdf",
        "page": 6,
        "chunk_id": "page_6_chunk_3"
    }
]


# --------------------------------------------------
# 4. Build critic context
# --------------------------------------------------

context = ""

for item in verification_results:

    context += f"""
CLAIM {item["claim_id"]}

Claim:
{item["claim"]}

Verification Verdict:
{item["verdict"]}

Verification Reason:
{item["reason"]}

Source:
{item["source"]}

Page:
{item["page"]}

Chunk ID:
{item["chunk_id"]}

"""


# --------------------------------------------------
# 5. Critic prompt
# --------------------------------------------------

prompt = f"""
You are a research critic agent.

Your job is to review the verification results of
multiple claims extracted from a research paper.

Research verification results:

{context}

Evaluate the overall reliability of the research.

Pay special attention to:

1. Unsupported claims.
2. Partially supported claims.
3. Incorrect attribution of results to the wrong model.
4. Missing or insufficient evidence.
5. Whether additional research is required.

Decision rules:

GOOD:
Use this only when all important claims are supported
by their evidence and there are no significant grounding
problems.

NEEDS_RESEARCH:
Use this when any important claim is unsupported,
partially supported, incorrectly attributed, or requires
additional evidence.

Do not use outside knowledge.

Return exactly this format:

DECISION: GOOD or NEEDS_RESEARCH

ISSUES:
- issue 1
- issue 2
- issue 3

RECOMMENDATION:
<what the research system should do next>
"""


# --------------------------------------------------
# 6. Run critic agent
# --------------------------------------------------

completion = groq.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


result = completion.choices[0].message.content


# --------------------------------------------------
# 7. Display critic result
# --------------------------------------------------

print("\n" + "=" * 60)
print("REAL PDF CRITIC AGENT")
print("=" * 60)

print("\nVerification Summary:")

for item in verification_results:

    print(
        f"\nClaim {item['claim_id']}: "
        f"{item['verdict']}"
    )


print("\n" + "=" * 60)
print("CRITIC DECISION")
print("=" * 60)

print(result)