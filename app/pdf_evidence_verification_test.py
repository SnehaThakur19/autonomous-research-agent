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
# 3. Claims extracted from the real PDF
# --------------------------------------------------

claims = [
    {
        "claim": (
            "Siamese-PTM achieved a statistically significant "
            "40% performance gain over the rule-based Random "
            "Forest baseline (p-value = 0.006)."
        ),
        "source": "data/research_paper.pdf",
        "page": 6,
        "chunk_id": "page_6_chunk_1",
        "evidence": (
            "LSTM ranks best with an F1 score of 0.92. "
            "This deep learning performance is much above "
            "the rule-based Random Forest model's performance "
            "(0.66 F1), in particular by a statistically "
            "significant 40% performance gain (p-value = 0.006)."
        )
    },

    {
        "claim": (
            "Siamese-PTM attained an F1 score of 0.92, "
            "which is higher than the Random Forest baseline's "
            "F1 score of 0.66."
        ),
        "source": "data/research_paper.pdf",
        "page": 6,
        "chunk_id": "page_6_chunk_1",
        "evidence": (
            "LSTM ranks best with an F1 score of 0.92. "
            "This deep learning performance is much above "
            "the rule-based Random Forest model's performance "
            "(0.66 F1)."
        )
    },

    {
        "claim": (
            "Siamese-PTM outperformed seven out of ten "
            "counterpart single-encoder models."
        ),
        "source": "data/research_paper.pdf",
        "page": 6,
        "chunk_id": "page_6_chunk_3",
        "evidence": (
            "Siamese-PTM is able to outperform seven out "
            "of the ten counterpart Single-encoder models, "
            "while performing comparably with the remaining three."
        )
    }
]


# --------------------------------------------------
# 4. Verify each claim
# --------------------------------------------------

print("\n" + "=" * 60)
print("REAL PDF EVIDENCE VERIFICATION")
print("=" * 60)


for i, item in enumerate(claims, start=1):

    prompt = f"""
You are an evidence verification agent.

Your task is to determine whether the provided evidence
supports the given claim.

CLAIM:
{item["claim"]}

EVIDENCE:
{item["evidence"]}

SOURCE:
{item["source"]}

PAGE:
{item["page"]}

CHUNK ID:
{item["chunk_id"]}

Classify the claim using exactly one of:

SUPPORTED
PARTIALLY SUPPORTED
NOT SUPPORTED

Rules:

SUPPORTED:
The evidence directly supports the complete claim.

PARTIALLY SUPPORTED:
The evidence supports only part of the claim,
or the claim contains an interpretation that goes
beyond what the evidence directly establishes.

NOT SUPPORTED:
The evidence does not support the claim.

Do not use outside knowledge.

Return exactly this format:

VERDICT: <one of the three labels>

REASON:
<short explanation>
"""


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
    # 5. Display verification result
    # --------------------------------------------------

    print("\n" + "-" * 60)
    print(f"CLAIM {i}")
    print("-" * 60)

    print("Claim:")
    print(item["claim"])

    print("\nSource:", item["source"])
    print("Page:", item["page"])
    print("Chunk ID:", item["chunk_id"])

    print("\nVerification:")
    print(result)