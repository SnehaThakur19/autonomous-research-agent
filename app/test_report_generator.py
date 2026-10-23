from app.report_generator import generate_report

from app.rag.vectorstore import get_vector_store
from app.rag.retriever import retrieve
from app.rag.reranker import rerank


query = "What performance improvement did Siamese-PTM achieve compared to the baseline?"


# Get existing PDF evidence
collection = get_vector_store()

retrieved = retrieve(
    collection,
    query,
    top_k=10
)

evidence = rerank(
    query,
    retrieved,
    top_k=3
)


# Sample results from our previous verification stage
claims = """
1. Siamese-PTM was reported to achieve a 40% performance improvement
over the baseline with p=0.006.

2. The reported F1 score of 0.92 needs careful attribution to the
specific model.
"""

verification_results = """
Claim 1:
Status: PARTIALLY SUPPORTED
Reason: The 40% improvement and p=0.006 are present, but the exact
baseline attribution needs verification.

Claim 2:
Status: NOT SUPPORTED
Reason: The retrieved evidence does not clearly establish that
Siamese-PTM achieved F1=0.92 against Random Forest F1=0.66.
"""

critic_result = """
Decision: NEEDS_RESEARCH

Reason: Important numerical results and model attribution remain
unresolved.
"""


report = generate_report(
    query=query,
    claims=claims,
    verification_results=verification_results,
    critic_result=critic_result,
    evidence=evidence,
    web_results=[]
)


print("\nFINAL RESEARCH REPORT")
print("=====================")
print(report)