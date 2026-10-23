from app.rag.vectorstore import get_vector_store
from app.rag.retriever import retrieve
from app.rag.reranker import rerank

from app.verification.claim_extractor import extract_claims
from app.verification.verifier import verify_claims
from app.verification.critic import critique_research


query = "What performance improvement did Siamese-PTM achieve compared to the baseline?"


# Get vector database
collection = get_vector_store()


# Retrieve evidence
retrieved = retrieve(
    collection,
    query,
    top_k=10
)


# Rerank evidence
evidence = rerank(
    query,
    retrieved,
    top_k=5
)


# Extract claims
claims = extract_claims(
    query,
    evidence
)


# Verify claims
verification = verify_claims(
    claims,
    evidence
)


# Critic
critique = critique_research(
    query,
    claims,
    verification
)


print("\nExtracted Claims:\n")
print(claims)

print("\nVerification Results:\n")
print(verification)

print("\nCritic Decision:\n")
print(critique)