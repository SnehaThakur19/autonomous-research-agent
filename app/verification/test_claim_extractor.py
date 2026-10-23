from app.rag.vectorstore import get_vector_store
from app.rag.retriever import retrieve
from app.rag.reranker import rerank
from app.verification.claim_extractor import extract_claims


query = "What performance improvement did Siamese-PTM achieve compared to the baseline?"


collection = get_vector_store()

retrieved = retrieve(
    collection,
    query,
    top_k=10
)

evidence = rerank(
    query,
    retrieved,
    top_k=5
)


claims = extract_claims(
    query,
    evidence
)


print("\nExtracted Claims:\n")
print(claims)