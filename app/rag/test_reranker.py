from app.rag.vectorstore import get_vector_store
from app.rag.retriever import retrieve
from app.rag.reranker import rerank


query = "What performance improvement did Siamese-PTM achieve compared to the baseline?"

collection = get_vector_store()

retrieved_chunks = retrieve(
    collection,
    query,
    top_k=10
)

reranked_chunks = rerank(
    query,
    retrieved_chunks,
    top_k=5
)

print("Query:", query)
print("\nReranked results:", len(reranked_chunks))

for i, result in enumerate(reranked_chunks, start=1):
    print(f"\n--- RERANKED RESULT {i} ---")
    print("Rerank Score:", result["rerank_score"])
    print("Metadata:", result["metadata"])
    print("Text:", result["text"][:500])