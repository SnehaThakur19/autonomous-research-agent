from app.rag.vectorstore import get_vector_store
from app.rag.retriever import retrieve


collection = get_vector_store()

query = "What performance improvement did Siamese-PTM achieve compared to the baseline?"

results = retrieve(
    collection,
    query,
    top_k=5
)

print("Query:", query)
print("\nRetrieved chunks:", len(results))

for i, result in enumerate(results, start=1):
    print(f"\n--- RESULT {i} ---")
    print("Distance:", result["distance"])
    print("Metadata:", result["metadata"])
    print("Text:", result["text"][:500])