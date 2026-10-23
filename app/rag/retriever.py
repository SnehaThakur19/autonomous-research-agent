from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def retrieve(collection, query, top_k=10):
    model = SentenceTransformer(MODEL_NAME)

    query_embedding = model.encode([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    retrieved_chunks = []

    for i in range(len(results["documents"][0])):
        retrieved_chunks.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return retrieved_chunks