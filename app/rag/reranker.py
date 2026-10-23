from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank(query, retrieved_chunks, top_k=5):
    model = CrossEncoder(MODEL_NAME)

    pairs = [
        [query, chunk["text"]]
        for chunk in retrieved_chunks
    ]

    scores = model.predict(pairs)

    reranked_chunks = []

    for chunk, score in zip(retrieved_chunks, scores):
        reranked_chunks.append({
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "distance": chunk["distance"],
            "rerank_score": float(score)
        })

    reranked_chunks.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked_chunks[:top_k]