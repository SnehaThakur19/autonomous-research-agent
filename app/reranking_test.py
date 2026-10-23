from sentence_transformers import CrossEncoder

# Load reranking model
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# User question
query = "How does RAG use external information?"

# Candidate chunks retrieved from Vector DB
chunks = [
    "RAG combines information retrieval with text generation.",
    "RAG uses external information to answer questions.",
    "Python is a popular programming language."
]

# Create query-chunk pairs
pairs = [[query, chunk] for chunk in chunks]

# Calculate relevance scores
scores = model.predict(pairs)

print("\nQUERY:")
print(query)

print("\nRERANKING SCORES:")

for i, score in enumerate(scores, start=1):
    print(f"Chunk {i}: {score:.4f}")

# Sort chunks by relevance
ranked = sorted(
    zip(chunks, scores),
    key=lambda x: x[1],
    reverse=True
)

print("\nRERANKED CHUNKS:")

for i, (chunk, score) in enumerate(ranked, start=1):
    print(f"\n{i}. Score: {score:.4f}")
    print(chunk)