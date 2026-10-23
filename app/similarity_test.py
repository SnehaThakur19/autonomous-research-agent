from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Our text chunks
chunks = [
    "RAG combines information retrieval with text generation.",
    "RAG uses external information to answer questions.",
    "Python is a popular programming language."
]

# Convert chunks into embeddings
chunk_embeddings = model.encode(chunks)

# User's question
query = "How does RAG use external information?"

# Convert question into an embedding
query_embedding = model.encode([query])

# Calculate similarity between question and every chunk
similarity_scores = cosine_similarity(
    query_embedding,
    chunk_embeddings
)[0]

# Display results
print("\nQUERY:")
print(query)

print("\nSIMILARITY SCORES:")

for i, score in enumerate(similarity_scores, start=1):
    print(f"Chunk {i}: {score:.4f}")

# Find the most relevant chunk
best_index = similarity_scores.argmax()

print("\nMOST RELEVANT CHUNK:")
print(chunks[best_index])

print(f"\nSimilarity Score: {similarity_scores[best_index]:.4f}")