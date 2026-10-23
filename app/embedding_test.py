from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example text chunks
chunks = [
    "RAG combines information retrieval with text generation.",
    "RAG uses external information to answer questions.",
    "Python is a popular programming language."
]

# Convert chunks into vectors
embeddings = model.encode(chunks)

# Display results
for i, embedding in enumerate(embeddings, start=1):
    print(f"\nChunk {i}")
    print("Vector length:", len(embedding))
    print("First 10 values:", embedding[:10])

print("\nTotal embeddings:", len(embeddings))