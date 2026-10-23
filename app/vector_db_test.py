import chromadb

# Create a local ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Get our collection
collection = client.get_collection(
    name="research_documents"
)

# User's question
query = "How does RAG use external information?"

# Search the vector database
results = collection.query(
    query_texts=[query],
    n_results=2
)

print("\nQUERY:")
print(query)

print("\nMOST RELEVANT CHUNKS:")

for i, document in enumerate(results["documents"][0], start=1):
    print(f"\n{i}. {document}")