from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb


# --------------------------------------------------
# 1. Load embedding model
# --------------------------------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# 2. Connect to ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="clinical_trial_paper"
)


# --------------------------------------------------
# 3. Get user research question
# --------------------------------------------------

query = input("\nEnter your research question: ")


# --------------------------------------------------
# 4. Convert query into embedding
# --------------------------------------------------

query_embedding = embedding_model.encode([query])[0]


# --------------------------------------------------
# 5. Retrieve candidate chunks from ChromaDB
# --------------------------------------------------

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=10
)


documents = results["documents"][0]
metadatas = results["metadatas"][0]


print("\n" + "=" * 60)
print("CHROMADB RETRIEVED CANDIDATES")
print("=" * 60)

print("\nCandidates retrieved:", len(documents))


# --------------------------------------------------
# 6. Load CrossEncoder reranker
# --------------------------------------------------

print("\nLoading CrossEncoder reranker...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# --------------------------------------------------
# 7. Create query-document pairs
# --------------------------------------------------

pairs = [
    [query, document]
    for document in documents
]


# --------------------------------------------------
# 8. Calculate reranking scores
# --------------------------------------------------

scores = reranker.predict(pairs)


# --------------------------------------------------
# 9. Combine document + metadata + score
# --------------------------------------------------

ranked_results = []

for document, metadata, score in zip(
    documents,
    metadatas,
    scores
):

    ranked_results.append({
        "document": document,
        "metadata": metadata,
        "score": float(score)
    })


# --------------------------------------------------
# 10. Sort by CrossEncoder score
# --------------------------------------------------

ranked_results.sort(
    key=lambda x: x["score"],
    reverse=True
)


# --------------------------------------------------
# 11. Display reranked evidence
# --------------------------------------------------

print("\n" + "=" * 60)
print("RERANKED PDF EVIDENCE")
print("=" * 60)


for i, result in enumerate(ranked_results, start=1):

    metadata = result["metadata"]

    print("\n" + "-" * 60)
    print(f"RANK {i}")
    print("-" * 60)

    print("Reranking Score:",
          round(result["score"], 4))

    print("Source:",
          metadata["source"])

    print("Page:",
          metadata["page"])

    print("Chunk ID:",
          metadata["chunk_id"])

    print("\nEvidence:")
    print(result["document"])