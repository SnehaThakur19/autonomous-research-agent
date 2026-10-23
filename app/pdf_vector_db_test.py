from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# --------------------------------------------------
# 1. READ PDF
# --------------------------------------------------

pdf_path = "data/research_paper.pdf"

reader = PdfReader(pdf_path)

print("\n" + "=" * 60)
print("PDF → EMBEDDINGS → VECTOR DATABASE")
print("=" * 60)

print("\nTotal pages:", len(reader.pages))


# --------------------------------------------------
# 2. CHUNK PDF TEXT
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

all_chunks = []

for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text()

    if not text:
        continue

    chunks = splitter.split_text(text)

    for chunk_number, chunk in enumerate(chunks, start=1):

        all_chunks.append({
            "text": chunk,
            "metadata": {
                "source": pdf_path,
                "page": page_number,
                "chunk_id": f"page_{page_number}_chunk_{chunk_number}"
            }
        })


print("Total chunks:", len(all_chunks))


# --------------------------------------------------
# 3. GENERATE EMBEDDINGS
# --------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [item["text"] for item in all_chunks]

embeddings = model.encode(texts)

print("Embedding dimension:", len(embeddings[0]))
print("Total embeddings:", len(embeddings))


# --------------------------------------------------
# 4. CONNECT TO CHROMADB
# --------------------------------------------------

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="clinical_trial_paper"
)


# --------------------------------------------------
# 5. STORE CHUNKS + EMBEDDINGS + METADATA
# --------------------------------------------------

collection.add(
    ids=[
        item["metadata"]["chunk_id"]
        for item in all_chunks
    ],
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=[
        item["metadata"]
        for item in all_chunks
    ]
)

print("\nChunks successfully stored in ChromaDB!")
print("Documents in collection:", collection.count())


# --------------------------------------------------
# 6. TEST SEMANTIC SEARCH
# --------------------------------------------------

query = input("\nEnter your research question: ")

query_embedding = model.encode([query])[0]

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3
)


# --------------------------------------------------
# 7. DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("MOST RELEVANT PDF EVIDENCE")
print("=" * 60)

for i in range(len(results["documents"][0])):

    document = results["documents"][0][i]
    metadata = results["metadatas"][0][i]
    distance = results["distances"][0][i]

    print("\n" + "-" * 60)
    print(f"RESULT {i + 1}")
    print("-" * 60)

    print("Chunk ID:", metadata["chunk_id"])
    print("Source:", metadata["source"])
    print("Page:", metadata["page"])
    print("Distance:", round(distance, 4))

    print("\nEvidence:")
    print(document)