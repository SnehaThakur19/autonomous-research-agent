from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from groq import Groq
from dotenv import load_dotenv
import os


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 2. Load embedding model
# --------------------------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 3. Connect to ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="clinical_trial_paper"
)


# --------------------------------------------------
# 4. Connect to Groq
# --------------------------------------------------

groq = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# --------------------------------------------------
# 5. Get research question
# --------------------------------------------------

query = input(
    "\nEnter your research question: "
)


# --------------------------------------------------
# 6. Convert query to embedding
# --------------------------------------------------

query_embedding = embedding_model.encode(
    [query]
)[0]


# --------------------------------------------------
# 7. Retrieve candidate evidence
# --------------------------------------------------

results = collection.query(
    query_embeddings=[
        query_embedding.tolist()
    ],
    n_results=10
)


documents = results["documents"][0]
metadatas = results["metadatas"][0]


# --------------------------------------------------
# 8. Load CrossEncoder
# --------------------------------------------------

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# --------------------------------------------------
# 9. Create query-document pairs
# --------------------------------------------------

pairs = [
    [query, document]
    for document in documents
]


# --------------------------------------------------
# 10. Calculate reranking scores
# --------------------------------------------------

scores = reranker.predict(pairs)


# --------------------------------------------------
# 11. Combine evidence + metadata + score
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
# 12. Sort evidence
# --------------------------------------------------

ranked_results.sort(
    key=lambda x: x["score"],
    reverse=True
)


# --------------------------------------------------
# 13. Select top evidence
# --------------------------------------------------

top_results = ranked_results[:5]


# --------------------------------------------------
# 14. Build evidence context
# --------------------------------------------------

context = ""

for i, result in enumerate(
    top_results,
    start=1
):

    metadata = result["metadata"]

    context += f"""
EVIDENCE {i}

Source: {metadata["source"]}
Page: {metadata["page"]}
Chunk ID: {metadata["chunk_id"]}

Text:
{result["document"]}

"""


# --------------------------------------------------
# 15. Create claim extraction prompt
# --------------------------------------------------

prompt = f"""
You are a research claim extraction agent.

Your task is to extract factual claims from the
provided research evidence.

User research question:
{query}

Evidence:
{context}

Instructions:

1. Extract only factual claims explicitly supported
   by the provided evidence.

2. Do not add information from your own knowledge.

3. Do not make assumptions.

4. Keep each claim concise and specific.

5. For every claim, include:
   - Claim
   - Evidence number
   - Source
   - Page
   - Chunk ID

6. Extract only important claims relevant to the
   user's research question.

Return the result in this format:

CLAIM 1:
Claim: ...
Evidence: ...
Source: ...
Page: ...
Chunk ID: ...

CLAIM 2:
Claim: ...
Evidence: ...
Source: ...
Page: ...
Chunk ID: ...
"""


# --------------------------------------------------
# 16. Send request to Groq
# --------------------------------------------------

completion = groq.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# --------------------------------------------------
# 17. Display extracted claims
# --------------------------------------------------

print("\n" + "=" * 60)
print("EXTRACTED CLAIMS")
print("=" * 60)

print(
    completion.choices[0].message.content
)