import chromadb


COLLECTION_NAME = "research_documents"
DB_PATH = "./chroma_db"


def create_vector_store():
    client = chromadb.PersistentClient(path=DB_PATH)

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME
    )

    return collection


def get_vector_store():
    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    return collection


def add_documents(collection, chunks, embeddings):
    ids = [chunk["metadata"]["chunk_id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    return collection