from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def create_embeddings(chunks):
    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(texts)

    return embeddings