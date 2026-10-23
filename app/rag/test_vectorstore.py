from app.rag.loader import load_pdf
from app.rag.chunker import chunk_pages
from app.rag.embeddings import create_embeddings
from app.rag.vectorstore import create_vector_store, add_documents


pdf_path = "data/research_paper.pdf"

pages = load_pdf(pdf_path)
chunks = chunk_pages(pages)
embeddings = create_embeddings(chunks)

collection = create_vector_store()

add_documents(
    collection,
    chunks,
    embeddings
)

print("Total chunks:", len(chunks))
print("Vector DB count:", collection.count())