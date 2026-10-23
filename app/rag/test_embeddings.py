from app.rag.loader import load_pdf
from app.rag.chunker import chunk_pages
from app.rag.embeddings import create_embeddings


pdf_path = "data/research_paper.pdf"

pages = load_pdf(pdf_path)
chunks = chunk_pages(pages)

embeddings = create_embeddings(chunks)

print("Total chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)
print("First embedding length:", len(embeddings[0]))