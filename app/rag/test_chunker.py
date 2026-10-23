from app.rag.loader import load_pdf
from app.rag.chunker import chunk_pages


pdf_path = "data/research_paper.pdf"

pages = load_pdf(pdf_path)
chunks = chunk_pages(pages)

print("Total pages:", len(pages))
print("Total chunks:", len(chunks))

for chunk in chunks[:3]:
    print("\n--- CHUNK ---")
    print("Metadata:", chunk["metadata"])
    print("Text:", chunk["text"][:300])