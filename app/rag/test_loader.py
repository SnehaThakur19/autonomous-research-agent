from app.rag.loader import load_pdf


pdf_path = "data/research_paper.pdf"

pages = load_pdf(pdf_path)

print("Total pages:", len(pages))

for page in pages[:2]:
    print("\n--- PAGE", page["page"], "---")
    print(page["text"][:500])