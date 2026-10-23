from pypdf import PdfReader

pdf_path = "data/research_paper.pdf"

reader = PdfReader(pdf_path)

print("\n" + "=" * 60)
print("PDF INGESTION")
print("=" * 60)

print("\nTotal pages:", len(reader.pages))

for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text()

    print("\n" + "-" * 60)
    print(f"PAGE {page_number}")
    print("-" * 60)

    print(text[:2000])