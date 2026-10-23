from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_path = "data/research_paper.pdf"

# Read PDF
reader = PdfReader(pdf_path)

# Create text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

all_chunks = []

# Process each page separately
for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text()

    if not text:
        continue

    # Split page text into chunks
    chunks = splitter.split_text(text)

    for chunk_number, chunk in enumerate(chunks, start=1):

        chunk_data = {
            "text": chunk,
            "metadata": {
                "source": pdf_path,
                "page": page_number,
                "chunk_id": f"page_{page_number}_chunk_{chunk_number}"
            }
        }

        all_chunks.append(chunk_data)


# Display results
print("\n" + "=" * 60)
print("PDF PAGE-AWARE CHUNKING")
print("=" * 60)

print("\nTotal pages:", len(reader.pages))
print("Total chunks:", len(all_chunks))

# Show first 5 chunks
for i, item in enumerate(all_chunks[:5], start=1):

    print("\n" + "-" * 60)
    print(f"CHUNK {i}")
    print("-" * 60)

    print("Chunk ID:", item["metadata"]["chunk_id"])
    print("Source:", item["metadata"]["source"])
    print("Page:", item["metadata"]["page"])

    print("\nText:")
    print(item["text"])