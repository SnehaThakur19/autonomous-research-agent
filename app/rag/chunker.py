from langchain_text_splitters import RecursiveCharacterTextSplitter

import os


def chunk_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        # Prefix chunk IDs with the source filename so multiple
        # PDFs never collide (e.g. "paperA_page_1_chunk_1")
        source_name = os.path.splitext(
            os.path.basename(page["source"])
        )[0]

        for chunk_number, chunk in enumerate(page_chunks, start=1):
            chunks.append({
                "text": chunk,
                "metadata": {
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_id": (
                        f"{source_name}"
                        f"_page_{page['page']}"
                        f"_chunk_{chunk_number}"
                    )
                }
            })

    return chunks