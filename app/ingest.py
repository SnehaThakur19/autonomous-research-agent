"""Ingest one or more PDFs into the ChromaDB vector store.

Usage:
    python -m app.ingest data/research_paper.pdf [--rebuild]
"""

import argparse
import os

from app.rag.loader import load_pdf
from app.rag.chunker import chunk_pages
from app.rag.embeddings import create_embeddings
from app.rag.vectorstore import (
    create_vector_store,
    get_vector_store,
    add_documents,
)


def ingest_pdfs(pdf_paths, rebuild=False):
    """Load, chunk, embed, and upsert PDFs into the vector store.

    Returns the number of chunks ingested.
    """
    if rebuild:
        collection = create_vector_store()
    else:
        collection = get_vector_store()

    total_chunks = 0

    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages = load_pdf(pdf_path)

        chunks = chunk_pages(pages)

        if not chunks:
            print(f"Warning: no text extracted from {pdf_path}")
            continue

        embeddings = create_embeddings(chunks)

        add_documents(collection, chunks, embeddings)

        total_chunks += len(chunks)

        print(
            f"Ingested {len(chunks)} chunks "
            f"({len(pages)} pages) from {pdf_path}"
        )

    return total_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into the research vector store"
    )
    parser.add_argument(
        "pdfs",
        nargs="+",
        help="Path(s) to PDF file(s)"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop and recreate the collection before ingesting"
    )

    args = parser.parse_args()

    total = ingest_pdfs(args.pdfs, rebuild=args.rebuild)

    print(f"Done. {total} chunks available for retrieval.")


if __name__ == "__main__":
    main()
