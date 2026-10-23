from pypdf import PdfReader


def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            pages.append({
                "text": text.strip(),
                "page": page_number,
                "source": pdf_path
            })

    return pages