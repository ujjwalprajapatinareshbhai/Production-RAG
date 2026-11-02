
from pathlib import Path


import pymupdf
import hashlib

from app.models import Page


def load_pdf(pdf_path: str | Path) -> list[Page]:
    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document_id = calculate_file_hash(pdf_path)
    filename = pdf_path.name
    pages: list[Page] = []

    with pymupdf.open(pdf_path) as document:
        for index, pdf_page in enumerate(document):
            text = pdf_page.get_text("text").strip()

            page = Page(
                document_id=document_id,
                filename=filename,
                page_number=index + 1,
                text=text,
            )

            pages.append(page)

    if not pages:
        raise ValueError(f"PDF contains no pages: {filename}")

    return pages

def calculate_file_hash(pdf_path: str | Path) -> str:
    """Return a SHA-256 hash of a file without loading it all into memory."""
    sha256 = hashlib.sha256()

    with open(pdf_path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()