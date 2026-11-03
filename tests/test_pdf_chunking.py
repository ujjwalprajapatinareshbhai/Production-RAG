from uuid import uuid4

from app.pdf_loader import load_pdf
from app.chunker import chunk_pages


PDF_PATH = "data/pdfs/Testing-in-a-Small-Company1.pdf"
TEST_TENANT_ID = uuid4()


def test_pdf_chunking():
    """Verify PDF loading and tenant-aware chunk creation."""

    pages = load_pdf(PDF_PATH)

    chunks = chunk_pages(
        pages,
        tenant_id=TEST_TENANT_ID,
        chunk_size=400,
        overlap=60,
    )

    assert pages
    assert chunks

    print("Pages loaded:", len(pages))
    print("Chunks created:", len(chunks))

    first = chunks[0]

    print("\nFirst chunk ID:", first.chunk_id)
    print("First chunk index:", first.chunk_index)
    print(
        "First chunk source pages:",
        first.start_page,
        "-",
        first.end_page,
    )
    print("First chunk word count:", len(first.text.split()))
    print("\nFirst chunk text:")
    print(first.text[:1000])

    # Verify tenant propagation.
    assert all(c.tenant_id == TEST_TENANT_ID for c in chunks)

    # Verify document identity.
    assert all(
        c.document_id == pages[0].document_id
        for c in chunks
    )

    # Verify page ranges.
    assert all(
        1 <= c.start_page <= c.end_page <= len(pages)
        for c in chunks
    )

    print(
        "\nAll chunks share one document ID:",
        all(c.document_id == pages[0].document_id for c in chunks),
    )

    print(
        "All chunks have valid page ranges:",
        all(
            1 <= c.start_page <= c.end_page <= len(pages)
            for c in chunks
        ),
    )

    print("Tenant ID:", TEST_TENANT_ID)
    print("Exact source pages:", first.source_pages)
    print("Paragraph breaks preserved:", "\n\n" in first.text)
    print("Number of source pages:", len(first.source_pages))