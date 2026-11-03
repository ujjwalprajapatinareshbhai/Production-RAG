from app.pdf_loader import load_pdf


PDF_PATH = "data/pdfs/Testing-in-a-Small-Company1.pdf"


def test_load_pdf():
    pages = load_pdf(PDF_PATH)

    assert len(pages) == 76

    # Verify first page metadata.
    assert pages[0].filename
    assert pages[0].page_number == 1
    assert pages[0].document_id

    assert isinstance(pages[0].text, str)
    assert pages[0].text.strip()

    # Verify last page.
    assert pages[-1].page_number == 76

    # Every page must belong to the same document.
    assert all(
        page.document_id == pages[0].document_id
        for page in pages
    )

    print("Total pages:", len(pages))
    print("First page filename:", pages[0].filename)
    print("First page number:", pages[0].page_number)
    print("Document ID:", pages[0].document_id)
    print("First page text:")
    print(pages[0].text[:500])

    print("\nLast page number:", pages[-1].page_number)
    print(
        "All pages share one document ID:",
        all(
            page.document_id == pages[0].document_id
            for page in pages
        ),
    )