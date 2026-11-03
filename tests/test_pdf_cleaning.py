from app.pdf_loader import load_pdf
from app.text_cleaner import clean_text


PDF_PATH = "data/pdfs/Testing-in-a-Small-Company1.pdf"


def test_pdf_text_cleaning():
    pages = load_pdf(PDF_PATH)

    cleaned_first_page = clean_text(pages[0].text)

    # Verify cleaned output.
    assert isinstance(cleaned_first_page, str)
    assert cleaned_first_page.strip()

    # Verify cleaning returns a string.
    assert len(cleaned_first_page) > 0

    # Verify page metadata remains available.
    assert pages[0].filename
    assert pages[0].page_number == 1

    # Verify every page has a text field.
    assert all(
        isinstance(page.text, str)
        for page in pages
    )

    print("Total pages:", len(pages))
    print("Original first-page characters:", len(pages[0].text))
    print("Cleaned first-page characters:", len(cleaned_first_page))

    print("\nCleaned first-page text:")
    print(cleaned_first_page[:1000])

    print("\nMetadata preserved:")
    print("Filename:", pages[0].filename)
    print("Page number:", pages[0].page_number)

    print(
        "\nAll pages have text fields:",
        all(isinstance(page.text, str) for page in pages),
    )