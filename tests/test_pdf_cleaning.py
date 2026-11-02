
from app.pdf_loader import load_pdf
from app.text_cleaner import clean_text


pdf_path = "data/pdfs/Testing-in-a-Small-Company1.pdf"

pages = load_pdf(pdf_path)

cleaned_first_page = clean_text(pages[0].text)

print("Total pages:", len(pages))
print("Original first-page characters:", len(pages[0].text))
print("Cleaned first-page characters:", len(cleaned_first_page))

print("\nCleaned first-page text:")
print(cleaned_first_page[:1000])

print("\nMetadata preserved:")
print("Filename:", pages[0].filename)
print("Page number:", pages[0].page_number)

print("\nAll pages have text fields:",
      all(isinstance(page.text, str) for page in pages))