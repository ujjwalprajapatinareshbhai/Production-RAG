
from app.pdf_loader import load_pdf


pdf_path = "data/pdfs/Testing-in-a-Small-Company1.pdf"

pages = load_pdf(pdf_path)

print("Total pages:", len(pages))
print("First page filename:", pages[0].filename)
print("First page number:", pages[0].page_number)
print("Document ID:", pages[0].document_id)
print("First page text:")
print(pages[0].text[:500])

print("\nLast page number:", pages[-1].page_number)
print("All pages share one document ID:",
      all(page.document_id == pages[0].document_id for page in pages))