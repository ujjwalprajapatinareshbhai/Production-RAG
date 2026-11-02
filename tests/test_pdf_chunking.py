
from app.pdf_loader import load_pdf
from app.chunker import chunk_pages


pdf_path = "data/pdfs/Testing-in-a-Small-Company1.pdf"

pages = load_pdf(pdf_path)
chunks = chunk_pages(pages, chunk_size=400, overlap=60)

print("Pages loaded:", len(pages))
print("Chunks created:", len(chunks))

if chunks:
    first = chunks[0]

    print("\nFirst chunk ID:", first.chunk_id)
    print("First chunk index:", first.chunk_index)
    print("First chunk source pages:",
          first.start_page, "-", first.end_page)
    print("First chunk word count:", len(first.text.split()))
    print("\nFirst chunk text:")
    print(first.text[:1000])

    print("\nAll chunks share one document ID:",
          all(c.document_id == pages[0].document_id for c in chunks))

    print("All chunks have valid page ranges:",
          all(1 <= c.start_page <= c.end_page <= len(pages)
              for c in chunks))
              
    print("Exact source pages:", first.source_pages)
    print("Paragraph breaks preserved:", "\n\n" in first.text)
    print("Number of source pages:", len(first.source_pages))