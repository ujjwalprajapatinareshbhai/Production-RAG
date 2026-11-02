from app.pdf_loader import load_pdf
from app.chunker import chunk_pages, chunk_pages_independently

PDF_PATH = "data/pdfs/Testing-in-a-Small-Company1.pdf"
TARGET = "current software process used in company x"

pages = load_pdf(PDF_PATH)

strategies = {
    "Current chunks (400 words, 60 overlap)": chunk_pages(
        pages,
        chunk_size=400,
        overlap=60,
    ),
    "Page-aware chunks (400 words, 60 overlap)": chunk_pages_independently(
        pages,
        chunk_size=400,
        overlap=60,
    ),
}

print("PDF pages:", len(pages))

for strategy_name, chunks in strategies.items():
    print("\n" + "=" * 65)
    print(strategy_name)
    print("=" * 65)

    print("Total chunks:", len(chunks))

    matches = [
        chunk
        for chunk in chunks
        if TARGET in chunk.text.lower()
    ]

    print("Chunks containing the Waterfall statement:", len(matches))

    for index, chunk in enumerate(matches, start=1):
        print(f"\nMatch {index}")
        print("Chunk index:", chunk.chunk_index)
        print("Source pages:", chunk.source_pages)
        print("Word count:", len(chunk.text.split()))
        print("Character count:", len(chunk.text))
        print("\nRelevant text:")

        position = chunk.text.lower().find(TARGET)
        start = max(0, position - 200)
        end = min(len(chunk.text), position + 700)

        print(chunk.text[start:end])