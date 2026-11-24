import sys
import re
from pathlib import Path
from uuid import UUID

# Add the project root (D:\Production-RAG) to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.chunker import chunk_pages
from app.pdf_loader import load_pdf


PDF_PATH = PROJECT_ROOT / "data" / "pdfs" / "Testing-in-a-Small-Company1.pdf"

TENANT_ID = UUID(
    "feeb3d7d-f2f4-4f10-9ab2-b59c6bb28358"
)


def get_tokens(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""

    return re.findall(r"\w+", text.lower())


def find_duplicated_phrases(
    abstract_text: str,
    general_chunks: list,
    phrase_size: int = 10,
) -> set[tuple[str, ...]]:
    """
    Find exact phrases of a given word length that occur
    in both the Abstract and general chunks.
    """

    abstract_tokens = get_tokens(abstract_text)

    abstract_ngrams = {
        tuple(abstract_tokens[i:i + phrase_size])
        for i in range(len(abstract_tokens) - phrase_size + 1)
    }

    general_ngrams = set()

    for general_chunk in general_chunks:
        tokens = get_tokens(general_chunk.text)

        for i in range(len(tokens) - phrase_size + 1):
            general_ngrams.add(
                tuple(tokens[i:i + phrase_size])
            )

    return abstract_ngrams.intersection(general_ngrams)


def main() -> None:
    # ---------------------------------------------------------
    # 1. Load the PDF.
    # ---------------------------------------------------------

    pages = load_pdf(str(PDF_PATH))

    # ---------------------------------------------------------
    # 2. Generate chunks using the production chunker.
    # ---------------------------------------------------------

    chunks = chunk_pages(
        pages,
        tenant_id=TENANT_ID,
    )

    abstract_chunks = [
        chunk
        for chunk in chunks
        if chunk.section_type == "abstract"
    ]

    general_chunks = [
        chunk
        for chunk in chunks
        if chunk.section_type == "general"
    ]

    # ---------------------------------------------------------
    # 3. Display chunk summary.
    # ---------------------------------------------------------

    print("\n========== CHUNK SUMMARY ==========")

    print("PDF pages:", len(pages))
    print("Total chunks:", len(chunks))
    print("General chunks:", len(general_chunks))
    print("Abstract chunks:", len(abstract_chunks))

    # ---------------------------------------------------------
    # 4. Verify the standalone Abstract.
    # ---------------------------------------------------------

    print("\n========== ABSTRACT VERIFICATION ==========")

    if len(abstract_chunks) != 1:
        print("FAILED: Expected exactly one abstract chunk.")
        return

    abstract = abstract_chunks[0]

    print("Section:", abstract.section_type)
    print("Chunk index:", abstract.chunk_index)
    print("Source pages:", abstract.source_pages)
    print("Start page:", abstract.start_page)
    print("End page:", abstract.end_page)

    print("\nAbstract text:\n")
    print(abstract.text)

    # ---------------------------------------------------------
    # 5. Verify Abstract correctness.
    # ---------------------------------------------------------

    print("\n========== ABSTRACT CHECKS ==========")

    checks = {
        "Correct source pages": abstract.source_pages == [5],

        "Correct start page": abstract.start_page == 5,

        "Correct end page": abstract.end_page == 5,

        "Contains abstract opening": (
            "Small software companies comprise"
            in abstract.text
        ),

        "Contains abstract ending": (
            "over a two year period."
            in abstract.text
        ),

        "Excludes introduction": (
            "Software has become so intrinsic"
            not in abstract.text
        ),

        "Correct section type": (
            abstract.section_type == "abstract"
        ),
    }

    for check_name, passed in checks.items():
        print(
            f"{check_name}: "
            f"{'PASSED' if passed else 'FAILED'}"
        )

    # ---------------------------------------------------------
    # 6. Check for duplicated Abstract phrases.
    # ---------------------------------------------------------

    print("\n========== DUPLICATE PHRASE CHECK ==========")

    duplicated_phrases = find_duplicated_phrases(
        abstract_text=abstract.text,
        general_chunks=general_chunks,
        phrase_size=10,
    )

    print(
        "Exact 10-word abstract phrases found in general chunks:",
        len(duplicated_phrases),
    )

    if duplicated_phrases:
        print("\nSample duplicated phrases:")

        for phrase in sorted(duplicated_phrases)[:5]:
            print(" ".join(phrase))

    else:
        print("No duplicated 10-word phrases found.")

    # ---------------------------------------------------------
    # 7. Final result.
    # ---------------------------------------------------------

    all_checks_passed = all(checks.values())

    print("\n========== FINAL RESULT ==========")

    if all_checks_passed and not duplicated_phrases:
        print("SUCCESS: Abstract chunk verification passed.")

    else:
        print("FAILED: Review the checks above.")


if __name__ == "__main__":
    main()