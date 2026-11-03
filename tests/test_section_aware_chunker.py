import re
from uuid import UUID

from app.chunker import chunk_pages_by_section
from app.models import Page


TEST_TENANT_ID = UUID("22222222-2222-2222-2222-222222222222")

DOCUMENT_ID = "a" * 64
FILENAME = "test_document.pdf"


def make_page(page_number: int, text: str) -> Page:
    return Page(
        document_id=DOCUMENT_ID,
        filename=FILENAME,
        page_number=page_number,
        text=text,
    )


def test_skips_table_of_contents_and_preserves_abstract():
    pages = [
        make_page(1, "Test Document\nAuthor Name"),
        make_page(
            2,
            "Table of Contents\n"
            "1 Introduction ........ 1\n"
            "2 Results ........ 10",
        ),
        make_page(
            3,
            "3 Conclusion ........ 20\n"
            "Table of Figures",
        ),
        make_page(
            4,
            "Abstract\n"
            "This study examines software testing "
            "practices in a small company.",
        ),
        make_page(
            5,
            "1 Introduction\n"
            "Software testing is important for quality.",
        ),
    ]

    chunks = chunk_pages_by_section(
        pages,
        tenant_id=TEST_TENANT_ID,
        chunk_size=100,
        overlap=10,
    )

    assert all(chunk.tenant_id == TEST_TENANT_ID for chunk in chunks)

    abstract_chunks = [
        chunk for chunk in chunks
        if chunk.text.startswith("Abstract\n")
    ]

    assert len(abstract_chunks) == 1
    assert abstract_chunks[0].source_pages == [4]

    all_text = "\n".join(chunk.text for chunk in chunks)

    assert "Table of Contents" not in all_text
    assert "Table of Figures" not in all_text
    assert "1 Introduction ........ 1" not in all_text

    # The title page is preserved separately from the abstract.
    title_chunks = [
        chunk for chunk in chunks
        if "Test Document" in chunk.text
    ]

    assert len(title_chunks) == 1
    assert 1 in title_chunks[0].source_pages
    assert 4 not in title_chunks[0].source_pages


def test_numbered_list_items_do_not_start_new_sections():
    pages = [
        make_page(
            1,
            "2.4.1 Testing Skills and Knowledge of Developers\n"
            "1. The developers did not receive rigorous training.\n"
            "2. No further training courses were attended.",
        ),
    ]

    chunks = chunk_pages_by_section(
        pages,
        tenant_id=TEST_TENANT_ID,
        chunk_size=100,
        overlap=10,
    )

    assert len(chunks) == 1
    assert chunks[0].tenant_id == TEST_TENANT_ID

    text = chunks[0].text

    assert text.startswith(
        "2.4.1 Testing Skills and Knowledge of Developers"
    )
    assert "1. The developers did not receive rigorous training." in text
    assert "2. No further training courses were attended." in text

    # No chunk should begin with a numbered list item as its heading.
    assert not re.match(r"^\s*\d+\.\s+[A-Z]", text)


def test_page_metadata_tracks_paragraphs_across_pages():
    pages = [
        make_page(
            1,
            "1 Introduction\n"
            "Software testing improves reliability and quality",
        ),
        make_page(
            2,
            "by helping developers discover defects before release.",
        ),
    ]

    chunks = chunk_pages_by_section(
        pages,
        tenant_id=TEST_TENANT_ID,
        chunk_size=100,
        overlap=0,
    )

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.tenant_id == TEST_TENANT_ID
    assert chunk.source_pages == [1, 2]
    assert chunk.start_page == 1
    assert chunk.end_page == 2
    assert "reliability and quality" in chunk.text
    assert "discover defects before release" in chunk.text


def test_chunks_respect_word_limit_and_have_unique_text():
    pages = [
        make_page(
            1,
            "1 Introduction\n"
            + " ".join(f"word{i}" for i in range(2500)),
        ),
    ]

    chunks = chunk_pages_by_section(
        pages,
        tenant_id=TEST_TENANT_ID,
        chunk_size=100,
        overlap=10,
    )

    assert len(chunks) > 1
    assert all(chunk.tenant_id == TEST_TENANT_ID for chunk in chunks)
    assert all(len(chunk.text.split()) <= 100 for chunk in chunks)
    assert len({chunk.text for chunk in chunks}) == len(chunks)


def test_invalid_chunk_settings_are_rejected():
    pages = [make_page(1, "1 Introduction\nSome content.")]

    for chunk_size, overlap in [
        (0, 0),
        (100, -1),
        (100, 100),
    ]:
        try:
            chunk_pages_by_section(
                pages,
                tenant_id=TEST_TENANT_ID,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Expected ValueError for "
                f"chunk_size={chunk_size}, overlap={overlap}"
            )