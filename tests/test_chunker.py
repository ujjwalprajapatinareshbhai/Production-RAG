from uuid import UUID

from app.models import Page
from app.chunker import chunk_pages, extract_abstract_chunk


TEST_TENANT_ID = UUID("22222222-2222-2222-2222-222222222222")


def make_pages() -> list[Page]:
    document_id = "test-document"

    return [
        Page(
            document_id=document_id,
            filename="sample.pdf",
            page_number=1,
            text=" ".join(f"word{i}" for i in range(1, 11)),
        ),
        Page(
            document_id=document_id,
            filename="sample.pdf",
            page_number=2,
            text=" ".join(f"word{i}" for i in range(11, 21)),
        ),
    ]


def test_chunk_count_and_overlap():
    pages = make_pages()

    chunks = chunk_pages(
        pages,
        tenant_id=TEST_TENANT_ID,
        chunk_size=8,
        overlap=2,
    )

    assert len(chunks) == 3
    assert chunks[0].text.split()[-2:] == ["word7", "word8"]
    assert chunks[1].text.split()[:2] == ["word7", "word8"]


def test_page_metadata():
    pages = make_pages()

    chunks = chunk_pages(
        pages,
        tenant_id=TEST_TENANT_ID,
        chunk_size=8,
        overlap=2,
    )

    assert chunks[0].start_page == 1
    assert chunks[0].end_page == 1
    assert chunks[1].start_page == 1
    assert chunks[1].end_page == 2
    assert chunks[2].start_page == 2
    assert chunks[2].end_page == 2


def test_invalid_overlap():
    try:
        chunk_pages(
            make_pages(),
            tenant_id=TEST_TENANT_ID,
            chunk_size=8,
            overlap=8,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid overlap")


def test_extract_abstract_chunk():
    pages = [
        Page(
            document_id="test-document",
            filename="sample.pdf",
            page_number=5,
            text=(
                "5 Abstract\n\n"
                "This report studies software testing problems.\n"
                "It proposes an improved testing process.\n\n"
                "6 Introduction\n\n"
                "Software is used in many businesses."
            ),
        )
    ]

    chunk = extract_abstract_chunk(
        pages,
        tenant_id=TEST_TENANT_ID,
    )

    assert chunk is not None
    assert chunk.tenant_id == TEST_TENANT_ID
    assert chunk.source_pages == [5]
    assert "software testing problems" in chunk.text
    assert "improved testing process" in chunk.text
    assert "Introduction" not in chunk.text
    assert "many businesses" not in chunk.text


def test_extract_abstract_returns_none_when_missing():
    pages = [
        Page(
            document_id="test-document",
            filename="sample.pdf",
            page_number=1,
            text="This page contains no abstract heading.",
        )
    ]

    assert extract_abstract_chunk(
        pages,
        tenant_id=TEST_TENANT_ID,
    ) is None


def test_extract_abstract_returns_none_when_empty():
    pages = [
        Page(
            document_id="test-document",
            filename="sample.pdf",
            page_number=5,
            text="5 Abstract\n\n6 Introduction\n\nIntroduction text.",
        )
    ]

    assert extract_abstract_chunk(
        pages,
        tenant_id=TEST_TENANT_ID,
    ) is None


if __name__ == "__main__":
    test_chunk_count_and_overlap()
    test_page_metadata()
    test_invalid_overlap()

    print("All chunker tests passed!")