from app.context_selector import select_context


def make_chunk(
    text: str,
    pages: list[int],
    section_type: str = "body",
) -> dict:
    return {
        "text": text,
        "source_pages": pages,
        "section_type": section_type,
        "reranker_score": 1.0,
    }


def main():
    # Simulated reranked results: body passages rank above the abstract.
    ranked_chunks = [
        make_chunk("Detailed testing practices.", [58]),
        make_chunk("Discussion of non-functional testing.", [63]),
        make_chunk(
            "The report proposes a practical, basic test process "
            "to improve software quality at Company X.",
            [5],
            "abstract",
        ),
    ]

    overview_query = "Why did the researchers conduct this study?"
    detail_query = "What testing practices were discussed?"

    # Test 1: Overview question should prioritize the abstract.
    overview_context = select_context(
        query=overview_query,
        ranked_chunks=ranked_chunks,
        limit=3,
    )

    print("\n" + "=" * 60)
    print("TEST 1: OVERVIEW QUESTION")
    print("Selected pages:", [
        chunk["source_pages"] for chunk in overview_context
    ])

    # Test 2: Specific-detail question should preserve ranking.
    detail_context = select_context(
        query=detail_query,
        ranked_chunks=ranked_chunks,
        limit=3,
    )

    print("\n" + "=" * 60)
    print("TEST 2: SPECIFIC-DETAIL QUESTION")
    print("Selected pages:", [
        chunk["source_pages"] for chunk in detail_context
    ])

    # Test 3: Overview question when the abstract is unavailable.
    chunks_without_abstract = [
        chunk
        for chunk in ranked_chunks
        if chunk.get("section_type") != "abstract"
    ]

    fallback_context = select_context(
        query=overview_query,
        ranked_chunks=chunks_without_abstract,
        limit=3,
    )

    print("\n" + "=" * 60)
    print("TEST 3: OVERVIEW WITHOUT ABSTRACT")
    print("Selected pages:", [
        chunk["source_pages"] for chunk in fallback_context
    ])

    # Verify the expected behavior.
    assert overview_context[0]["section_type"] == "abstract"
    assert [
        chunk["source_pages"] for chunk in detail_context
    ] == [[58], [63], [5]]
    assert [
        chunk["source_pages"] for chunk in fallback_context
    ] == [[58], [63]]

    print("\nAll 3 context-selection tests passed!")


if __name__ == "__main__":
    main()