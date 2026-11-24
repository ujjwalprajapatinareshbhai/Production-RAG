
import sys
import uuid
from pathlib import Path
from uuid import UUID

from qdrant_client import models


# Make the project root importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from app.config import settings
from app.ingestion import prepare_document
from app.bm25_retriever import refresh_bm25_index
from app.vector_store import (
    get_vector_client,
    upsert_chunks,
)


PDF_PATH = PROJECT_ROOT / "data" / "pdfs" / (
    "Testing-in-a-Small-Company1.pdf"
)

TENANT_ID = UUID(
    "feeb3d7d-f2f4-4f10-9ab2-b59c6bb28358"
)

EXPECTED_OLD_COUNT = 86
EXPECTED_NEW_COUNT = 87
MAX_EXPECTED_COUNT = EXPECTED_OLD_COUNT + EXPECTED_NEW_COUNT


def get_document_filter(document_id: str):
    """Build a filter for one tenant and one document."""

    return models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(
                    value=str(TENANT_ID)
                ),
            ),
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(
                    value=document_id
                ),
            ),
        ]
    )


def get_all_document_points(client, document_filter):
    """Retrieve every Qdrant point for this document."""

    all_points = []
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=document_filter,
            limit=100,
            offset=offset,
            with_payload=[
                "chunk_id",
                "chunk_index",
                "section_type",
                "source_pages",
            ],
            with_vectors=False,
        )

        all_points.extend(points)

        if next_offset is None:
            break

        offset = next_offset

    return all_points


def main():
    print("\n========== STEP 1: PREPARE DOCUMENT ==========")

    chunks = prepare_document(
        str(PDF_PATH),
        tenant_id=TENANT_ID,
    )

    if not chunks:
        raise RuntimeError("No chunks were generated.")

    document_ids = {
        chunk.document_id for chunk in chunks
    }

    if len(document_ids) != 1:
        raise RuntimeError(
            "Prepared chunks contain multiple document IDs."
        )

    document_id = chunks[0].document_id

    abstract_chunks = [
        chunk
        for chunk in chunks
        if chunk.section_type == "abstract"
    ]

    print("Document ID:", document_id)
    print("Prepared chunks:", len(chunks))
    print("Abstract chunks:", len(abstract_chunks))

    if len(chunks) != EXPECTED_NEW_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_NEW_COUNT} chunks, "
            f"got {len(chunks)}."
        )

    if len(abstract_chunks) != 1:
        raise RuntimeError(
            "Expected exactly one standalone abstract chunk."
        )

    if abstract_chunks[0].source_pages != [5]:
        raise RuntimeError(
            "Abstract chunk must come from page 5."
        )

    print("\n========== STEP 2: VERIFY EXISTING POINTS ==========")

    client = get_vector_client()
    document_filter = get_document_filter(document_id)

    existing_points = get_all_document_points(
        client,
        document_filter,
    )

    existing_count = len(existing_points)

    print("Existing Qdrant points:", existing_count)

    # Accept the original state, the fully inserted state,
    # or a partially cleaned replacement state.
    allowed_count = (
        existing_count == EXPECTED_OLD_COUNT
        or EXPECTED_NEW_COUNT <= existing_count <= MAX_EXPECTED_COUNT
    )

    if not allowed_count:
        raise RuntimeError(
            f"Unexpected point count: {existing_count}. "
            f"Expected {EXPECTED_OLD_COUNT} original points "
            f"or between {EXPECTED_NEW_COUNT} and "
            f"{MAX_EXPECTED_COUNT} replacement points. "
            "Aborting."
        )

    print("\n========== STEP 3: CALCULATE NEW POINT IDs ==========")

    expected_point_ids = {
        str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                chunk.chunk_id,
            )
        )
        for chunk in chunks
    }

    if len(expected_point_ids) != len(chunks):
        raise RuntimeError(
            "Duplicate point IDs detected in prepared chunks."
        )

    print("Expected new point IDs:", len(expected_point_ids))

    print("\n========== STEP 4: UPSERT NEW CHUNKS ==========")

    stored_count = upsert_chunks(chunks)

    print("Upserted chunks:", stored_count)

    if stored_count != EXPECTED_NEW_COUNT:
        raise RuntimeError(
            "Upsert count does not match expected chunk count."
        )

    print("\n========== STEP 5: VERIFY NEW POINTS ==========")

    current_points = get_all_document_points(
        client,
        document_filter,
    )

    actual_point_ids = {
        str(point.id)
        for point in current_points
    }

    missing_ids = expected_point_ids - actual_point_ids
    stale_ids = actual_point_ids - expected_point_ids

    print("Total points currently stored:", len(current_points))
    print("Expected new points:", len(expected_point_ids))
    print("Missing new points:", len(missing_ids))
    print("Obsolete old points:", len(stale_ids))

    if missing_ids:
        raise RuntimeError(
            "Some new points are missing. "
            "Old points have NOT been deleted."
        )

    if len(stale_ids) > EXPECTED_OLD_COUNT:
        raise RuntimeError(
            "More obsolete points than expected. "
            "Refusing to delete."
        )

    print("\n========== ABSTRACT POINT VERIFICATION ==========")

    abstract_points = [
        point
        for point in current_points
        if point.payload.get("section_type") == "abstract"
    ]

    for point in abstract_points:
        print("Point ID:", point.id)
        print("Chunk index:", point.payload.get("chunk_index"))
        print("Source pages:", point.payload.get("source_pages"))

    # Confirm the expected new abstract point is present.
    expected_abstract_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            abstract_chunks[0].chunk_id,
        )
    )

    if expected_abstract_id not in actual_point_ids:
        raise RuntimeError(
            "The new abstract point is missing. "
            "Refusing to delete old points."
        )

    print("\n========== STEP 6: DELETE OBSOLETE POINTS ==========")

    print("Obsolete point IDs to delete:", len(stale_ids))

    if stale_ids:
        client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=models.PointIdsList(
                points=list(stale_ids)
            ),
            wait=True,
        )

    print("Obsolete points deleted.")

    print("\n========== STEP 7: FINAL VERIFICATION ==========")

    final_points = get_all_document_points(
        client,
        document_filter,
    )

    final_point_ids = {
        str(point.id)
        for point in final_points
    }

    final_abstract_points = [
        point
        for point in final_points
        if point.payload.get("section_type") == "abstract"
    ]

    missing_final_ids = expected_point_ids - final_point_ids
    unexpected_final_ids = final_point_ids - expected_point_ids

    print("Final Qdrant point count:", len(final_points))
    print("Expected point count:", EXPECTED_NEW_COUNT)
    print("Missing point IDs:", len(missing_final_ids))
    print("Unexpected point IDs:", len(unexpected_final_ids))
    print("Abstract point count:", len(final_abstract_points))

    if len(final_points) != EXPECTED_NEW_COUNT:
        raise RuntimeError(
            "Final Qdrant point count is incorrect."
        )

    if missing_final_ids or unexpected_final_ids:
        raise RuntimeError(
            "Final Qdrant point IDs do not match."
        )

    if len(final_abstract_points) != 1:
        raise RuntimeError(
            "Expected exactly one abstract point."
        )

    abstract_payload = final_abstract_points[0].payload

    if abstract_payload.get("source_pages") != [5]:
        raise RuntimeError(
            "Abstract point does not have the expected source page."
        )

    if abstract_payload.get("section_type") != "abstract":
        raise RuntimeError(
            "Abstract point has an incorrect section type."
        )

    print(
        "Abstract source pages:",
        abstract_payload.get("source_pages"),
    )

    print("\n========== STEP 8: REFRESH BM25 INDEX ==========")

    try:
        refresh_bm25_index(TENANT_ID)
        print("BM25 index refreshed successfully.")
    except Exception as exc:
        print("WARNING: BM25 refresh failed:", repr(exc))
        print(
            "Qdrant replacement completed, "
            "but BM25 refresh needs attention."
        )

    print("\n========== FINAL RESULT ==========")
    print("SUCCESS: Document replacement completed.")
    print("Final chunks:", len(final_points))
    print("Final abstract chunks:", len(final_abstract_points))


if __name__ == "__main__":
    main()
