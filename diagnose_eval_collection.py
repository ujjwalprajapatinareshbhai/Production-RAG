from app.vector_store import get_vector_client

COLLECTION_NAME = "document_chunks_eval_abstract"
TARGET = "current software process used in company x"

client = get_vector_client()

# Check collection metadata.
info = client.get_collection(COLLECTION_NAME)

print("Collection:", COLLECTION_NAME)
print("Stored points:", info.points_count)

# Read every point from the evaluation collection.
offset = None
total_scanned = 0
matches = []

while True:
    points, next_offset = client.scroll(
        collection_name=COLLECTION_NAME,
        offset=offset,
        limit=100,
        with_payload=True,
        with_vectors=False,
    )

    total_scanned += len(points)

    for point in points:
        payload = point.payload or {}
        text = payload.get("text", "")

        if TARGET in text.lower():
            matches.append(
                {
                    "id": point.id,
                    "filename": payload.get("filename"),
                    "document_id": payload.get("document_id"),
                    "source_pages": payload.get("source_pages"),
                    "section_type": payload.get("section_type"),
                    "text": text,
                }
            )

    if next_offset is None:
        break

    offset = next_offset

print("\nTotal points scanned:", total_scanned)
print("Exact-statement matches:", len(matches))

for match in matches:
    print("\n--- MATCH ---")
    print("Point ID:", match["id"])
    print("Filename:", match["filename"])
    print("Document ID:", match["document_id"])
    print("Source pages:", match["source_pages"])
    print("Section type:", match["section_type"])
    print("Text:", match["text"][:1500])
