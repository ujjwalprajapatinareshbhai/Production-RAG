from qdrant_client import QdrantClient

COLLECTION = "document_chunks_eval_abstract"

ABSTRACT_CHUNK_ID = (
    "157fccdd123e2e64856a7d816d9dd5076853ec353afdce092e146f602d88d239"
)


def main():
    client = QdrantClient(url="http://localhost:6333")

    points, _ = client.scroll(
        collection_name=COLLECTION,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )

    abstract_point = next(
        (
            point
            for point in points
            if point.payload.get("chunk_id") == ABSTRACT_CHUNK_ID
        ),
        None,
    )

    if abstract_point is None:
        raise RuntimeError("Abstract chunk not found; nothing was changed.")

    client.set_payload(
        collection_name=COLLECTION,
        payload={"section_type": "abstract"},
        points=[abstract_point.id],
    )

    print("Updated evaluation collection only.")
    print("Chunk ID:", ABSTRACT_CHUNK_ID)
    print("Section type: abstract")


if __name__ == "__main__":
    main()