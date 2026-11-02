import uuid

from fastembed import SparseTextEmbedding
from qdrant_client import models

from app.chunker import extract_abstract_chunk
from app.embeddings import embed_text
from app.pdf_loader import load_pdf
from app.vector_store import get_vector_client


PDF_PATH = "data/pdfs/Testing-in-a-Small-Company1.pdf"

COLLECTIONS = [
    "document_chunks_eval_current_v1",
    "document_chunks_eval_page_v1",
]

ABSTRACT_CHUNK_ID = (
    "157fccdd123e2e64856a7d816d9dd507"
    "6853ec353afdce092e146f602d88d239"
)

client = get_vector_client()
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")


def main():
    # Step 1: Verify both evaluation collections exist.
    for collection_name in COLLECTIONS:
        if not client.collection_exists(collection_name):
            raise RuntimeError(
                f"Required evaluation collection does not exist: "
                f"{collection_name}. Nothing was inserted."
            )

    # Step 2: Extract the abstract from the PDF.
    pages = load_pdf(PDF_PATH)
    abstract = extract_abstract_chunk(pages)

    if abstract is None:
        raise RuntimeError(
            "Could not extract the abstract. Nothing was inserted."
        )

    # Step 3: Verify that the extracted chunk is the expected abstract.
    if abstract.chunk_id != ABSTRACT_CHUNK_ID:
        raise RuntimeError(
            "The extracted abstract chunk ID differs from the expected ID. "
            "Nothing was inserted."
        )

    print("Abstract chunk ID:", abstract.chunk_id)
    print("Source pages:", abstract.source_pages)
    print("Word count:", len(abstract.text.split()))

    # Step 4: Convert the SHA-256 chunk ID to a deterministic UUID.
    # This matches the point-ID scheme used by the existing indexing script.
    point_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            abstract.chunk_id,
        )
    )

    # Step 5: Generate dense and sparse vectors.
    dense_vector = embed_text(abstract.text)
    sparse_result = next(sparse_model.embed([abstract.text]))

    # Step 6: Build the Qdrant point.
    point = models.PointStruct(
        id=point_id,
        vector={
            "": dense_vector,
            "bm25": models.SparseVector(
                indices=sparse_result.indices.tolist(),
                values=sparse_result.values.tolist(),
            ),
        },
        payload={
            "chunk_id": abstract.chunk_id,
            "document_id": abstract.document_id,
            "filename": abstract.filename,
            "chunk_index": abstract.chunk_index,
            "text": abstract.text,
            "start_page": abstract.start_page,
            "end_page": abstract.end_page,
            "source_pages": abstract.source_pages,
        },
    )

    # Step 7: Check both collections before inserting into either one.
    # If the abstract already exists anywhere, stop without inserting.
    for collection_name in COLLECTIONS:
        existing = client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )

        if existing:
            raise RuntimeError(
                f"Abstract point already exists in {collection_name}. "
                "No collections were modified by this script."
            )

    # Step 8: Insert the abstract into the two evaluation collections.
    for collection_name in COLLECTIONS:
        client.upsert(
            collection_name=collection_name,
            points=[point],
            wait=True,
        )

        print(f"Inserted abstract into: {collection_name}")

    print("\nAbstract insertion completed.")
    print("Production collection was not modified.")


if __name__ == "__main__":
    main()