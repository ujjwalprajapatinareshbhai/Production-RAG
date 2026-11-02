
from sentence_transformers import SentenceTransformer
from app.config import settings

# Load the embedding model once, when this module is imported.
MODEL_NAME = settings.dense_model

_model = SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> list[float]:
    """Convert one text string into a 384-dimensional embedding."""
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    vector = _model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return vector.tolist()

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert multiple texts into embedding vectors in one call."""
    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("Texts cannot contain empty strings.")

    vectors = _model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=False,
    )

    return vectors.tolist()
