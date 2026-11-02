import logging
from typing import Any
from app.document_management import delete_document
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.registry import (
    get_document,
    initialize_registry,
    list_completed_document_ids,
    list_documents,
)
from app.search import search_documents
from app.vector_store import get_vector_client
from app.answer import answer_question


logger = logging.getLogger(__name__)

app = FastAPI(
    title="Production RAG API",
    description="Search ingested PDF documents using hybrid retrieval and reranking.",
    version="1.0.0",
)

@app.on_event("startup")
def startup_event() -> None:
    """Initialize required database tables when the API starts."""
    initialize_registry()
    logger.info("Document registry initialized.")


class SearchRequest(BaseModel):
    """Input provided by a client searching the document collection."""

    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=3, ge=1, le=20)
    candidate_limit: int = Field(default=10, ge=1, le=100)
    document_id: str | None = None


class SearchResponse(BaseModel):
    """Retrieved passages and their metadata."""

    query: str
    count: int
    results: list[dict[str, Any]]

class AskRequest(BaseModel):
    """Input for document question answering."""

    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=10)
    candidate_limit: int = Field(default=20, ge=1, le=100)
    document_id: str | None = None


class AskResponse(BaseModel):
    """Generated answer and its supporting source passages."""

    question: str
    answer: str
    sources: list[dict[str, Any]]
    insufficient_evidence: bool

class DeleteDocumentResponse(BaseModel):
    """Result of deleting a document."""

    document_id: str
    filename: str
    status: str

class DocumentResponse(BaseModel):
    """Metadata for a registered document."""

    document_id: str
    filename: str
    status: str
    error_message: str | None = None
    created_at: Any
    updated_at: Any


class DocumentListResponse(BaseModel):
    """Registered documents and their total count."""

    count: int
    documents: list[DocumentResponse]


@app.get("/documents", response_model=DocumentListResponse)
def get_documents() -> DocumentListResponse:
    """List all registered documents, regardless of status."""

    try:
        documents = list_documents()

        return DocumentListResponse(
            count=len(documents),
            documents=documents,
        )

    except Exception as exc:
        logger.exception("Listing documents failed.")

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve registered documents.",
        ) from exc


@app.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
)
def get_document_by_id(document_id: str) -> DocumentResponse:
    """Retrieve metadata for one registered document."""

    try:
        document = get_document(document_id)

        if document is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        return DocumentResponse(**document)

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Retrieving document failed.")

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve the document.",
        ) from exc

@app.delete(
    "/documents/{document_id}",
    response_model=DeleteDocumentResponse,
)
def delete_document_by_id(
    document_id: str,
) -> DeleteDocumentResponse:
    """Delete a document and its indexed chunks."""

    try:
        result = delete_document(document_id)
        return DeleteDocumentResponse(**result)

    except ValueError as exc:
        status_code = (
            404 if str(exc) == "Document not found." else 400
        )
        raise HTTPException(
            status_code=status_code,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Document deletion failed.")

        raise HTTPException(
            status_code=500,
            detail="Document deletion failed. Check the document indexes and registry.",
        ) from exc

@app.get("/health")
def health_check() -> dict[str, str]:
    """Check whether the API process is running."""

    return {"status": "ok"}


@app.get("/ready")
def readiness_check() -> dict[str, str]:
    """Check whether the API's database and Qdrant dependencies are reachable."""

    try:
        client = get_vector_client()

        # Confirm that the configured Qdrant collection exists.
        client.get_collection(
            collection_name=settings.qdrant_collection
        )

        # Access PostgreSQL through the existing document registry.
        list_completed_document_ids()

        return {"status": "ready"}

    except Exception as exc:
        logger.warning("Readiness check failed: %s", exc)

        raise HTTPException(
            status_code=503,
            detail="A required service is unavailable or not configured correctly.",
        ) from exc


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Retrieve and rank relevant passages from completed documents."""

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=422,
            detail="Query cannot be empty or contain only whitespace.",
        )

    try:
        results = search_documents(
            query=query,
            limit=request.limit,
            candidate_limit=request.candidate_limit,
            document_id=request.document_id,
        )

        return SearchResponse(
            query=query,
            count=len(results),
            results=results,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Document search failed.")

        raise HTTPException(
            status_code=500,
            detail="Search failed due to an internal server error.",
        ) from exc


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question using retrieved PDF passages."""

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty or contain only whitespace.",
        )

    try:
        result = answer_question(
            query=query,
            limit=request.limit,
            candidate_limit=request.candidate_limit,
            document_id=request.document_id,
        )

        return AskResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Question answering failed.")

        raise HTTPException(
            status_code=502,
            detail=(
                "Question answering failed. Check that Ollama is running "
                "and the configured model is available."
            ),
        ) from exc