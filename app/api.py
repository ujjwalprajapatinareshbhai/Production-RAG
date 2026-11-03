import logging
from typing import Any
from uuid import UUID

import jwt
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    status,
)
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
from pydantic import BaseModel, Field

from app.answer import answer_question
from app.auth import decode_access_token
from app.config import settings
from app.db import get_connection
from app.document_management import delete_document
from app.registry import (
    get_active_membership,
    get_document,
    get_user_by_id,
    initialize_registry,
    list_documents,
)
from app.search import search_documents
from app.vector_store import get_vector_client


logger = logging.getLogger(__name__)

app = FastAPI(
    title="Production RAG API",
    description="Search ingested PDF documents using hybrid retrieval and reranking.",
    version="1.0.0",
)

# HTTP Bearer authentication.
# auto_error=False allows us to return our own consistent 401 response.
bearer_scheme = HTTPBearer(auto_error=False)


@app.on_event("startup")
def startup_event() -> None:
    """Initialize required database tables when the API starts."""
    initialize_registry()
    logger.info("Document registry initialized.")


# ---------------------------------------------------------
# Authentication dependency
# ---------------------------------------------------------

def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        bearer_scheme
    ),
) -> dict:
    """
    Authenticate a request using JWT and verify current
    user and tenant membership status in PostgreSQL.
    """

    # Step 1: Require a Bearer token.
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # Step 2: Decode and validate the JWT.
    try:
        payload = decode_access_token(token)

        user_id = UUID(payload["sub"])
        tenant_id = UUID(payload["tenant_id"])

    except (jwt.PyJWTError, ValueError, TypeError, KeyError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Step 3: Verify that the user still exists and is active.
    user = get_user_by_id(user_id)

    if user is None or user.get("is_active") is not True:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is inactive or does not exist.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Step 4: Verify active membership in the selected tenant.
    membership = get_active_membership(
        tenant_id=tenant_id,
        user_id=user_id,
    )

    if membership is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have active membership in this tenant.",
        )

    # Step 5: Return trusted identity information.
    # The role comes from PostgreSQL, NOT the JWT.
    return {
        "user_id": user_id,
        "tenant_id": tenant_id,
        "email": user.get("email"),
        "role": membership["role"],
    }


# ---------------------------------------------------------
# Request and response models
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Document endpoints
# ---------------------------------------------------------

@app.get(
    "/documents",
    response_model=DocumentListResponse,
)
def get_documents(
    current_user: dict = Depends(get_current_user),
) -> DocumentListResponse:
    """List documents belonging only to the authenticated tenant."""

    try:
        tenant_id = current_user["tenant_id"]

        documents = list_documents(tenant_id)

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
def get_document_by_id(
    document_id: str,
    current_user: dict = Depends(get_current_user),
) -> DocumentResponse:
    """Retrieve metadata for a document inside the current tenant."""

    try:
        tenant_id = current_user["tenant_id"]

        document = get_document(
            tenant_id,
            document_id,
        )

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
    current_user: dict = Depends(get_current_user),
) -> DeleteDocumentResponse:
    """Delete a document belonging only to the authenticated tenant."""

    try:
        tenant_id = current_user["tenant_id"]

        result = delete_document(
            tenant_id,
            document_id,
        )

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
            detail=(
                "Document deletion failed. "
                "Check the document indexes and registry."
            ),
        ) from exc


# ---------------------------------------------------------
# Health and readiness endpoints
# ---------------------------------------------------------

@app.get("/health")
def health_check() -> dict[str, str]:
    """Check whether the API process is running."""

    return {"status": "ok"}


@app.get("/ready")
def readiness_check() -> dict[str, str]:
    """Check whether PostgreSQL and Qdrant are reachable."""

    try:
        # Check Qdrant.
        client = get_vector_client()

        client.get_collection(
            collection_name=settings.qdrant_collection
        )

        # Check PostgreSQL independently of any tenant.
        with get_connection() as conn:
            conn.execute("SELECT 1")

        return {"status": "ready"}

    except Exception as exc:
        logger.warning("Readiness check failed: %s", exc)

        raise HTTPException(
            status_code=503,
            detail=(
                "A required service is unavailable or "
                "not configured correctly."
            ),
        ) from exc


# ---------------------------------------------------------
# Search endpoint
# ---------------------------------------------------------

@app.post(
    "/search",
    response_model=SearchResponse,
)
def search(
    request: SearchRequest,
    current_user: dict = Depends(get_current_user),
) -> SearchResponse:
    """Search documents belonging only to the authenticated tenant."""

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=422,
            detail="Query cannot be empty or contain only whitespace.",
        )

    try:
        results = search_documents(
            tenant_id=current_user["tenant_id"],
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


# ---------------------------------------------------------
# Question-answering endpoint
# ---------------------------------------------------------

@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask(
    request: AskRequest,
    current_user: dict = Depends(get_current_user),
) -> AskResponse:
    """Answer questions using the current tenant's PDF documents."""

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty or contain only whitespace.",
        )

    try:
        result = answer_question(
            tenant_id=current_user["tenant_id"],
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