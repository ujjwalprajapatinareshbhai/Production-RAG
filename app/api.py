import logging
from typing import Any
from uuid import UUID

import jwt
from contextlib import asynccontextmanager
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
from app.auth import (
    create_access_token,
    decode_access_token,
    hash_invitation_token,
    verify_password,
)
from app.config import settings
from app.db import get_connection
from app.document_management import delete_document
from app.registry import (
    accept_invitation,
    get_active_membership,
    get_document,
    get_user_by_email,
    get_user_by_id,
    initialize_registry,
    list_documents,
)
from app.search import search_documents
from app.vector_store import get_vector_client


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.

    Startup:
        Initialize the PostgreSQL document registry.

    Shutdown:
        Release application resources if needed in the future.
    """

    # Startup logic.
    initialize_registry()
    logger.info("Document registry initialized.")

    yield

    # Shutdown logic can be added here when required.


app = FastAPI(
    title="Production RAG API",
    description="Search ingested PDF documents using hybrid retrieval and reranking.",
    version="1.0.0",
    lifespan=lifespan,
)


# HTTP Bearer authentication.
# auto_error=False allows us to return our own consistent 401 response.
bearer_scheme = HTTPBearer(auto_error=False)


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

class LoginRequest(BaseModel):
    """Credentials and tenant selection for user login."""

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=12, max_length=128)
    tenant_id: UUID


class AcceptInvitationRequest(BaseModel):
    """Credentials and invitation token used to join a tenant."""

    token: str = Field(min_length=1, max_length=512)
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=12, max_length=128)


class LoginResponse(BaseModel):
    """JWT access token returned after successful login."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    tenant_id: UUID
    role: str


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

# ---------------------------------------------------------
# Authentication endpoints
# ---------------------------------------------------------

@app.post(
    "/auth/login",
    response_model=LoginResponse,
)
def login(request: LoginRequest) -> LoginResponse:
    """Authenticate a user and issue a tenant-scoped JWT."""

    # 1. Find the user by email.
    user = get_user_by_email(request.email.strip().lower())

    # 2. Verify credentials without revealing which one failed.
    if user is None or not verify_password(
        request.password,
        user["password_hash"],
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 3. Check whether the account is active.
    if user.get("is_active") is not True:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 4. Verify active membership in the selected tenant.
    membership = get_active_membership(
        tenant_id=request.tenant_id,
        user_id=user["user_id"],
    )

    if membership is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No active membership in the selected tenant.",
        )

    # 5. Read the trusted role from PostgreSQL.
    role = membership["role"]

    # 6. Generate a tenant-scoped JWT.
    access_token = create_access_token(
        user_id=str(user["user_id"]),
        tenant_id=str(request.tenant_id),
        role=role,
    )

    # 7. Return the token and its metadata.
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
        tenant_id=request.tenant_id,
        role=role,
    )


@app.post(
    "/auth/accept-invitation",
    response_model=LoginResponse,
)
def accept_invitation_endpoint(
    request: AcceptInvitationRequest,
) -> LoginResponse:
    """Authenticate an invited user and activate tenant membership."""

    # 1. Find the user by email.
    user = get_user_by_email(request.email.strip().lower())

    # 2. Authenticate the user.
    if user is None or not verify_password(
        request.password,
        user["password_hash"],
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 3. Ensure the account is active.
    if user.get("is_active") is not True:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 4. Hash the raw invitation token.
    token_hash = hash_invitation_token(request.token)

    # 5. Accept the invitation in PostgreSQL.
    try:
        invitation = accept_invitation(
            token_hash=token_hash,
            user_id=user["user_id"],
        )

    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    # 6. Get trusted tenant and role from the accepted invitation.
    tenant_id = invitation["tenant_id"]
    role = invitation["role"]

    # 7. Generate a JWT for the newly joined tenant.
    access_token = create_access_token(
        user_id=str(user["user_id"]),
        tenant_id=str(tenant_id),
        role=role,
    )

    # 8. Return the token and tenant metadata.
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
        tenant_id=tenant_id,
        role=role,
    )


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