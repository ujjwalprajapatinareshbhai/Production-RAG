from unittest.mock import patch

import pytest

from app.document_management import delete_document


DOCUMENT_ID = "a" * 64


@patch("app.document_management.refresh_bm25_index")
@patch(
    "app.document_management.delete_document_record",
    side_effect=RuntimeError("PostgreSQL unavailable"),
)
@patch("app.document_management.delete_document_chunks")
@patch("app.document_management.update_document_status")
@patch(
    "app.document_management.get_document",
    return_value={
        "document_id": DOCUMENT_ID,
        "filename": "sample.pdf",
    },
)
def test_registry_delete_failure_leaves_document_marked_failed(
    mock_get_document,
    mock_update_status,
    mock_delete_chunks,
    mock_delete_record,
    mock_refresh,
):
    with pytest.raises(RuntimeError, match="PostgreSQL unavailable"):
        delete_document(DOCUMENT_ID)

    mock_update_status.assert_called_once_with(
        DOCUMENT_ID,
        status="failed",
        error_message="Document deletion in progress.",
    )
    mock_delete_chunks.assert_called_once_with(DOCUMENT_ID)
    mock_delete_record.assert_called_once_with(DOCUMENT_ID)
    mock_refresh.assert_not_called()