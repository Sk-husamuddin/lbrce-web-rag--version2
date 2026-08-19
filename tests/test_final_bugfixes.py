from types import SimpleNamespace

from starlette.middleware.cors import CORSMiddleware

from backend.graph.nodes import (
    _DEPARTMENT_ALIASES,
    _is_current_role_query,
)
from backend.ingestion.chunker import (
    chunk_cleaned_content,
    chunk_image_document,
    chunk_parsed_document,
    chunk_pdf_document,
)


def test_bare_role_queries_use_current_role_path():
    questions = [
        "who is the HOD of AIDS?",
        "who is the HOD of CSE",
        "who is the current HOD of CSE?",
        "Head of the department of ECE",
    ]
    assert all(_is_current_role_query(question) for question in questions)


def test_csm_aliases_do_not_collide_with_plain_cse():
    assert "cse" not in _DEPARTMENT_ALIASES["cse_ai_ml"]
    assert _DEPARTMENT_ALIASES["cse_ai_ml"] == {
        "csm", "ai-ml", "ai_ml", "aiml"
    }


def test_cors_wildcard_disables_credentials():
    wildcard = ["*"]
    explicit = ["http://localhost:3000"]

    wildcard_middleware = CORSMiddleware(
        app=lambda scope, receive, send: None,
        allow_origins=wildcard,
        allow_credentials=("*" not in wildcard),
    )
    explicit_middleware = CORSMiddleware(
        app=lambda scope, receive, send: None,
        allow_origins=explicit,
        allow_credentials=("*" not in explicit),
    )

    assert wildcard_middleware.allow_credentials is False
    assert explicit_middleware.allow_credentials is True


def test_all_chunk_wrappers_preserve_metadata():
    parsed_metadata = {"parser": "html", "trace_id": "parsed"}
    parsed = SimpleNamespace(
        document_id="parsed",
        title="Parsed",
        source_url="https://lbrce.ac.in/parsed",
        content="Parsed content.",
        department="CSE",
        source_type="html",
        metadata=parsed_metadata,
    )
    assert chunk_parsed_document(parsed)[0].metadata == parsed_metadata

    pdf_metadata = {"page_number": 2, "page_count": 10}
    pdf = SimpleNamespace(
        document_id="pdf",
        title="PDF",
        source_url="https://lbrce.ac.in/test.pdf",
        content="PDF content.",
        department="CSE",
        source_type="pdf",
        metadata=pdf_metadata,
    )
    assert chunk_pdf_document(pdf)[0].metadata == pdf_metadata

    image_metadata = {"extraction_method": "ocr", "model": "test"}
    image = SimpleNamespace(
        document_id="image",
        title="Image",
        source_url="https://lbrce.ac.in/test.jpg",
        content="Image content.",
        department="CSE",
        source_type="image",
        metadata=image_metadata,
    )
    assert chunk_image_document(image)[0].metadata == image_metadata

    cleaned_metadata = {"cleaner": "test", "original_length": 42}
    cleaned = SimpleNamespace(
        document_id="cleaned",
        title="Cleaned",
        source_url="https://lbrce.ac.in/cleaned",
        cleaned_text="Cleaned content.",
        department="CSE",
        page_number=1,
        metadata=cleaned_metadata,
    )
    assert chunk_cleaned_content(cleaned)[0].metadata == cleaned_metadata
