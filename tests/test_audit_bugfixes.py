from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

from starlette.middleware.cors import CORSMiddleware

from backend.embedding.embedding_generator import EmbeddingGenerator
from backend.graph.nodes import _DEPARTMENT_ALIASES, _is_current_role_query
from backend.ingestion.chunker import (
    chunk_cleaned_content,
    chunk_image_document,
    chunk_parsed_document,
    chunk_pdf_document,
)
from backend.retrieval.rag import RAGPipeline


def test_bare_role_queries_are_current_role_queries():
    questions = [
        "who is the HOD of AIDS?",
        "who is the HOD of CSE",
        "who is the current HOD of CSE?",
        "Head of the department of ECE",
    ]
    assert all(_is_current_role_query(question) for question in questions)


def test_csm_aliases_are_isolated_from_plain_cse():
    assert _DEPARTMENT_ALIASES["cse_ai_ml"] == {
        "csm", "ai-ml", "ai_ml", "aiml"
    }
    assert "cse" not in _DEPARTMENT_ALIASES["cse_ai_ml"]


def test_mock_embeddings_are_stable_across_processes():
    script = (
        "from backend.embedding.embedding_generator import EmbeddingGenerator; "
        "print(EmbeddingGenerator(api_key='mock_pinecone_key').embed('same text'))"
    )
    first = subprocess.check_output([sys.executable, "-c", script], text=True)
    second = subprocess.check_output([sys.executable, "-c", script], text=True)
    assert first == second


def test_legacy_rag_mock_handles_missing_chunk_text():
    pipeline = RAGPipeline(object(), object(), api_key="mock_openrouter_key")
    with patch(
        "backend.retrieval.retrieve",
        return_value=[{"title": "No text field", "source_url": "https://lbrce.ac.in"}],
    ):
        result = pipeline.generate_answer("test query")
    assert result["answer"] == "Mock response based on context: ..."


def test_legacy_rag_mock_handles_empty_results():
    pipeline = RAGPipeline(object(), object(), api_key="mock_openrouter_key")
    with patch("backend.retrieval.retrieve", return_value=[]):
        result = pipeline.generate_answer("test query")
    assert result["sources"] == []
    assert "couldn't find reliable information" in result["answer"]


def test_mock_key_detection_is_exact():
    assert "mock_openrouter_key" in {"mock_openrouter_key", "mock-openrouter", "mock_openrouter"}
    pipeline = RAGPipeline(object(), object(), api_key="real-mock-looking-key")
    with patch(
        "backend.retrieval.retrieve",
        return_value=[{"chunk_text": "Evidence", "source_url": "https://lbrce.ac.in"}],
    ), patch.object(pipeline.client.chat.completions, "create") as create:
        create.return_value.choices = [
            SimpleNamespace(message=SimpleNamespace(content="Generated answer"))
        ]
        result = pipeline.generate_answer("test query")
    create.assert_called_once()
    assert result["answer"] == "Generated answer"


def test_cors_wildcard_and_explicit_origins():
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
        document_id="parsed", title="Parsed", source_url="https://lbrce.ac.in/parsed",
        content="Parsed content.", department="CSE", source_type="html",
        metadata=parsed_metadata,
    )
    assert chunk_parsed_document(parsed)[0].metadata == parsed_metadata

    pdf_metadata = {"page_number": 2, "page_count": 10}
    pdf = SimpleNamespace(
        document_id="pdf", title="PDF", source_url="https://lbrce.ac.in/test.pdf",
        content="PDF content.", department="CSE", source_type="pdf",
        metadata=pdf_metadata,
    )
    assert chunk_pdf_document(pdf)[0].metadata == pdf_metadata

    image_metadata = {"extraction_method": "ocr", "model": "test"}
    image = SimpleNamespace(
        document_id="image", title="Image", source_url="https://lbrce.ac.in/test.jpg",
        content="Image content.", department="CSE", source_type="image",
        metadata=image_metadata,
    )
    assert chunk_image_document(image)[0].metadata == image_metadata

    cleaned_metadata = {"cleaner": "test", "original_length": 42}
    cleaned = SimpleNamespace(
        document_id="cleaned", title="Cleaned", source_url="https://lbrce.ac.in/cleaned",
        cleaned_text="Cleaned content.", department="CSE", page_number=1,
        metadata=cleaned_metadata,
    )
    assert chunk_cleaned_content(cleaned)[0].metadata == cleaned_metadata
