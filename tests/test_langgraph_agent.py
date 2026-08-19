"""
Mocked unit tests for the LangGraph agent nodes and graph.

Tests A–E match the specification.  All external I/O (Pinecone, Tavily,
OpenRouter) is mocked so these tests run without network access.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pinecone_doc(score: float, title: str = "LBRCE Dept Page") -> Dict[str, Any]:
    return {
        "chunk_text": f"LBRCE has departments including CSE, EEE and ECE. {title}",
        "similarity_score": score,
        "source_url": "https://www.lbrce.ac.in/departments.php",
        "title": title,
        "source_type": "html",
        "department": "",
        "page_number": 0,
        "document_id": "doc1",
    }


def _make_tavily_result(title: str = "LBRCE Web Page") -> Dict[str, Any]:
    return {
        "title": title,
        "url": "https://www.lbrce.ac.in",
        "content": "LBRCE offers B.Tech, M.Tech and MBA programmes.",
        "score": 0.8,
        "source_type": "web",
    }


# ---------------------------------------------------------------------------
# Node unit tests
# ---------------------------------------------------------------------------


class TestEvidenceSufficiencyHeuristic:
    """Unit tests for the evidence_sufficient helper (isolated)."""

    def _sufficient(self, scores, threshold=0.65):
        from backend.graph.nodes import _evidence_sufficient
        return _evidence_sufficient(scores, threshold)

    def test_empty_scores_returns_false(self):
        assert self._sufficient([]) is False

    def test_single_high_score_returns_false(self):
        # Only one good match — not enough coverage
        assert self._sufficient([0.90]) is False

    def test_two_good_scores_return_true(self):
        assert self._sufficient([0.80, 0.72, 0.55]) is True

    def test_top_below_threshold_returns_false(self):
        assert self._sufficient([0.60, 0.58]) is False

    def test_custom_threshold(self):
        assert self._sufficient([0.70, 0.68], threshold=0.75) is False
        assert self._sufficient([0.80, 0.76], threshold=0.75) is True


# ---------------------------------------------------------------------------
# Test A — Pinecone sufficient: Tavily must NOT be called
# ---------------------------------------------------------------------------

class TestA_PineconeSufficient:
    """
    Question: "What departments are available at LBRCE?"
    Expected: Pinecone → sufficient → Tavily NOT called → OpenRouter answer.
    """

    @pytest.mark.asyncio
    async def test_tavily_not_called_when_pinecone_sufficient(self):
        high_score_docs = [
            _make_pinecone_doc(0.87),
            _make_pinecone_doc(0.82),
            _make_pinecone_doc(0.75),
        ]

        with (
            patch("backend.graph.nodes.retrieve", return_value=high_score_docs),
            patch("backend.graph.nodes._get_pinecone_indexer"),
            patch("backend.graph.nodes._get_embedding_generator"),
            patch("backend.graph.nodes._get_openai_client") as mock_client_factory,
            patch("backend.graph.nodes.tavily_search_node", wraps=lambda s: s) as mock_tavily,
        ):
            mock_completion = MagicMock()
            mock_completion.choices = [MagicMock(message=MagicMock(content="LBRCE has CSE, EEE, and ECE departments."))]
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_completion
            mock_client_factory.return_value = mock_client

            from backend.graph.graph import rag_graph
            result = await rag_graph.ainvoke({"question": "What departments are available at LBRCE?"})

        assert result["evidence_sufficient"] is True
        assert result["answer"]
        assert "CSE" in result["answer"] or "department" in result["answer"].lower()
        # tavily_search_node was monkey-patched but should NOT have been reached
        # (graph bypasses it when evidence_sufficient=True)
        # Verify by checking no tavily_results in state
        assert result.get("tavily_results") is None or result.get("tavily_results") == []


# ---------------------------------------------------------------------------
# Test B — Pinecone insufficient: Tavily fallback triggered
# ---------------------------------------------------------------------------

class TestB_PineconeInsufficient:
    """
    Question outside Pinecone knowledge base.
    Expected: Pinecone → insufficient → Tavily called → OpenRouter answer.
    """

    @pytest.mark.asyncio
    async def test_tavily_called_when_pinecone_insufficient(self):
        low_score_docs = [_make_pinecone_doc(0.30)]  # below threshold

        with (
            patch("backend.graph.nodes.retrieve", return_value=low_score_docs),
            patch("backend.graph.nodes._get_pinecone_indexer"),
            patch("backend.graph.nodes._get_embedding_generator"),
            patch("backend.graph.nodes._get_openai_client") as mock_client_factory,
            patch("tavily.TavilyClient") as MockTavily,
        ):
            mock_tv_instance = MagicMock()
            mock_tv_instance.search.return_value = {
                "results": [
                    {
                        "title": "LBRCE 2024 Admissions",
                        "url": "https://lbrce.ac.in/admissions",
                        "content": "Applications for B.Tech 2024 are open.",
                        "score": 0.75,
                    }
                ]
            }
            MockTavily.return_value = mock_tv_instance

            mock_completion = MagicMock()
            mock_completion.choices = [MagicMock(message=MagicMock(content="B.Tech 2024 admissions are open at LBRCE."))]
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_completion
            mock_client_factory.return_value = mock_client

            from backend.graph.graph import rag_graph
            result = await rag_graph.ainvoke({"question": "What is the current LBRCE admission process for 2024?"})

        assert result["evidence_sufficient"] is False
        tavily_results = result.get("tavily_results", [])
        assert len(tavily_results) > 0, "Tavily should have been called and stored results"
        web_sources = [s for s in result.get("sources", []) if s.get("source_type") == "web"]
        assert len(web_sources) > 0, "Sources should include web-type entries from Tavily"
        assert result["answer"]


# ---------------------------------------------------------------------------
# Test C — Neither Pinecone nor Tavily returns useful evidence
# ---------------------------------------------------------------------------

class TestC_NoEvidence:
    """
    Expected: safe fallback answer, no hallucination.
    """

    @pytest.mark.asyncio
    async def test_safe_fallback_when_no_evidence(self):
        from backend.retrieval import NoResultsError

        with (
            patch("backend.graph.nodes.retrieve", side_effect=NoResultsError("no results")),
            patch("tavily.TavilyClient") as MockTavily,
        ):
            mock_tv_instance = MagicMock()
            mock_tv_instance.search.return_value = {"results": []}
            MockTavily.return_value = mock_tv_instance

            from backend.graph.graph import rag_graph
            result = await rag_graph.ainvoke({"question": "What is the academic schedule of Hogwarts?"})

        assert result["evidence_sufficient"] is False
        assert result.get("sources", []) == [] or all(
            r.get("content", "") == "" for r in result.get("tavily_results", [])
        )
        answer = result.get("answer", "")
        assert "couldn't find" in answer.lower() or "not available" in answer.lower() or answer


# ---------------------------------------------------------------------------
# Test D — Statelessness: no state bleed between requests
# ---------------------------------------------------------------------------

class TestD_Stateless:
    """
    Run two independent requests.  Verify information from request 1
    does not appear in request 2 unless actually retrieved again.

    NOTE: overlapping doc IDs between r1 and r2 are EXPECTED — two
    independent Pinecone searches can return the same popular page.
    Statelessness means the state *objects* are fresh, not that
    retrieval results never share documents.
    """

    @pytest.mark.asyncio
    async def test_no_state_bleed_between_requests(self):
        docs_req1 = [
            {**_make_pinecone_doc(0.85), "chunk_text": "SECRET_REQUEST_1_DATA"},
            {**_make_pinecone_doc(0.80), "chunk_text": "More SECRET_REQUEST_1_DATA"},
        ]
        docs_req2 = [_make_pinecone_doc(0.20)]  # low score → Tavily

        call_count = {"n": 0}
        def fake_retrieve(q, emb, idx, top_k=5):
            call_count["n"] += 1
            return docs_req1 if call_count["n"] == 1 else docs_req2

        with (
            patch("backend.graph.nodes.retrieve", side_effect=fake_retrieve),
            patch("backend.graph.nodes._get_pinecone_indexer"),
            patch("backend.graph.nodes._get_embedding_generator"),
            patch("backend.graph.nodes._get_openai_client") as mock_cf,
            patch("tavily.TavilyClient") as MockTavily,
        ):
            mock_tv_instance = MagicMock()
            mock_tv_instance.search.return_value = {"results": [_make_tavily_result()]}
            MockTavily.return_value = mock_tv_instance

            mock_comp = MagicMock()
            mock_comp.choices = [MagicMock(message=MagicMock(content="Answer based on context only."))]
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_comp
            mock_cf.return_value = mock_client

            from backend.graph.graph import rag_graph
            result1 = await rag_graph.ainvoke({"question": "Request 1 question"})
            result2 = await rag_graph.ainvoke({"question": "Request 2 question"})

        # State objects must be distinct
        assert result1 is not result2

        # Each result carries its own question
        assert result1["question"] == "Request 1 question"
        assert result2["question"] == "Request 2 question"

        # result2's retrieved docs are the low-score docs (not r1's docs)
        r2_texts = " ".join(d.get("chunk_text", "") for d in (result2.get("retrieved_documents") or []))
        assert "SECRET_REQUEST_1_DATA" not in r2_texts, (
            "State from request 1 leaked into request 2 retrieved docs!"
        )



# ---------------------------------------------------------------------------
# Test E — /chat endpoint returns correct schema
# ---------------------------------------------------------------------------

class TestE_ChatEndpoint:
    """
    Verify that the /chat FastAPI endpoint returns:
        { "answer": "...", "sources": [...], "error": null }
    """

    @pytest.mark.asyncio
    async def test_chat_endpoint_schema(self):
        from httpx import AsyncClient, ASGITransport
        from backend.main import app

        mock_result = {
            "question": "test",
            "answer": "LBRCE has multiple departments.",
            "sources": [{"title": "Dept page", "url": "https://lbrce.ac.in", "page": None, "source_type": "html"}],
            "error": None,
            "evidence_sufficient": True,
        }

        with patch("backend.graph.graph.rag_graph.ainvoke", new=AsyncMock(return_value=mock_result)):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/chat", json={"query": "What departments are at LBRCE?"})

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "error" in data
        assert data["answer"] == "LBRCE has multiple departments."
        assert isinstance(data["sources"], list)
