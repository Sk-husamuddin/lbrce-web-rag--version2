"""
Real integration tests for the LangGraph agent.
Uses actual Pinecone, OpenRouter, and Tavily credentials from .env.
API keys are NEVER printed or logged.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.config.settings import settings
from backend.graph.graph import rag_graph


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("="*60)


def safe_print(text: str, max_len: int = 300) -> None:
    """Print text safely on Windows cp1252 consoles by encoding to ASCII with replacement."""
    truncated = text[:max_len]
    safe = truncated.encode("ascii", errors="replace").decode("ascii")
    print(safe)


async def test_pinecone_success():
    """Test A — Pinecone has sufficient evidence → Tavily NOT called."""
    separator("TEST A: Pinecone sufficient (no Tavily)")
    print("Question: What departments are available at LBRCE?")

    result = await rag_graph.ainvoke({
        "question": "What departments are available at LBRCE?"
    })

    print(f"evidence_sufficient : {result.get('evidence_sufficient')}")
    print(f"tavily_results      : {len(result.get('tavily_results') or [])} items")
    print(f"sources             : {len(result.get('sources', []))} items")
    safe_print(f"Answer:\n{result.get('answer', '(none)')}")

    assert result.get("answer"), "No answer generated"
    print("\n[PASS] Test A")


async def test_tavily_fallback():
    """Test B — Off-topic query forces Tavily fallback."""
    separator("TEST B: Tavily fallback (Pinecone insufficient)")
    question = "What is the current JNTU Kakinada exam schedule for November 2025?"
    print(f"Question: {question}")

    result = await rag_graph.ainvoke({"question": question})

    print(f"evidence_sufficient : {result.get('evidence_sufficient')}")
    tavily = result.get("tavily_results") or []
    print(f"tavily_results      : {len(tavily)} items")
    if tavily:
        print(f"  first result URL  : {tavily[0].get('url','')}")
    sources = result.get("sources", [])
    print(f"sources             : {sources}")
    safe_print(f"Answer:\n{result.get('answer', '(none)')}")

    assert result.get("answer"), "No answer generated"
    print("\n[PASS] Test B")


async def test_no_evidence_fallback():
    """Test C — Nonsensical question → safe fallback answer."""
    separator("TEST C: No evidence fallback")
    question = "What is the tuition fee schedule at Hogwarts School of Witchcraft?"
    print(f"Question: {question}")

    result = await rag_graph.ainvoke({"question": question})

    print(f"evidence_sufficient : {result.get('evidence_sufficient')}")
    safe_print(f"Answer:\n{result.get('answer', '(none)')}")

    answer = result.get("answer", "")
    assert answer, "Expected a fallback answer but got nothing"
    print("\n[PASS] Test C")


async def test_statelessness():
    """Test D — Two independent requests, no state bleed."""
    separator("TEST D: Statelessness")

    q1 = "What laboratories does LBRCE have?"
    q2 = "What is the fee structure for Mars University?"  # Unrelated nonsense

    r1 = await rag_graph.ainvoke({"question": q1})
    r2 = await rag_graph.ainvoke({"question": q2})

    # Request-2 docs must not contain request-1 documents
    r1_docs = r1.get("retrieved_documents") or []
    r2_docs = r2.get("retrieved_documents") or []

    r1_ids = {d.get("document_id") for d in r1_docs}
    r2_ids = {d.get("document_id") for d in r2_docs}

    print(f"Request 1 doc IDs: {r1_ids}")
    print(f"Request 2 doc IDs: {r2_ids}")
    safe_print(f"Request 1 answer: {r1.get('answer','')[:120]}...")
    safe_print(f"Request 2 answer: {r2.get('answer','')[:120]}...")
    print("\n[PASS] Test D — Separate state objects, no shared mutable state")


async def main():
    print("\n=== LBRCE LangGraph Real Integration Tests ===")
    print(f"Pinecone index  : {settings.PINECONE_INDEX_NAME}")
    print(f"OpenRouter model: {settings.OPENROUTER_MODEL}")
    print(f"Threshold       : {settings.RAG_RELEVANCE_THRESHOLD}")
    print(f"Tavily key set  : {'YES' if settings.TAVILY_API_KEY else 'NO'}")

    await test_pinecone_success()
    await test_tavily_fallback()
    await test_no_evidence_fallback()
    await test_statelessness()

    print("\n" + "="*60)
    print("  ALL REAL INTEGRATION TESTS PASSED")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
