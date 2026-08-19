"""
LangGraph graph definition for the LBRCE AI Assistant.

Graph topology:

    START
      |
    plan_query_node
      |
    query_rewrite_node  # LLM rewrite only for general intent; otherwise passthrough
      |
    retrieve_node
      |
    evaluate_evidence_node
      +-- sufficient=True  --> assemble_context_node --> generate_answer_node --> check_groundedness_node
      +-- sufficient=False --> tavily_search_node     --> assemble_context_node --> generate_answer_node --> check_groundedness_node

    check_groundedness_node
      +-- needs_web_retry=True  --> tavily_search_node --> assemble_context_node --> generate_answer_node --> check_groundedness_node --> END
      +-- needs_web_retry=False --> END

The groundedness retry loop is self-limiting: tavily_search_node always sets
tavily_attempted=True whenever it runs, and check_groundedness_node only
requests a retry when tavily_attempted is still False. So the graph can pass
through tavily_search_node at most twice per request (once on the initial
insufficient-evidence path, or once on the groundedness-retry path — never
both, since the initial Tavily path already sets tavily_attempted=True).

The graph is stateless: no checkpointer is attached, and each ainvoke()
call begins with a fresh GraphState.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import StateGraph, END

from backend.graph.state import GraphState
from backend.graph.nodes import (
    retrieve_node,
    evaluate_evidence_node,
    tavily_search_node,
    assemble_context_node,
    generate_answer_node,
    check_groundedness_node,
    plan_query_node,
    query_rewrite_node,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional edge routers
# ---------------------------------------------------------------------------

def _route_after_evaluation(
    state: GraphState,
) -> Literal["assemble_context", "tavily_search"]:
    """
    Route the graph after the evidence evaluation node.

    Returns:
        "assemble_context"  — when Pinecone evidence is sufficient.
        "tavily_search"     — when evidence is insufficient (Tavily fallback).
    """
    if state.get("evidence_sufficient", False):
        logger.info("[router] Evidence sufficient → assemble_context (Pinecone path)")
        return "assemble_context"
    logger.info("[router] Evidence insufficient → tavily_search (fallback path)")
    return "tavily_search"


def _route_after_groundedness(
    state: GraphState,
) -> Literal["tavily_search", "__end__"]:
    """
    Route the graph after check_groundedness_node.

    Returns:
        "tavily_search" — the Pinecone-sourced answer was ungrounded and
                           Tavily has not been attempted yet this request.
        "__end__"       — the answer is grounded, or Tavily has already been
                           tried (either as the original fallback path or as
                           this same retry), so there is nothing further to
                           attempt.
    """
    if state.get("needs_web_retry", False):
        return "tavily_search"
    return "__end__"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    builder = StateGraph(GraphState)

    # Register nodes
    builder.add_node("plan_query", plan_query_node)
    builder.add_node("query_rewrite", query_rewrite_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("evaluate_evidence", evaluate_evidence_node)
    builder.add_node("tavily_search", tavily_search_node)
    builder.add_node("assemble_context", assemble_context_node)
    builder.add_node("generate_answer", generate_answer_node)
    builder.add_node("check_groundedness", check_groundedness_node)

    # Linear edges
    builder.set_entry_point("plan_query")
    builder.add_edge("plan_query", "query_rewrite")
    builder.add_edge("query_rewrite", "retrieve")
    builder.add_edge("retrieve", "evaluate_evidence")

    # Conditional edge: evidence_sufficient?
    builder.add_conditional_edges(
        "evaluate_evidence",
        _route_after_evaluation,
        {
            "assemble_context": "assemble_context",
            "tavily_search": "tavily_search",
        },
    )

    # Tavily path → assemble_context (same node, different upstream data)
    builder.add_edge("tavily_search", "assemble_context")

    # Both paths converge at assemble_context → generate_answer → check_groundedness
    builder.add_edge("assemble_context", "generate_answer")
    builder.add_edge("generate_answer", "check_groundedness")

    # Conditional edge: does the answer need a web-search retry?
    builder.add_conditional_edges(
        "check_groundedness",
        _route_after_groundedness,
        {
            "tavily_search": "tavily_search",
            "__end__": END,
        },
    )

    return builder


# Compile once at module load.
# No checkpointer → stateless: each ainvoke() call is independent.
rag_graph = _build_graph().compile()

logger.info("LangGraph rag_graph compiled successfully.")