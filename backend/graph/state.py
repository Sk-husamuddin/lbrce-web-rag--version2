"""
LangGraph state definition for the LBRCE AI Assistant agent.

Contains ONLY the data required for a single request/response cycle.
Nothing here is persisted between requests — the agent is stateless.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):
    """
    Stateless per-request state shared between LangGraph nodes.

    Fields are populated progressively as the graph executes:

        question                 – the original user query
        retrieved_documents      – list of result dicts from Pinecone
        retrieval_scores         – parallel list of similarity scores
        retrieval_used_metadata_filter – True when retrieval used an exact
                                         Pinecone metadata filter
        evidence_sufficient      – True when Pinecone evidence passes
                                   the threshold heuristic. Reset to False
                                   by tavily_search_node whenever it runs,
                                   so assemble_context_node always branches
                                   onto the freshest evidence source.
        tavily_results           – web-search results (fallback only)
        tavily_attempted         – True once tavily_search_node has run at
                                   least once this request, regardless of
                                   whether it returned any results. Used to
                                   prevent infinite retry loops in
                                   check_groundedness_node — do NOT infer
                                   this from `bool(tavily_results)`, since an
                                   empty-but-attempted search must not look
                                   the same as "never attempted".
        grounded                 – structured answer-generation result; True
                                   when the model says the evidence answers the
                                   question, False when it does not. Optional
                                   so deterministic and legacy node outputs
                                   default safely to grounded=True.
        needs_web_retry           – set by check_groundedness_node; True when
                                   the Pinecone-sourced answer was ungrounded
                                   and Tavily hasn't been tried yet.
        context                  – assembled evidence text for the LLM
        context_docs             – per-document evidence used to build context
        answer                   – final LLM-generated answer
        sources                  – list of { title, url, page, source_type }
        visual_resources         – matched timetable images and PDFs for display
        error                    – error message to surface to the caller

    Conversation history, user memory, and any cross-request persistence
    are deliberately absent.
    """

    # Input
    question: str
    top_k: int

    # Planner output
    intent: str
    normalized_query: str
    retrieval_query: str
    query_filters: Dict[str, Any]
    retrieval_metadata_filter: Dict[str, Any]
    source_policy: str
    planner_confidence: float

    # Query rewrite. Populated by query_rewrite_node only for general intent;
    # failures fall back to the original question and never block the pipeline.
    rewritten_query: str
    query_rewrite_applied: bool

    # Retrieval
    retrieved_documents: List[Dict[str, Any]]
    retrieval_scores: List[float]
    retrieval_used_metadata_filter: bool

    # Routing decision
    evidence_sufficient: bool

    # Tavily fallback
    tavily_results: List[Dict[str, Any]]
    tavily_attempted: bool

    # Groundedness retry
    grounded: Optional[bool]
    needs_web_retry: bool

    # LLM input / output
    context: str
    context_docs: List[Dict[str, Any]]
    answer: str
    sources: List[Dict[str, Any]]
    visual_resources: List[Dict[str, Any]]

    # Error propagation
    error: Optional[str]