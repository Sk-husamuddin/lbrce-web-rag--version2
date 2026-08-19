"""
FastAPI /chat route — LBRCE AI Assistant.

Invokes the LangGraph stateless agent (rag_graph) and returns the
grounded answer with sources.  The request/response contract is unchanged.
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from backend.graph.graph import rag_graph

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User question about LBRCE college.",
    )
    top_k: int = Field(4, ge=1, le=10, description="Number of retrieval results to use.")

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty or whitespace-only")
        return value


class VisualResource(BaseModel):
    title: str
    url: str
    type: str
    page: int | None = None
    department: str | None = None
    academic_year: str | None = None
    term: str | None = None
    semester: str | None = None
    section: str | None = None
    url_metadata_only: bool | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    visual_resources: list[VisualResource] = Field(default_factory=list)
    error: str | None = None


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Process a user query via the LangGraph RAG agent and return a grounded answer.

    The agent follows this flow:
        question → retrieve (Pinecone) → evaluate evidence
            → sufficient  : assemble_context → generate_answer
            → insufficient: tavily_search → assemble_context → generate_answer
    """
    try:
        initial_state = {
            "question": request.query,
            "top_k": request.top_k,
        }
        result = await rag_graph.ainvoke(initial_state)

        return ChatResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            visual_resources=result.get("visual_resources", []),
            error=result.get("error"),
        )
    except Exception as e:
        logger.exception("Chat endpoint error")
        raise HTTPException(
            status_code=500,
            detail="The assistant is temporarily unavailable. Please try again later.",
        ) from e
