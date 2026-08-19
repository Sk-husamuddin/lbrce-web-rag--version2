"""
LangGraph package for the LBRCE AI Assistant stateless agent.

Exports the compiled graph that the FastAPI /chat route invokes.
"""

from backend.graph.graph import rag_graph

__all__ = ["rag_graph"]
