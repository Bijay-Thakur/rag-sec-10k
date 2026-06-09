"""Adapters bridging the FastAPI layer to the existing RAG pipeline."""

from backend.app.adapters.rag_adapter import RAGPipelineError, ask_question, index_is_ready
from backend.app.rag.schemas import RAGResponse

__all__ = ["RAGPipelineError", "RAGResponse", "ask_question", "index_is_ready"]
