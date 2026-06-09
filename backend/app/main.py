"""
SEC Insight AI — FastAPI backend.

Run from repo root:
    uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, List

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.adapters.rag_adapter import (
    RAGPipelineError,
    ask_question,
    get_sample_questions,
    index_is_ready,
)
from backend.app.filings.registry import filing_catalog, list_filing_ids
from backend.app.constants import DEMO_VIDEO_URL
from backend.app.config import get_settings
from backend.app.schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    EvalSummaryResponse,
    FilingInfo,
    FilingsResponse,
    HealthResponse,
    SampleQuestion,
    SampleQuestionsResponse,
)
from backend.app.services.eval_service import get_eval_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sec_insight_api")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    filing_ids = list_filing_ids()
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    logger.info("Registered filings (%d): %s", len(filing_ids), ", ".join(filing_ids))
    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY not set — /api/ask requires demo_mode=true")
    yield
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=(
            "Production-style API for SEC 10-K RAG Q&A. "
            f"Demo video: {DEMO_VIDEO_URL}"
        ),
        version=settings.app_version,
        lifespan=lifespan,
    )

    # CORS — FRONTEND_ORIGIN may be a comma-separated list
    origins: List[str] = [
        o.strip() for o in settings.frontend_origin.split(",") if o.strip()
    ]
    if not origins:
        origins = ["http://localhost:3000"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Exception handlers
    # ------------------------------------------------------------------

    @app.exception_handler(RAGPipelineError)
    async def rag_pipeline_error_handler(_request: Request, exc: RAGPipelineError) -> JSONResponse:
        status = 503
        if exc.error_code == "unknown_filing":
            status = 404
        elif exc.error_code == "missing_api_key":
            status = 503
        elif exc.error_code in ("chunks_missing", "index_empty", "index_unavailable"):
            status = 503
        logger.error("Pipeline error [%s]: %s", exc.error_code, exc)
        return JSONResponse(
            status_code=status,
            content=ErrorResponse(detail=str(exc), error_code=exc.error_code).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                detail="Request validation failed",
                error_code="validation_error",
            ).model_dump()
            | {"errors": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                detail="An internal server error occurred.",
                error_code="internal_error",
            ).model_dump(),
        )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        ready = index_is_ready()
        filings = list_filing_ids()
        return HealthResponse(
            status="ok" if ready else "degraded",
            version=settings.app_version,
            index_ready=ready,
            default_filing_id=settings.default_filing_id,
            available_filings=filings,
            filing_count=len(filings),
        )

    @app.get("/api/filings", response_model=FilingsResponse, tags=["rag"])
    async def list_filings() -> FilingsResponse:
        return FilingsResponse(
            default_filing_id=settings.default_filing_id,
            filings=[FilingInfo(**row) for row in filing_catalog()],
        )

    @app.get("/api/sample-questions", response_model=SampleQuestionsResponse, tags=["rag"])
    async def sample_questions(
        filing_id: str | None = Query(default=None, description="Filing identifier"),
    ) -> SampleQuestionsResponse:
        try:
            fid, questions = get_sample_questions(filing_id)
        except RAGPipelineError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        return SampleQuestionsResponse(
            filing_id=fid,
            questions=[SampleQuestion(**q) for q in questions],
        )

    @app.post("/api/ask", response_model=AskResponse, tags=["rag"])
    async def ask(body: AskRequest) -> AskResponse:
        try:
            return ask_question(
                body.question,
                filing_id=body.filing_id,
                demo_mode=body.demo_mode,
            )
        except RAGPipelineError:
            raise  # handled by exception handler
        except Exception as exc:
            logger.exception("Unexpected error in /api/ask")
            raise HTTPException(status_code=500, detail="Failed to process question") from exc

    @app.get("/api/eval/summary", response_model=EvalSummaryResponse, tags=["eval"])
    async def eval_summary() -> EvalSummaryResponse:
        return get_eval_summary()

    return app


app = create_app()
