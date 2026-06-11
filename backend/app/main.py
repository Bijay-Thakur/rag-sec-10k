"""
SEC Insight AI — FastAPI backend.

Run from repo root (local dev):
    uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8770

For production, bind to 127.0.0.1 behind a reverse proxy (nginx/caddy).
Never expose the backend port directly to the public internet.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, List

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from backend.app.adapters.rag_adapter import (
    RAGPipelineError,
    ask_question,
    get_sample_questions,
    index_is_ready,
)
from backend.app.auth.deps import get_optional_user
from backend.app.auth.models import UserContext
from backend.app.filings.registry import filing_catalog, list_filing_ids
from backend.app.constants import DEMO_VIDEO_URL
from backend.app.config import get_settings
from backend.app.middleware.rate_limit import limiter, tiered_rate_limit
from backend.app.schemas import (
    AskRequest,
    AskResponse,
    EntitlementsResponse,
    ErrorResponse,
    EvalSummaryResponse,
    FilingInfo,
    FilingsResponse,
    HealthResponse,
    SampleQuestion,
    SampleQuestionsResponse,
)
from backend.app.services.access_policy import resolve_ask_access
from backend.app.services.eval_service import get_eval_summary
from backend.app.services.cost_guard import record_global_llm_usage
from backend.app.services.quota_service import (
    SupabaseUnavailableError,
    get_entitlements,
    record_token_usage,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sec_insight_api")

_DEBUG = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    filing_ids = list_filing_ids()
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    logger.info("Registered filings (%d): %s", len(filing_ids), ", ".join(filing_ids))
    if settings.enforce_access_policy:
        logger.info(
            "Access policy enabled — anonymous_demo_only=%s",
            settings.anonymous_demo_only,
        )
    else:
        logger.warning(
            "ENFORCE_ACCESS_POLICY=false — anonymous callers may trigger live LLM. "
            "Set ENFORCE_ACCESS_POLICY=true in production."
        )

    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY not set — live LLM requires authenticated user + key")

    if not settings.enable_live_llm_calls:
        logger.warning(
            "ENABLE_LIVE_LLM_CALLS=false — all live LLM requests are blocked; demo mode only."
        )
    if settings.max_daily_llm_calls > 0 or settings.max_daily_estimated_cost_usd > 0:
        logger.info(
            "Deployment cost guards: max_daily_llm_calls=%s max_daily_estimated_cost_usd=%s",
            settings.max_daily_llm_calls or "unlimited",
            settings.max_daily_estimated_cost_usd or "unlimited",
        )

    if settings.supabase_url and not settings.supabase_service_role_key:
        logger.critical(
            "SUPABASE_URL is set but SUPABASE_SERVICE_ROLE_KEY is missing. "
            "Durable quota tracking is DISABLED — users can exhaust quota and "
            "regain it on every server restart. Set SUPABASE_SERVICE_ROLE_KEY "
            "to enable persistent quota enforcement."
        )

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
        # Disable interactive docs in production to avoid exposing the API schema.
        docs_url="/docs" if _DEBUG else None,
        redoc_url="/redoc" if _DEBUG else None,
        openapi_url="/openapi.json" if _DEBUG else None,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    origins: List[str] = [
        o.strip() for o in settings.frontend_origin.split(",") if o.strip()
    ]
    if not origins:
        origins = ["http://localhost:3000"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept"],
    )

    @app.exception_handler(RAGPipelineError)
    async def rag_pipeline_error_handler(_request: Request, exc: RAGPipelineError) -> JSONResponse:
        status = 503
        if exc.error_code == "unknown_filing":
            status = 404
        elif exc.error_code == "missing_api_key":
            status = 503
        elif exc.error_code in ("chunks_missing", "index_empty", "index_unavailable"):
            status = 503
        elif exc.error_code == "auth_required":
            status = 403
        elif exc.error_code == "budget_exceeded":
            status = 429
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

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    @limiter.limit("60/minute")
    async def health(request: Request) -> HealthResponse:
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
    @limiter.limit("60/minute")
    async def list_filings(request: Request) -> FilingsResponse:
        return FilingsResponse(
            default_filing_id=settings.default_filing_id,
            filings=[FilingInfo(**row) for row in filing_catalog()],
        )

    @app.get("/api/sample-questions", response_model=SampleQuestionsResponse, tags=["rag"])
    @limiter.limit("60/minute")
    async def sample_questions(
        request: Request,
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

    @app.get("/api/me/entitlements", response_model=EntitlementsResponse, tags=["auth"])
    @limiter.limit("30/minute")
    async def entitlements(
        request: Request,
        user: UserContext | None = Depends(get_optional_user),
    ) -> EntitlementsResponse:
        runtime_settings = get_settings()
        if user is None:
            from backend.app.auth.deps import extract_bearer_token

            if extract_bearer_token(request):
                raise HTTPException(
                    status_code=401,
                    detail="Session token invalid or expired. Sign out and sign in again.",
                )
            return EntitlementsResponse(
                authenticated=False,
                plan="anonymous",
                demo_mode_only=runtime_settings.anonymous_demo_only,
                can_use_llm=False,
            )
        try:
            ent = get_entitlements(user, runtime_settings)
        except SupabaseUnavailableError:
            raise HTTPException(
                status_code=503,
                detail="Quota service temporarily unavailable. Please try again.",
            )
        return EntitlementsResponse(
            authenticated=True,
            email=ent.email,
            plan=ent.plan,
            llm_calls_used=ent.llm_calls_used,
            llm_calls_limit=ent.llm_calls_limit,
            llm_calls_remaining=ent.llm_calls_remaining,
            can_use_llm=ent.can_use_llm,
            demo_mode_only=False,
            daily_token_budget=ent.daily_token_budget,
            daily_tokens_used=ent.daily_tokens_used,
        )

    async def _ask_handler(
        request: Request,
        body: AskRequest,
        user: UserContext | None = Depends(get_optional_user),
    ) -> AskResponse:
        del request
        runtime_settings = get_settings()
        access = resolve_ask_access(
            requested_demo_mode=body.demo_mode,
            user=user,
            settings=runtime_settings,
        )
        try:
            response = ask_question(
                body.question,
                filing_id=body.filing_id,
                demo_mode=access.effective_demo_mode,
            )
        except RAGPipelineError:
            raise
        except Exception as exc:
            logger.exception("Unexpected error in /api/ask")
            raise HTTPException(status_code=500, detail="Failed to process question") from exc

        if access.user and not access.effective_demo_mode:
            total_tokens = 0
            if response.token_usage:
                total_tokens = int(response.token_usage.total_tokens or 0)
            record_token_usage(access.user.user_id, total_tokens, runtime_settings)
            record_global_llm_usage(
                runtime_settings,
                estimated_cost_usd=response.estimated_cost_usd,
            )

        return response

    ask_handler = _ask_handler
    if settings.rate_limit_enabled:
        # tiered_rate_limit inspects the caller at request time:
        # anonymous → 30/hour, free → 120/hour, pro → 300/hour
        ask_handler = limiter.limit(tiered_rate_limit)(_ask_handler)

    app.post("/api/ask", response_model=AskResponse, tags=["rag"])(ask_handler)

    @app.get("/api/eval/summary", response_model=EvalSummaryResponse, tags=["eval"])
    @limiter.limit("30/minute")
    async def eval_summary(request: Request) -> EvalSummaryResponse:
        return get_eval_summary()

    return app


app = create_app()
