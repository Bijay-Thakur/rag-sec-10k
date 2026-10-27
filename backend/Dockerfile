# SEC Insight AI — FastAPI backend
# Build from repo root:
#   docker build -f backend/Dockerfile --ignorefile backend/.dockerignore -t sec-insight-api .
#
# Cloud Run sets PORT automatically (default 8080). The CMD below binds to it.
# No dummy vector index is included. A verified real index may be mounted.

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# lxml + torch/sentence-transformers build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Application code uses backend.app.* imports in both local and container runs.
COPY backend/__init__.py ./backend/__init__.py
COPY backend/app ./backend/app

# V1 RAG pipeline (retrieval + generation)
COPY src ./src
COPY scripts/run_ingestion_worker.py ./scripts/run_ingestion_worker.py

# Offline BM25 chunks and historical evaluation summaries
COPY data/chunks ./data/chunks
COPY data/eval ./data/eval

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8080}/health" || exit 1

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
