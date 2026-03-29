FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip && pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

FROM python:3.11-slim AS runtime
RUN groupadd --gid 1001 appuser && useradd --uid 1001 --gid appuser --shell /bin/bash --create-home appuser
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl && rm -rf /wheels
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser ml/ ./ml/
COPY --chown=appuser:appuser data/ ./data/
COPY --chown=appuser:appuser .env ./.env
RUN mkdir -p /tmp/model_cache /tmp/hf_cache && chown -R appuser:appuser /tmp/model_cache /tmp/hf_cache
USER appuser
ENV PORT=8080 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TRANSFORMERS_CACHE=/tmp/hf_cache HF_HOME=/tmp/hf_cache
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 CMD curl -f http://localhost:${PORT}/health/live || exit 1
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
