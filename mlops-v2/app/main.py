import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import health, inference, models, drift
from app.core.config import settings
from app.core.model_registry import ModelRegistry
from app.core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting MLOps Platform...")
    registry = ModelRegistry()
    app.state.model_registry = registry
    await registry.load_production_model()
    logger.info(f"Model loaded: {registry.model_name} v{registry.model_version}")
    yield
    logger.info("Shutting down...")
    registry.cleanup()


app = FastAPI(
    title="MLOps Platform & LLM Inference System",
    description="Production-grade ML inference API with MLflow, drift detection, and GCP Cloud Run deployment.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(inference.router, prefix="/api/v1/inference", tags=["Inference"])
app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])
app.include_router(drift.router, prefix="/api/v1/drift", tags=["Drift Detection"])


@app.get("/", include_in_schema=False)
async def root():
    return {"service": "MLOps Platform", "version": "1.0.0", "docs": "/docs", "health": "/health"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=settings.DEBUG)
