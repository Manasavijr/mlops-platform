import logging, time
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
app.add_middleware(CORSMiddleware, allow_origins=settings.ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
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


@app.get("/ui", include_in_schema=False)
async def custom_ui():
    from fastapi.responses import HTMLResponse
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>MLOps Platform API</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }
        .header { background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 100%); border-bottom: 1px solid #2d3748; padding: 32px 48px; }
        .header h1 { font-size: 28px; font-weight: 700; color: #fff; }
        .header p { color: #718096; margin-top: 8px; font-size: 15px; }
        .badge { display: inline-block; background: #2d3748; color: #68d391; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; margin-top: 12px; }
        .container { max-width: 960px; margin: 0 auto; padding: 40px 48px; }
        .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 32px; }
        .card { background: #1a1f2e; border: 1px solid #2d3748; border-radius: 12px; padding: 24px; transition: border-color 0.2s; }
        .card:hover { border-color: #4a5568; }
        .card h3 { font-size: 16px; font-weight: 600; color: #fff; margin-bottom: 8px; }
        .card p { color: #718096; font-size: 14px; line-height: 1.6; }
        .method { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; margin-right: 8px; }
        .post { background: #276749; color: #68d391; }
        .get { background: #2a4365; color: #63b3ed; }
        .endpoint { font-family: monospace; font-size: 13px; color: #a0aec0; }
        .endpoints { margin-top: 16px; display: flex; flex-direction: column; gap: 8px; }
        .ep { display: flex; align-items: center; padding: 8px 12px; background: #0f1117; border-radius: 6px; }
        .links { display: flex; gap: 16px; margin-top: 32px; }
        .btn { padding: 10px 24px; border-radius: 8px; font-size: 14px; font-weight: 600; text-decoration: none; }
        .btn-primary { background: #3182ce; color: white; }
        .btn-secondary { background: #2d3748; color: #e2e8f0; }
        .status { display: flex; align-items: center; gap: 8px; margin-top: 32px; padding: 16px 20px; background: #1a1f2e; border: 1px solid #276749; border-radius: 8px; }
        .dot { width: 8px; height: 8px; border-radius: 50%; background: #68d391; }
        .section-title { font-size: 18px; font-weight: 600; color: #fff; margin-top: 40px; margin-bottom: 4px; }
        .section-sub { color: #718096; font-size: 14px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>MLOps Platform & LLM Inference System</h1>
        <p>Production-grade ML inference API with MLflow, drift detection, and GCP Cloud Run deployment</p>
        <span class="badge">v1.0.0 · Live on GCP Cloud Run</span>
    </div>
    <div class="container">
        <div class="status">
            <div class="dot"></div>
            <span style="color:#68d391;font-weight:600;">Service Healthy</span>
            <span style="color:#718096;margin-left:8px;">· DistilBERT loaded · Drift detection active</span>
        </div>

        <div class="links" style="margin-top:24px;">
            <a class="btn btn-primary" href="/docs">Swagger UI</a>
            <a class="btn btn-secondary" href="/health">Health Check</a>
            <a class="btn btn-secondary" href="https://github.com/Manasavijr/mlops-platform" target="_blank">GitHub</a>
        </div>

        <div class="section-title" style="margin-top:40px;">API Endpoints</div>
        <div class="section-sub">All endpoints accept and return JSON</div>

        <div class="grid">
            <div class="card">
                <h3>🤖 Inference</h3>
                <p>Sentiment classification using DistilBERT. Supports batch prediction up to 32 texts.</p>
                <div class="endpoints">
                    <div class="ep"><span class="method post">POST</span><span class="endpoint">/api/v1/inference/predict</span></div>
                    <div class="ep"><span class="method get">GET</span><span class="endpoint">/api/v1/inference/warmup</span></div>
                </div>
            </div>
            <div class="card">
                <h3>📊 Drift Detection</h3>
                <p>KS-test statistical drift monitoring. Auto-triggers retraining when p &lt; 0.05.</p>
                <div class="endpoints">
                    <div class="ep"><span class="method post">POST</span><span class="endpoint">/api/v1/drift/check</span></div>
                    <div class="ep"><span class="method get">GET</span><span class="endpoint">/api/v1/drift/history</span></div>
                </div>
            </div>
            <div class="card">
                <h3>🗂️ Model Registry</h3>
                <p>MLflow model versioning, stage promotion, and hot-reload without downtime.</p>
                <div class="endpoints">
                    <div class="ep"><span class="method get">GET</span><span class="endpoint">/api/v1/models/current</span></div>
                    <div class="ep"><span class="method post">POST</span><span class="endpoint">/api/v1/models/promote</span></div>
                    <div class="ep"><span class="method post">POST</span><span class="endpoint">/api/v1/models/reload</span></div>
                </div>
            </div>
            <div class="card">
                <h3>❤️ Health</h3>
                <p>Liveness and readiness probes for Cloud Run and Kubernetes deployments.</p>
                <div class="endpoints">
                    <div class="ep"><span class="method get">GET</span><span class="endpoint">/health</span></div>
                    <div class="ep"><span class="method get">GET</span><span class="endpoint">/health/live</span></div>
                    <div class="ep"><span class="method get">GET</span><span class="endpoint">/health/ready</span></div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
    return HTMLResponse(content=html)
