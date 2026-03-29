import time
from datetime import datetime
from fastapi import APIRouter, Request
from app.core.config import settings
from app.schemas.schemas import HealthResponse

router = APIRouter()
_START_TIME = time.time()


@router.get("", response_model=HealthResponse)
async def health(request: Request):
    registry = getattr(request.app.state, "model_registry", None)
    model_loaded = registry.is_loaded if registry else False
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        version=settings.VERSION,
        model_loaded=model_loaded,
        model_info=registry.info if registry else None,
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )


@router.get("/live")
async def liveness():
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@router.get("/ready")
async def readiness(request: Request):
    registry = getattr(request.app.state, "model_registry", None)
    if registry and registry.is_loaded:
        return {"status": "ready"}
    return {"status": "not_ready"}
