import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.config import settings
from app.schemas.schemas import InferenceRequest, InferenceResponse, PredictionResult

logger = logging.getLogger(__name__)
router = APIRouter()


def get_registry(request: Request):
    return request.app.state.model_registry


@router.post("/predict", response_model=InferenceResponse)
async def predict(payload: InferenceRequest, registry=Depends(get_registry)):
    if not registry.is_loaded:
        raise HTTPException(status_code=503, detail="Model not ready")

    request_id = payload.request_id or str(uuid.uuid4())
    t0 = time.perf_counter()

    try:
        raw_preds = registry.predict(payload.texts)
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    latency_ms = (time.perf_counter() - t0) * 1000
    predictions = [PredictionResult(**p) for p in raw_preds]

    if settings.LOG_PREDICTIONS:
        _log_predictions(request_id, predictions)

    logger.info(f"Inference | id={request_id} n={len(predictions)} latency={latency_ms:.1f}ms")
    return InferenceResponse(
        predictions=predictions,
        model_name=registry.model_name,
        latency_ms=round(latency_ms, 2),
        request_id=request_id,
    )


@router.get("/warmup")
async def warmup(registry=Depends(get_registry)):
    if not registry.is_loaded:
        raise HTTPException(status_code=503, detail="Model not ready")
    test_pred = registry.predict(["Warmup request."])
    return {"status": "warm", "test_prediction": test_pred[0]}


def _log_predictions(request_id: str, predictions: list) -> None:
    try:
        log_path = Path(settings.PREDICTION_LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            for pred in predictions:
                f.write(json.dumps({
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "label": pred.label,
                    "score": pred.score,
                    "model_version": pred.model_version,
                }) + "\n")
    except Exception as e:
        logger.warning(f"Failed to log predictions: {e}")
