import logging

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from ml.drift.detector import DriftDetector
from app.schemas.schemas import DriftCheckResponse, DriftDataInput

logger = logging.getLogger(__name__)
router = APIRouter()
_detector = DriftDetector()


@router.post("/check", response_model=DriftCheckResponse)
async def check_drift(payload: DriftDataInput):
    try:
        result = _detector.check(payload.scores)
        if result.drift_detected and settings.RETRAIN_TRIGGER_ENABLED:
            logger.warning(f"DRIFT DETECTED — p_value={result.reports[0].p_value:.4f}")
            result.retraining_triggered = True
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reference-stats")
async def get_reference_stats():
    return _detector.reference_stats()


@router.post("/reset-reference")
async def reset_reference():
    _detector.reset()
    return {"message": "Reference distribution reset."}


@router.get("/history")
async def get_drift_history(limit: int = 20):
    return {"history": _detector.history[-limit:], "total": len(_detector.history)}
