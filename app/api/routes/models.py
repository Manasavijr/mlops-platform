import logging
from typing import Optional
import mlflow
from fastapi import APIRouter, HTTPException, Request
from app.core.config import settings
from app.schemas.schemas import ModelInfo, ModelListResponse, ModelVersion, PromoteModelRequest

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/current", response_model=ModelInfo)
async def get_current_model(request: Request):
    return ModelInfo(**request.app.state.model_registry.info)

@router.get("/versions", response_model=ModelListResponse)
async def list_model_versions(model_name: Optional[str] = None, stage: Optional[str] = None):
    try:
        client = mlflow.tracking.MlflowClient(settings.MLFLOW_TRACKING_URI)
        name = model_name or settings.MLFLOW_MODEL_NAME
        kwargs = {"stages": [stage]} if stage else {}
        versions = client.get_latest_versions(name, **kwargs)
        result = [ModelVersion(name=v.name, version=v.version, stage=v.current_stage, run_id=v.run_id, creation_timestamp=v.creation_timestamp, description=v.description or "", tags={}) for v in versions]
        return ModelListResponse(models=result, total=len(result))
    except Exception as e:
        logger.warning(f"MLflow unavailable: {e}")
        return ModelListResponse(models=[], total=0)

@router.post("/promote")
async def promote_model(body: PromoteModelRequest):
    try:
        client = mlflow.tracking.MlflowClient(settings.MLFLOW_TRACKING_URI)
        client.transition_model_version_stage(name=body.model_name, version=body.version, stage=body.stage, archive_existing_versions=(body.stage == "Production"))
        return {"message": f"Model {body.model_name} v{body.version} promoted to {body.stage}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/reload", response_model=ModelInfo)
async def reload_model(request: Request):
    registry = request.app.state.model_registry
    registry.cleanup()
    await registry.load_production_model()
    return ModelInfo(**registry.info)

@router.get("/experiments")
async def list_experiments():
    try:
        client = mlflow.tracking.MlflowClient(settings.MLFLOW_TRACKING_URI)
        experiments = client.search_experiments()
        return {"experiments": [{"experiment_id": e.experiment_id, "name": e.name, "lifecycle_stage": e.lifecycle_stage} for e in experiments]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {e}")
