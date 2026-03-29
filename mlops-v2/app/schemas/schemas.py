from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class InferenceRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    texts: List[str] = Field(..., min_length=1, max_length=32)
    request_id: Optional[str] = None

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v):
        for text in v:
            if not text or not text.strip():
                raise ValueError("Each text must be non-empty")
            if len(text) > 10_000:
                raise ValueError("Each text must be under 10,000 characters")
        return [t.strip() for t in v]


class PredictionResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    text: str
    label: str
    score: float = Field(..., ge=0.0, le=1.0)
    model_version: str


class InferenceResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predictions: List[PredictionResult]
    model_name: str
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: Optional[str]
    model_uri: Optional[str]
    is_loaded: bool
    backend: str


class ModelVersion(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    version: str
    stage: str
    run_id: Optional[str]
    creation_timestamp: Optional[int]
    description: Optional[str]
    tags: Dict[str, str] = {}


class ModelListResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    models: List[ModelVersion]
    total: int


class PromoteModelRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    version: str
    stage: str = Field(..., pattern="^(Staging|Production|Archived)$")


class DriftReport(BaseModel):
    feature: str
    statistic: float
    p_value: float
    drift_detected: bool
    threshold: float


class DriftCheckResponse(BaseModel):
    drift_detected: bool
    reports: List[DriftReport]
    sample_size: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    retraining_triggered: bool = False


class DriftDataInput(BaseModel):
    scores: List[float] = Field(..., description="Recent prediction confidence scores")

    @field_validator("scores")
    @classmethod
    def validate_scores(cls, v):
        if len(v) < 10:
            raise ValueError("Need at least 10 scores for drift detection")
        return v


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    version: str
    model_loaded: bool
    model_info: Optional[Dict[str, Any]] = None
    uptime_seconds: Optional[float] = None
