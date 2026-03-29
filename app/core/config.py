from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")
    APP_NAME: str = "mlops-platform"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    PORT: int = 8080
    WORKERS: int = 2
    ALLOWED_ORIGINS: List[str] = ["*"]
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "sentiment-classifier"
    MLFLOW_MODEL_NAME: str = "sentiment-classifier-prod"
    MLFLOW_MODEL_STAGE: str = "Production"
    MODEL_CACHE_DIR: str = "/tmp/model_cache"
    HF_MODEL_NAME: str = "distilbert-base-uncased-finetuned-sst-2-english"
    MAX_SEQUENCE_LENGTH: int = 512
    INFERENCE_BATCH_SIZE: int = 32
    MODEL_TIMEOUT_SECONDS: int = 30
    DRIFT_THRESHOLD: float = 0.05
    DRIFT_WINDOW_SIZE: int = 1000
    DRIFT_CHECK_INTERVAL_MINUTES: int = 30
    RETRAIN_TRIGGER_ENABLED: bool = True
    GCP_PROJECT_ID: Optional[str] = None
    GCP_REGION: str = "us-central1"
    GCS_BUCKET_NAME: Optional[str] = None
    LOG_PREDICTIONS: bool = True
    PREDICTION_LOG_PATH: str = "/tmp/predictions.jsonl"
    ENABLE_METRICS: bool = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
