import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.pyfunc
import torch
from transformers import pipeline

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self):
        self.model = None
        self.pipeline = None
        self.model_name: str = settings.MLFLOW_MODEL_NAME
        self.model_version: Optional[str] = None
        self.model_uri: Optional[str] = None
        self._cache_dir = Path(settings.MODEL_CACHE_DIR)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    async def load_production_model(self) -> None:
        try:
            await self._load_from_mlflow()
        except Exception as e:
            logger.warning(f"MLflow load failed ({e}), falling back to HuggingFace Hub")
            await self._load_from_huggingface()

    async def _load_from_mlflow(self) -> None:
        client = mlflow.tracking.MlflowClient()
        # Use search_model_versions (works in all MLflow 2.x versions)
        versions = client.search_model_versions(f"name='{self.model_name}'")
        prod = [v for v in versions if v.current_stage == "Production"]
        if not prod:
            raise ValueError(f"No Production version found for '{self.model_name}'")
        mv = prod[0]
        self.model_version = mv.version
        self.model_uri = f"models:/{self.model_name}/Production"
        logger.info(f"Loading MLflow model: {self.model_uri}")
        self.model = mlflow.pyfunc.load_model(self.model_uri)
        logger.info(f"MLflow model loaded (version {self.model_version})")

    async def _load_from_huggingface(self) -> None:
        hf_model = settings.HF_MODEL_NAME
        logger.info(f"Loading HuggingFace model: {hf_model}")
        device = 0 if torch.cuda.is_available() else -1
        # NOTE: do NOT pass cache_dir or truncation here — unsupported in newer transformers
        self.pipeline = pipeline(
            "text-classification",
            model=hf_model,
            tokenizer=hf_model,
            device=device,
        )
        self.model_version = "hf-fallback"
        self.model_name = hf_model
        logger.info(f"HuggingFace model loaded: {hf_model}")

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")
        results = []
        if self.pipeline is not None:
            # Pass truncation at call time, not pipeline init
            raw = self.pipeline(
                texts,
                truncation=True,
                max_length=settings.MAX_SEQUENCE_LENGTH,
                batch_size=settings.INFERENCE_BATCH_SIZE,
            )
            for text, pred in zip(texts, raw):
                results.append({
                    "text": text,
                    "label": pred["label"].lower(),
                    "score": round(float(pred["score"]), 4),
                    "model_version": self.model_version,
                })
        elif self.model is not None:
            import pandas as pd
            preds = self.model.predict(pd.DataFrame({"text": texts}))
            for text, pred in zip(texts, preds):
                if isinstance(pred, dict):
                    label = pred.get("label", "unknown")
                    score = pred.get("score", 1.0)
                else:
                    label = str(pred)
                    score = 1.0
                results.append({
                    "text": text,
                    "label": label,
                    "score": round(float(score), 4),
                    "model_version": self.model_version,
                })
        return results

    @property
    def is_loaded(self) -> bool:
        return self.model is not None or self.pipeline is not None

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_uri": self.model_uri,
            "is_loaded": self.is_loaded,
            "backend": "mlflow" if self.model else "huggingface",
        }

    def cleanup(self) -> None:
        self.model = None
        self.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model resources released.")
