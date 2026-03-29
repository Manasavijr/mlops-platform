import argparse, json, logging, sys
from pathlib import Path
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, str(Path(__file__).parents[2]))
from app.core.config import settings
from ml.training.train import generate_synthetic_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(run_id=None, model_uri=None):
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    if model_uri is None and run_id:
        model_uri = f"runs:/{run_id}/model"
    elif model_uri is None:
        model_uri = f"models:/{settings.MLFLOW_MODEL_NAME}/Production"
    logger.info(f"Loading model: {model_uri}")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        logger.error(f"Could not load model: {e}")
        return {}

    texts, labels = generate_synthetic_data(500)
    import pandas as pd
    preds_raw = model.predict(pd.DataFrame({"text": texts}))
    preds = [1 if isinstance(p, dict) and p.get("label","").lower() in ("positive","1") else 0 for p in preds_raw]
    metrics = {"test_accuracy": round(accuracy_score(labels, preds), 4), "test_f1": round(f1_score(labels, preds, average="weighted"), 4), "test_samples": len(texts)}
    logger.info(f"\n{classification_report(labels, preds, target_names=['negative','positive'])}")
    logger.info(f"Metrics: {metrics}")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--model-uri", type=str, default=None)
    args = parser.parse_args()
    print(json.dumps(evaluate_model(run_id=args.run_id, model_uri=args.model_uri), indent=2))
