import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/mlruns_test"
os.environ["DEBUG"] = "true"


@pytest.fixture
def mock_registry():
    r = MagicMock()
    r.is_loaded = True
    r.model_name = "test-model"
    r.model_version = "1"
    r.info = {
        "model_name": "test-model",
        "model_version": "1",
        "model_uri": None,
        "is_loaded": True,
        "backend": "huggingface",
    }
    r.predict.return_value = [
        {"text": "great", "label": "positive", "score": 0.95, "model_version": "1"}
    ]
    return r


@pytest.fixture
def client(mock_registry):
    with patch(
        "app.core.model_registry.ModelRegistry.load_production_model",
        return_value=None,
    ):
        from app.main import app
        from fastapi.testclient import TestClient
        app.state.model_registry = mock_registry
        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.model_registry = mock_registry
            yield c


def test_liveness(client):
    assert client.get("/health/live").status_code == 200


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict(client, mock_registry):
    mock_registry.is_loaded = True
    mock_registry.predict.return_value = [
        {"text": "great product", "label": "positive", "score": 0.95, "model_version": "1"}
    ]
    client.app.state.model_registry = mock_registry
    r = client.post("/api/v1/inference/predict", json={"texts": ["great product"]})
    assert r.status_code == 200
    assert r.json()["predictions"][0]["label"] == "positive"


def test_predict_empty_rejected(client):
    assert client.post("/api/v1/inference/predict", json={"texts": ["   "]}).status_code == 422


def test_predict_too_many(client):
    assert client.post("/api/v1/inference/predict", json={"texts": ["t"] * 33}).status_code == 422


def test_predict_model_not_loaded(client, mock_registry):
    mock_registry.is_loaded = False
    client.app.state.model_registry = mock_registry
    assert client.post("/api/v1/inference/predict", json={"texts": ["test"]}).status_code == 503


def test_drift_check(client):
    scores = list(np.random.default_rng(1).beta(8, 2, 100))
    r = client.post("/api/v1/drift/check", json={"scores": scores})
    assert r.status_code == 200
    assert r.json()["sample_size"] == 100


def test_drift_too_few(client):
    assert client.post("/api/v1/drift/check", json={"scores": [0.9, 0.8]}).status_code == 422


def test_reference_stats(client):
    assert "mean" in client.get("/api/v1/drift/reference-stats").json()


def test_drift_history(client):
    assert "history" in client.get("/api/v1/drift/history").json()
