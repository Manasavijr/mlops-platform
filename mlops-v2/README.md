# MLOps Platform & LLM Inference System

Production-grade MLOps platform with FastAPI inference, MLflow experiment tracking, KS-test drift detection, and GCP Cloud Run deployment.

## Stack
Python · FastAPI · MLflow · DistilBERT · PyTorch · Docker · GCP Cloud Run · GitHub Actions

---

## Local Setup (No Docker needed)

### 1. Install dependencies
```bash
cd mlops-platform
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate sample data
```bash
python data/generate_data.py
```

### 3. Start MLflow (new terminal tab)
```bash
source .venv/bin/activate
mlflow server --host 0.0.0.0 --port 5001 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
```
Open http://localhost:5001

### 4. Train a model (first terminal)
```bash
source .venv/bin/activate
export MLFLOW_TRACKING_URI=http://localhost:5001
python ml/training/train.py --epochs 2 --max-samples 500 --auto-promote
```

### 5. Promote model to Production
```bash
python - <<'EOF'
import mlflow
mlflow.set_tracking_uri("http://localhost:5001")
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="sentiment-classifier-prod", version="1",
    stage="Production", archive_existing_versions=False)
print("Promoted to Production")
EOF
```

### 6. Start API server (new terminal tab)
```bash
source .venv/bin/activate
export MLFLOW_TRACKING_URI=http://localhost:5001
uvicorn app.main:app --reload --port 8080
```
Open http://localhost:8080/docs

---

## Test the API

```bash
# Health check
curl http://localhost:8080/health | python3 -m json.tool

# Predict sentiment
curl -X POST http://localhost:8080/api/v1/inference/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["This product is amazing!", "Terrible experience."]}' \
  | python3 -m json.tool

# Drift detection
curl -X POST http://localhost:8080/api/v1/drift/check \
  -H "Content-Type: application/json" \
  -d '{"scores": [0.5,0.51,0.49,0.52,0.48,0.50,0.53,0.47,0.51,0.49,0.50,0.48]}' \
  | python3 -m json.tool
```

---

## Run Tests
```bash
pytest tests/ -v
```

---

## Deploy to GCP Cloud Run

### Prerequisites
- GCP project with Cloud Run + Container Registry enabled
- `gcloud` CLI installed and authenticated
- Docker Desktop running

### Steps
```bash
export GCP_PROJECT_ID=your-project-id
gcloud auth configure-docker gcr.io
docker build --platform linux/amd64 -t gcr.io/$GCP_PROJECT_ID/mlops-platform:latest .
docker push gcr.io/$GCP_PROJECT_ID/mlops-platform:latest
gcloud run deploy mlops-platform \
  --image gcr.io/$GCP_PROJECT_ID/mlops-platform:latest \
  --platform managed --region us-central1 \
  --memory 4Gi --cpu 2 --allow-unauthenticated
```

---

## GitHub Actions CI/CD

Push to `main` → tests run → Docker image built → deployed to Cloud Run automatically.

**Required GitHub Secrets:**
| Secret | Description |
|---|---|
| `GCP_SA_KEY` | GCP Service Account JSON key |
| `GCP_PROJECT_ID` | Your GCP project ID |
| `MLFLOW_TRACKING_URI` | MLflow server URL |

---

## Architecture

```
GitHub Actions CI/CD
  └── lint → test → build → push GCR → deploy Cloud Run

GCP Cloud Run
  └── FastAPI App
        ├── /health        health checks
        ├── /api/v1/inference/predict   DistilBERT sentiment
        ├── /api/v1/drift/check         KS-test drift detection
        └── /api/v1/models              MLflow model registry

MLflow Server
  └── Experiment tracking + Model Registry
```
