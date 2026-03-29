# MLOps Platform & LLM Inference System

A production-grade MLOps platform built for portfolio demonstration. Combines experiment tracking, automated drift detection, REST inference serving, containerized deployment, and a full CI/CD pipeline — all wired together end-to-end.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GitHub Actions CI/CD                          │
│   lint → test → build → push GCR → deploy Cloud Run (staging/prod) │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │         GCP Cloud Run               │
              │   ┌──────────────────────────────┐  │
              │   │      FastAPI Application      │  │
              │   │  /health  /inference  /drift  │  │
              │   │       /models  /docs          │  │
              │   └──────────┬───────────────────┘  │
              └──────────────┼──────────────────────┘
                             │
        ┌────────────────────┼──────────────────────┐
        │                    │                       │
┌───────▼──────┐   ┌─────────▼─────────┐   ┌───────▼────────┐
│ MLflow Server│   │  HuggingFace Hub   │   │ Drift Detector │
│ (experiment  │   │  DistilBERT SST-2  │   │  KS-test on    │
│  tracking +  │   │  (fallback load)   │   │  score dist.   │
│  model reg.) │   └────────────────────┘   └───────┬────────┘
└──────────────┘                                     │
                                            Triggers retraining
                                            GitHub Actions workflow
```

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Experiment Tracking | MLflow 2.18 |
| Model | DistilBERT (HuggingFace Transformers) |
| Drift Detection | SciPy KS-test |
| Containerization | Docker (multi-stage build) |
| Cloud Deployment | GCP Cloud Run |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus + structured JSON logs |
| Config | Pydantic Settings (env-driven) |

---

## Folder Structure

```
mlops-platform/
├── app/                          # FastAPI application
│   ├── main.py                   # Entry point, middleware, lifespan
│   ├── core/
│   │   ├── config.py             # Pydantic Settings (all env vars)
│   │   ├── logging_config.py     # Structured JSON logging (GCP-compatible)
│   │   └── model_registry.py     # MLflow + HuggingFace model loader
│   ├── api/routes/
│   │   ├── inference.py          # POST /api/v1/inference/predict
│   │   ├── health.py             # GET /health, /health/live, /health/ready
│   │   ├── models.py             # Model registry management endpoints
│   │   └── drift.py              # Drift check + history endpoints
│   └── schemas/
│       └── schemas.py            # All Pydantic request/response models
│
├── ml/
│   ├── training/
│   │   └── train.py              # Full training pipeline with MLflow logging
│   ├── evaluation/
│   │   └── evaluate.py           # Evaluation metrics + champion/challenger compare
│   └── drift/
│       └── detector.py           # KS-test drift detector
│
├── data/
│   ├── generate_data.py          # Synthetic dataset generator
│   └── raw/                      # train.csv, val.csv, test.csv, drift_test.csv
│
├── tests/
│   └── unit/
│       └── test_api.py           # 20+ unit tests for all endpoints + components
│
├── .github/workflows/
│   └── ci-cd.yml                 # Full CI/CD: lint→test→build→deploy
│
├── scripts/
│   ├── setup_local.sh            # Bootstrap local dev environment
│   └── deploy_gcp.sh             # Manual GCP Cloud Run deploy script
│
├── monitoring/
│   └── prometheus.yml            # Prometheus scrape config
│
├── Dockerfile                    # Multi-stage production build
├── docker-compose.yml            # Local dev: API + MLflow server + Prometheus
├── requirements.txt
├── pyproject.toml                # pytest + ruff config
└── .env.example                  # Environment variable template
```

---

## Local Development Setup

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- Git

### Step 1 — Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/mlops-platform.git
cd mlops-platform

# Automated setup (creates venv, installs deps, generates data)
chmod +x scripts/setup_local.sh
./scripts/setup_local.sh

# Or manually:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python data/generate_data.py
```

### Step 2 — Start MLflow Server

```bash
# Option A: Docker Compose (recommended)
docker compose up -d mlflow
# MLflow UI → http://localhost:5000

# Option B: Direct
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
```

### Step 3 — Train & Register a Model

```bash
source .venv/bin/activate

python ml/training/train.py \
  --epochs 3 \
  --batch-size 16 \
  --max-samples 2000 \
  --auto-promote       # auto-promotes to Staging in MLflow
```

You'll see per-epoch metrics logged to MLflow. Open `http://localhost:5000` to explore experiments, runs, and the registered model.

To promote to Production manually:
```bash
# Via MLflow UI, or via the API after starting the server:
curl -X POST http://localhost:8080/api/v1/models/promote \
  -H "Content-Type: application/json" \
  -d '{"model_name": "sentiment-classifier-prod", "version": "1", "stage": "Production"}'
```

### Step 4 — Start the API Server

```bash
# Development (hot reload)
uvicorn app.main:app --reload --port 8080

# Or via Docker Compose (full stack)
docker compose up
```

API is live at `http://localhost:8080`

**Swagger UI:** `http://localhost:8080/docs`
**ReDoc:** `http://localhost:8080/redoc`

---

## API Reference

### Inference

```bash
# Single text prediction
curl -X POST http://localhost:8080/api/v1/inference/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["This product is absolutely fantastic!"]}'

# Response:
{
  "predictions": [
    {
      "text": "This product is absolutely fantastic!",
      "label": "positive",
      "score": 0.9987,
      "model_version": "1"
    }
  ],
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
  "latency_ms": 42.3,
  "timestamp": "2025-03-28T10:00:00Z"
}

# Batch (up to 32 texts)
curl -X POST http://localhost:8080/api/v1/inference/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great!", "Terrible.", "Average product."]}'
```

### Health Checks

```bash
curl http://localhost:8080/health          # Full status + model info
curl http://localhost:8080/health/live     # Liveness (Cloud Run probe)
curl http://localhost:8080/health/ready    # Readiness (model loaded?)
```

### Drift Detection

```bash
# Check for distribution drift in recent prediction scores
curl -X POST http://localhost:8080/api/v1/drift/check \
  -H "Content-Type: application/json" \
  -d '{"scores": [0.92, 0.45, 0.51, 0.49, 0.50, 0.48, 0.52, 0.47, 0.53, 0.46]}'

# Response:
{
  "drift_detected": true,
  "reports": [{"feature": "prediction_score", "statistic": 0.412, "p_value": 0.001, ...}],
  "retraining_triggered": true
}

# Get reference distribution stats
curl http://localhost:8080/api/v1/drift/reference-stats

# View drift history
curl http://localhost:8080/api/v1/drift/history
```

### Model Management

```bash
# Current loaded model
curl http://localhost:8080/api/v1/models/current

# List all registered versions
curl http://localhost:8080/api/v1/models/versions

# Hot-reload model (after promoting new version)
curl -X POST http://localhost:8080/api/v1/models/reload

# List MLflow experiments
curl http://localhost:8080/api/v1/models/experiments

# List recent training runs
curl http://localhost:8080/api/v1/models/runs/sentiment-classifier
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=app --cov=ml --cov-report=html

# Specific test class
pytest tests/unit/test_api.py::TestInferenceEndpoint -v

# Drift tests only
pytest tests/unit/test_api.py::TestDriftDetector -v
```

---

## Docker

```bash
# Build image
docker build -t mlops-platform:local .

# Run container
docker run -p 8080:8080 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e HF_MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english \
  mlops-platform:local

# Full stack (API + MLflow + Prometheus)
docker compose up

# With monitoring profile
docker compose --profile monitoring up
```

---

## GCP Cloud Run Deployment

### Prerequisites

1. GCP project with billing enabled
2. APIs enabled: Cloud Run, Container Registry, Secret Manager
3. `gcloud` CLI installed and authenticated
4. Service account with roles: `Cloud Run Admin`, `Storage Admin`, `Secret Manager Accessor`

### One-Shot Deploy

```bash
export GCP_PROJECT_ID=your-project-id

# Deploy to staging
./scripts/deploy_gcp.sh staging

# Deploy to production
./scripts/deploy_gcp.sh production
```

### Manual gcloud Deploy

```bash
# Build and push
gcloud auth configure-docker gcr.io
docker build --platform linux/amd64 -t gcr.io/$PROJECT_ID/mlops-platform:latest .
docker push gcr.io/$PROJECT_ID/mlops-platform:latest

# Deploy
gcloud run deploy mlops-platform \
  --image gcr.io/$PROJECT_ID/mlops-platform:latest \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 10 \
  --allow-unauthenticated \
  --set-env-vars GCP_PROJECT_ID=$PROJECT_ID
```

### GitHub Actions Secrets Required

| Secret | Description |
|---|---|
| `GCP_SA_KEY` | GCP Service Account JSON key |
| `GCP_PROJECT_ID` | Your GCP project ID |
| `MLFLOW_TRACKING_URI` | MLflow server URL |
| `GCS_BUCKET_NAME` | GCS bucket for artifacts |
| `PRODUCTION_SERVICE_URL` | Deployed Cloud Run URL |

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci-cd.yml`) runs on every push:

```
Push to any branch
    └── lint (ruff + mypy)
        └── test (pytest, 20+ tests)
            └── [main/develop only] build & push Docker image to GCR
                ├── [develop] deploy to staging Cloud Run + smoke test
                └── [main] deploy to production Cloud Run (requires approval)
                    └── Smoke test production URL
```

**Automated Retraining** is triggered when:
- The drift detector fires (`p_value < 0.05` on prediction scores)
- A `workflow_dispatch` is manually triggered in GitHub Actions
- A Pub/Sub message is published to the retrain topic (production setup)

---

## MLOps Workflow

```
1. Data arrives / drift detected
       ↓
2. Training pipeline runs (ml/training/train.py)
   - Logs params, metrics, artifacts to MLflow
       ↓
3. Model registered to MLflow Model Registry
   - Stage: None → Staging
       ↓
4. Evaluation runs (ml/evaluation/evaluate.py)
   - Champion vs Challenger comparison
       ↓
5. Model promoted: Staging → Production
   (via API endpoint or MLflow UI)
       ↓
6. Hot-reload: POST /api/v1/models/reload
   - No downtime, no restart needed
       ↓
7. Drift detector resets reference distribution
```

---

## Key Design Decisions

**Why FastAPI?** Async-native, auto-generated OpenAPI docs, Pydantic validation, production-proven with Uvicorn workers.

**Why MLflow over W&B?** Self-hostable, no external dependency, works perfectly with GCS artifact store, free.

**Why KS-test for drift?** Non-parametric, no distribution assumptions, interpretable p-value threshold, fast on prediction score vectors.

**Why Cloud Run over GKE?** Zero infrastructure management, scales to zero (cost-efficient for portfolio), managed TLS, faster cold starts than Lambda.

**Multi-stage Dockerfile** — builder stage installs all Python wheels; runtime stage copies only the compiled wheels and app source, keeping the final image minimal (~1.2GB with PyTorch).

---

## Resume Talking Points

- Built end-to-end MLOps pipeline: experiment tracking (MLflow) → model registry → REST serving (FastAPI) → drift-triggered retraining → CI/CD (GitHub Actions) → cloud deployment (GCP Cloud Run)
- Implemented KS-test statistical drift detection that automatically triggers retraining when prediction score distribution shifts beyond p < 0.05
- Containerized with multi-stage Docker build (builder/runtime separation); deployed to GCP Cloud Run with min-instances=1 for zero cold-start latency
- Achieved full observability via structured JSON logging (GCP Cloud Logging compatible), Prometheus metrics, and per-request latency headers
- Wrote 20+ unit tests covering inference, drift detection, schema validation, and model management endpoints

---

## Screenshots Guide

When adding screenshots to this README, capture:

1. **MLflow Experiments UI** — `http://localhost:5000` showing experiment list, run metrics table, loss curves
2. **MLflow Model Registry** — registered model with Staging/Production stages
3. **Swagger UI** — `http://localhost:8080/docs` showing all endpoints
4. **Inference Response** — curl or Postman showing predict response with label + score
5. **Drift Detection** — `/api/v1/drift/check` response with `drift_detected: true`
6. **GitHub Actions** — successful CI/CD pipeline run with all green checks
7. **GCP Cloud Run** — deployed service page in GCP Console

---

## License

MIT — free to use, fork, and build on for your own portfolio.
