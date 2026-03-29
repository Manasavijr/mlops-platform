#!/usr/bin/env bash
# deploy_gcp.sh — One-shot GCP Cloud Run deployment script
# Usage: ./scripts/deploy_gcp.sh [staging|production]
# Prerequisites: gcloud CLI authenticated, Docker installed

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="mlops-platform"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
ENVIRONMENT="${1:-staging}"
TAG="${ENVIRONMENT}-$(git rev-parse --short HEAD 2>/dev/null || echo latest)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MLOps Platform — GCP Cloud Run Deployment"
echo "  Environment : ${ENVIRONMENT}"
echo "  Project     : ${PROJECT_ID}"
echo "  Region      : ${REGION}"
echo "  Image tag   : ${TAG}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Authenticate & configure Docker ───────────────────────────────────────
echo "▶ Configuring Docker for GCR..."
gcloud auth configure-docker gcr.io --quiet
gcloud config set project "${PROJECT_ID}"

# ── 2. Build & push image ─────────────────────────────────────────────────────
echo "▶ Building Docker image..."
docker build \
  --platform linux/amd64 \
  --tag "${IMAGE}:${TAG}" \
  --tag "${IMAGE}:${ENVIRONMENT}" \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --cache-from "${IMAGE}:${ENVIRONMENT}" \
  .

echo "▶ Pushing image to GCR..."
docker push "${IMAGE}:${TAG}"
docker push "${IMAGE}:${ENVIRONMENT}"

# ── 3. Deploy to Cloud Run ────────────────────────────────────────────────────
echo "▶ Deploying to Cloud Run (${ENVIRONMENT})..."

# Environment-specific settings
if [[ "${ENVIRONMENT}" == "production" ]]; then
  MIN_INSTANCES=1
  MAX_INSTANCES=10
  MEMORY="4Gi"
  CPU=2
  SERVICE_SUFFIX=""
  MODEL_STAGE="Production"
else
  MIN_INSTANCES=0
  MAX_INSTANCES=3
  MEMORY="2Gi"
  CPU=1
  SERVICE_SUFFIX="-staging"
  MODEL_STAGE="Staging"
fi

FULL_SERVICE_NAME="${SERVICE_NAME}${SERVICE_SUFFIX}"

gcloud run deploy "${FULL_SERVICE_NAME}" \
  --image "${IMAGE}:${TAG}" \
  --platform managed \
  --region "${REGION}" \
  --memory "${MEMORY}" \
  --cpu "${CPU}" \
  --concurrency 80 \
  --min-instances "${MIN_INSTANCES}" \
  --max-instances "${MAX_INSTANCES}" \
  --timeout 300 \
  --port 8080 \
  --ingress all \
  --allow-unauthenticated \
  --set-env-vars "GCP_PROJECT_ID=${PROJECT_ID}" \
  --set-env-vars "GCP_REGION=${REGION}" \
  --set-env-vars "MLFLOW_MODEL_STAGE=${MODEL_STAGE}" \
  --set-env-vars "DEBUG=false" \
  --set-secrets "MLFLOW_TRACKING_URI=mlflow-tracking-uri:latest" \
  --set-secrets "GCS_BUCKET_NAME=gcs-bucket-name:latest" \
  --labels "environment=${ENVIRONMENT},app=${SERVICE_NAME}"

# ── 4. Get service URL and smoke test ────────────────────────────────────────
SERVICE_URL=$(gcloud run services describe "${FULL_SERVICE_NAME}" \
  --region="${REGION}" \
  --format="value(status.url)")

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Deployment complete!"
echo "  Service URL: ${SERVICE_URL}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "▶ Running smoke tests..."
sleep 5

# Liveness check
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${SERVICE_URL}/health/live")
if [[ "${HTTP_STATUS}" == "200" ]]; then
  echo "  ✅ /health/live → ${HTTP_STATUS}"
else
  echo "  ❌ /health/live → ${HTTP_STATUS}"
  exit 1
fi

# Test inference endpoint
INFERENCE_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
  -X POST "${SERVICE_URL}/api/v1/inference/predict" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["This is a great deployment test."]}')

if [[ "${INFERENCE_STATUS}" == "200" ]]; then
  echo "  ✅ /api/v1/inference/predict → ${INFERENCE_STATUS}"
else
  echo "  ⚠️  /api/v1/inference/predict → ${INFERENCE_STATUS} (model may still be loading)"
fi

echo ""
echo "  📊 MLflow UI   : Set MLFLOW_TRACKING_URI and open http://localhost:5000"
echo "  📖 API Docs    : ${SERVICE_URL}/docs"
echo "  ❤️  Health     : ${SERVICE_URL}/health"
echo ""
