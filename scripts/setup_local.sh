#!/usr/bin/env bash
set -euo pipefail
echo "Setting up MLOps Platform..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
python data/generate_data.py
mkdir -p /tmp/model_cache /tmp/hf_cache /tmp/predictions
echo ""
echo "Setup complete! Next:"
echo "  1. source .venv/bin/activate"
echo "  2. docker compose up -d mlflow"
echo "  3. python ml/training/train.py --epochs 2 --max-samples 500 --auto-promote"
echo "  4. uvicorn app.main:app --reload"
echo "  5. open http://localhost:8080/docs"
