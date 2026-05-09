#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] health endpoint"
curl -sf "http://localhost:8000/health"
echo

echo "[2/3] predict endpoint"
curl -sf -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
echo

echo "[3/3] metrics endpoint"
curl -sf "http://localhost:8000/metrics" | grep -E "api_requests_total|prediction_latency_seconds"
echo

