#!/usr/bin/env bash
set -euo pipefail

curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @scripts/sample_payload.json
echo

