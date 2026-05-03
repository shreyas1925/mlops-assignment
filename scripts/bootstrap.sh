#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"
python scripts/run_training_pipeline.py

echo "Bootstrap complete. Artifacts available under artifacts/."

