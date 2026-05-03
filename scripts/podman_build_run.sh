#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

IMAGE_NAME="heart-disease-api:latest"
CONTAINER_NAME="heart-disease-api"

podman build -t "${IMAGE_NAME}" .
podman rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
podman run -d --name "${CONTAINER_NAME}" -p 8000:8000 "${IMAGE_NAME}"

echo "Container started: ${CONTAINER_NAME}"
echo "Health check: curl http://localhost:8000/health"

