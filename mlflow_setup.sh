#!/usr/bin/env bash
# MLflow tracking infrastructure bootstrap.
# Local mode (default): file store + local artifact root.
# Server mode: pass --server to run a tracking server backed by SQLite,
# suitable for a small team; swap BACKEND_URI/ARTIFACT_ROOT for
# Postgres + GCS/S3 in production.
set -euo pipefail

MODE="${1:-local}"
BACKEND_URI="${MLFLOW_BACKEND_URI:-sqlite:///mlflow.db}"
ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-./mlruns}"
HOST="${MLFLOW_HOST:-0.0.0.0}"
PORT="${MLFLOW_PORT:-5000}"

if ! command -v mlflow >/dev/null 2>&1; then
  echo "mlflow not found — installing..."
  pip install --quiet mlflow
fi

if [[ "$MODE" == "--server" ]]; then
  echo "Starting MLflow tracking server on ${HOST}:${PORT}"
  echo "  backend:   ${BACKEND_URI}"
  echo "  artifacts: ${ARTIFACT_ROOT}"
  exec mlflow server \
    --backend-store-uri "${BACKEND_URI}" \
    --default-artifact-root "${ARTIFACT_ROOT}" \
    --host "${HOST}" --port "${PORT}"
else
  mkdir -p mlruns
  echo "Local file-store tracking configured at ./mlruns"
  echo "Set MLFLOW_TRACKING_URI=file:./mlruns (already the config default)."
fi
