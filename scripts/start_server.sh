#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Med-V²QA — production server startup script
#  Used by: systemd service medv2qa.service
# ─────────────────────────────────────────────────────────────
set -euo pipefail

CONDA_BASE="/home/long/miniconda3"
CONDA_ENV="medv2qa"
APP_DIR="/mnt/Data/Long-Data/myProjects/med-v2qa"
HOST="127.0.0.1"
PORT="8765"
WORKERS=1  

# Activate conda env
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${APP_DIR}"

exec uvicorn api.main:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    --log-level info \
    --access-log
