# ──────────────────────────────────────────────────────────────────────────────
# Med-V²QA API  —  Dockerfile
#
# Build:
#   docker build -t med-v2qa-api .
#
# Run (with checkpoint mounted at runtime):
#   docker run -p 8000:8000 \
#     -v $(pwd)/med_pretrain_29_rad_34.pth:/app/med_pretrain_29_rad_34.pth:ro \
#     med-v2qa-api
#
# The checkpoint is NOT baked into the image (2.2 GB) — mount it at runtime.
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# System-level dependencies for OpenCV, Pillow, scientific libs, and audio decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full source tree (checkpoint excluded via .dockerignore)
COPY . .

# Expose API port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
