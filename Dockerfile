# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    ca-certificates \
    libgomp1 \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Install deps (cache-friendly) ----
COPY requirements-cloud.txt /app/requirements-cloud.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /app/requirements-cloud.txt

# ---- Copy project code + configs ----
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY configs /app/configs
COPY scripts /app/script

# If you want editable install (optional). If you don't need it, remove this line.
RUN pip install -e .

RUN mkdir -p /app/data

ENV CFG_PATH=/app/configs/run_all_cloud.yaml
CMD ["python", "-m",  "/app/scripts/run_pipeline.py", "--cfg", "/app/configs/run_all_cloud.yaml"]

