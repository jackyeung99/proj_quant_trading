# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# ---- System deps ----
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

# ---- Install Python deps ----
COPY requirements-cloud.txt /app/requirements-cloud.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /app/requirements-cloud.txt

# ---- Copy project ----
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY configs /app/configs
COPY scripts /app/scripts

# optional editable install
RUN pip install -e .

# runtime dirs
RUN mkdir -p /app/data /app/artifacts

# ---- DEFAULT ENTRYPOINT (NOT HARDCODED CFG) ----
ENTRYPOINT ["python", "/app/scripts/run_pipeline.py"]

# ---- DEFAULT ARGS (can be overridden by ECS/ECR) ----
CMD ["--cfg", "/app/configs/deployments/sector_deployment_long_only.yaml"]