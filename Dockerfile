# =============================================================================
# Ovi 1.1 RunPod Serverless - Production Dockerfile
# =============================================================================
# Multi-stage build for optimized image size
# Target: RunPod Serverless with A100/H100 GPUs
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base image with CUDA and Python
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# Stage 2: Build Flash Attention
# -----------------------------------------------------------------------------
FROM base AS flash-attn-builder

# Install build dependencies
# Cache bust: 2026-01-15
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Build Flash Attention 2 from source
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV MAX_JOBS=4

RUN pip install flash-attn==2.7.4.post1 --no-build-isolation

# -----------------------------------------------------------------------------
# Stage 3: Install Python dependencies
# -----------------------------------------------------------------------------
FROM base AS python-deps

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.1
RUN pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Flash Attention from builder stage
COPY --from=flash-attn-builder /usr/local/lib/python3.11/dist-packages/flash_attn* /usr/local/lib/python3.11/dist-packages/

# -----------------------------------------------------------------------------
# Stage 4: Clone Ovi repository
# -----------------------------------------------------------------------------
FROM python-deps AS ovi-clone

WORKDIR /app

# Clone Ovi repository
RUN git clone --depth 1 https://github.com/character-ai/Ovi.git /app/Ovi

# -----------------------------------------------------------------------------
# Stage 5: Download models
# -----------------------------------------------------------------------------
FROM ovi-clone AS model-download

WORKDIR /app

# Copy download script
COPY scripts/download_models.py /app/scripts/download_models.py

# Download all model weights (this will be LARGE ~30GB+)
# Uses HF_TOKEN from environment if available (for gated models)
# Build with: DOCKER_BUILDKIT=1 docker build --secret id=hf_token,env=HF_TOKEN -t image .
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    python3 /app/scripts/download_models.py \
    --output-dir /models \
    --models 720x720_5s 960x960_5s 960x960_10s

# -----------------------------------------------------------------------------
# Stage 6: Production image
# -----------------------------------------------------------------------------
FROM python-deps AS production

LABEL maintainer="AI Platform"
LABEL version="1.0.0"
LABEL description="Ovi 1.1 Video Generation - RunPod Serverless"

WORKDIR /app

# Copy Ovi source code
COPY --from=ovi-clone /app/Ovi /app/Ovi

# Copy downloaded models
COPY --from=model-download /models /models

# Copy service code
COPY src/ /app/src/
COPY configs/ /app/configs/

# Set Python path to include Ovi
ENV PYTHONPATH="/app/src:/app/Ovi:${PYTHONPATH}"
ENV OVI_PATH="/app/Ovi"
ENV OVI_CKPT_DIR="/models"
ENV OVI_MODEL_NAME="960x960_10s"

# RunPod specific environment
ENV RUNPOD_DEBUG_LEVEL="INFO"

# Default Cloudflare R2 environment variables (to be overridden at runtime)
ENV R2_ACCOUNT_ID=""
ENV R2_ACCESS_KEY_ID=""
ENV R2_SECRET_ACCESS_KEY=""
ENV R2_BUCKET_NAME="ovi-videos"
ENV R2_PUBLIC_URL=""

# Performance settings
ENV OMP_NUM_THREADS=4
ENV CUDA_VISIBLE_DEVICES=0

# Pre-load model at startup (optional, increases cold start time but faster first request)
ENV PRELOAD_MODEL="false"

# Memory optimization (set to true for 24GB GPUs)
ENV OVI_CPU_OFFLOAD="false"
ENV OVI_FP8="false"
ENV OVI_QINT8="false"

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python3 -c "import torch; print(torch.cuda.is_available())" || exit 1

# Run the handler
CMD ["python3", "-u", "/app/src/handler.py"]
