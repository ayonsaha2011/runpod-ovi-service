# =============================================================================
# Ovi 1.1 RunPod Serverless - Production Dockerfile (OPTIMIZED)
# =============================================================================
# Lightweight image - models downloaded at runtime to network volume
# Target: RunPod Serverless with A100/H100/Blackwell GPUs
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base image with CUDA and Python
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS base

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
# Stage 2: Install Python dependencies
# -----------------------------------------------------------------------------
FROM base AS python-deps

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.8 for Blackwell (sm_120) support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install Flash Attention 2 from prebuilt wheel (required by Ovi)
# Using prebuilt wheel instead of building from source (50+ min -> ~1 min)
RUN pip install flash-attn --no-build-isolation || \
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl || \
    echo "Flash Attention installation failed, will try to build from source" && \
    pip install flash-attn --no-build-isolation

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 3: Production image (NO MODEL DOWNLOAD - uses network volume)
# -----------------------------------------------------------------------------
FROM python-deps AS production

LABEL maintainer="AI Platform"
LABEL version="1.1.0"
LABEL description="Ovi 1.1 Video Generation - RunPod Serverless (Lightweight)"

# Clone Ovi repository
RUN git clone --depth 1 https://github.com/character-ai/Ovi.git /app/Ovi

# Copy service code and scripts
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/

# Set working directory to Ovi root (required for relative config paths in Ovi code)
WORKDIR /app/Ovi

# Set Python path to include Ovi and src
ENV PYTHONPATH="/app/src:/app/Ovi:${PYTHONPATH}"
ENV OVI_PATH="/app/Ovi"

# Models are stored on RunPod Network Volume (mounted at /runpod-volume)
# On first cold start, models will be downloaded automatically
ENV OVI_CKPT_DIR="/runpod-volume/models"
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
