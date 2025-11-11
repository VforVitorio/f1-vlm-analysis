# ==============================================================================
# VLM Image Captioning - Dockerfile
# ==============================================================================
# Multi-model VLM system for F1 dataset captioning
# Models: BLIP-base, GIT-base, ViT2DistilGPT2, Swin-Tiny-DistilGPT2
# Target GPU: GTX 1050 (4GB VRAM)
# ==============================================================================

# Base image with CUDA 11.8 support
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Environment Configuration
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0
ENV CUDA_HOME=/usr/local/cuda
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface

# System Dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Working Directory
WORKDIR /app

# Python Dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Download NLTK data for evaluation
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

# Copy source code
COPY src/ ./src/

# Create directories
RUN mkdir -p /app/.cache/huggingface \
    results/blip \
    results/git-base \
    results/vit2distilgpt2 \
    results/swin-tiny \
    results/comparison

# Volume Mount Points
VOLUME ["/app/dataset"]
VOLUME ["/app/results"]

# Default Command
CMD ["python3", "-m", "src.vlm_inference", "--all"]