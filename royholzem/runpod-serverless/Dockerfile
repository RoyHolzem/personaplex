# RunPod Serverless Dockerfile for PersonaPlex
# Base image with CUDA support
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    libopus-dev \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy requirements and install Python dependencies
COPY royholzem/runpod-serverless/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy entire repository
COPY . /app/

# Install PersonaPlex moshi package
RUN pip install --no-cache-dir /app/moshi/.

# Set environment variables for persistent caching on network volume
ENV HF_HOME=/runpod-volume/hf
ENV HF_HUB_CACHE=/runpod-volume/hf
ENV TRANSFORMERS_CACHE=/runpod-volume/hf
ENV TORCH_HOME=/runpod-volume/torch
ENV MODEL_DIR=/runpod-volume/models/personaplex
ENV PYTHONUNBUFFERED=1

# RunPod serverless does not need exposed ports
# The handler is invoked directly

# Start the handler
CMD ["python3", "-u", "/app/royholzem/runpod-serverless/handler.py"]
