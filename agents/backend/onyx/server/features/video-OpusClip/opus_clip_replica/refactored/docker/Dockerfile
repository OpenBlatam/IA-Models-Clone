# Multi-stage Dockerfile for Refactored Opus Clip
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY refactored/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY refactored/ .

# Create necessary directories
RUN mkdir -p uploads static/css static/js templates logs

# Set permissions
RUN chmod -R 755 /app

# Expose ports
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Default command
CMD ["python", "api/refactored_opus_clip_api.py"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    flake8 \
    mypy \
    jupyter

# Copy development configuration
COPY refactored/docker/development.yaml config.yaml

# Override command for development
CMD ["python", "api/refactored_opus_clip_api.py", "--reload"]

# Production stage
FROM base as production

# Create non-root user
RUN groupadd -r opusclip && useradd -r -g opusclip opusclip

# Change ownership of app directory
RUN chown -R opusclip:opusclip /app

# Switch to non-root user
USER opusclip

# Copy production configuration
COPY refactored/docker/production.yaml config.yaml

# Production optimizations
ENV PYTHONOPTIMIZE=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

# Override command for production
CMD ["gunicorn", "api.refactored_opus_clip_api:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]


