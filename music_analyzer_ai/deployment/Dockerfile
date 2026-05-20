# Multi-stage Dockerfile for Music Analyzer AI
# Optimized for production deployments with minimal image size and fast builds

# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt deployment/requirements-serverless.txt* ./

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --user -r requirements.txt && \
    if [ -f requirements-serverless.txt ]; then \
        pip install --no-cache-dir --user -r requirements-serverless.txt; \
    fi

# Stage 2: Runtime image
FROM python:3.11-slim

# Add metadata
LABEL maintainer="Blatam Academy" \
      version="2.21.0" \
      description="Music Analyzer AI - Advanced music analysis system"

WORKDIR /app

# Install runtime dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy Python dependencies from builder
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser . .

# Make sure scripts are executable
RUN chmod +x /app/deployment/scripts/*.sh 2>/dev/null || true

# Set environment variables
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8010

# Health check with proper user permissions
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8010/health || exit 1

# Use uvicorn with production settings
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010", "--workers", "4", "--loop", "uvloop"]

