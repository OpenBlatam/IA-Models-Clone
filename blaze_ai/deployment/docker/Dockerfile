# Multi-stage production Dockerfile for Blaze AI
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r blazeai && useradd -r -g blazeai blazeai

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY . .

# Set ownership
RUN chown -R blazeai:blazeai /app

# Switch to non-root user
USER blazeai

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8001 8002

# Production command
CMD ["python", "-m", "uvicorn", "blaze_ai.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
