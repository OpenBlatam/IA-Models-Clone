# Gamma App - AI-Powered Content Generation System
# Multi-stage Docker build for production deployment

# Stage 1: Base Python environment
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Stage 2: Dependencies
FROM base as dependencies

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Development
FROM dependencies as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    isort \
    flake8 \
    mypy

# Copy source code
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 8000

# Development command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 4: Production
FROM dependencies as production

# Create non-root user
RUN groupadd -r gamma && useradd -r -g gamma gamma

# Copy source code
COPY --chown=gamma:gamma . .

# Create necessary directories
RUN mkdir -p uploads data logs && \
    chown -R gamma:gamma uploads data logs

# Switch to non-root user
USER gamma

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]



























