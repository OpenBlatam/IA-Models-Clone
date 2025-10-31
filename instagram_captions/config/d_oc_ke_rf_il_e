# Multi-stage production Dockerfile for Instagram Captions API
FROM python:3.11-slim-bullseye as builder

# Build arguments
ARG APP_VERSION=4.0.0
ARG BUILD_DATE
ARG VCS_REF

# Labels for container metadata
LABEL maintainer="dev-team@company.com" \
      version="${APP_VERSION}" \
      description="Instagram Captions API - Production Ready" \
      build-date="${BUILD_DATE}" \
      vcs-ref="${VCS_REF}"

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-production.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-production.txt

# Production stage
FROM python:3.11-slim-bullseye as production

# Install runtime dependencies and security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tini \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create app directory and set ownership
WORKDIR /app
RUN chown -R appuser:appuser /app

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app/logs /app/data

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ENVIRONMENT=production \
    HOST=0.0.0.0 \
    PORT=8080 \
    WORKERS=4 \
    LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Run the application
CMD ["python", "production_api.py"]

# Security scanning metadata
LABEL org.opencontainers.image.source="https://github.com/company/instagram-captions-api" \
      org.opencontainers.image.documentation="https://docs.company.com/instagram-captions-api" \
      org.opencontainers.image.licenses="MIT" 