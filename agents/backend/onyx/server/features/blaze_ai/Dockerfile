# Enhanced Blaze AI Dockerfile
# Multi-stage build for production-ready deployment

# =============================================================================
# Stage 1: Base Image with System Dependencies
# =============================================================================
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
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r blazeai && useradd -r -g blazeai blazeai

# Create application directory
WORKDIR /app

# =============================================================================
# Stage 2: Dependencies Installation
# =============================================================================
FROM base as dependencies

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 3: Application Build
# =============================================================================
FROM base as build

# Copy dependencies from previous stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs cache outputs && \
    chown -R blazeai:blazeai /app

# =============================================================================
# Stage 4: Production Image
# =============================================================================
FROM base as production

# Copy dependencies and application from build stage
COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY --from=build /app /app

# Set ownership
RUN chown -R blazeai:blazeai /app

# Switch to non-root user
USER blazeai

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "main.py"]

# =============================================================================
# Stage 5: Development Image (Optional)
# =============================================================================
FROM build as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    isort \
    ruff \
    mypy

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for development
CMD ["python", "main.py", "--dev"]

# =============================================================================
# Stage 6: Testing Image (Optional)
# =============================================================================
FROM build as testing

# Install testing dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    requests \
    httpx

# Copy test files
COPY test_enhanced_features.py .
COPY demo_enhanced_features.py .

# Expose ports
EXPOSE 8000 9090

# Default command for testing
CMD ["python", "test_enhanced_features.py"]

# =============================================================================
# Labels and Metadata
# =============================================================================
LABEL maintainer="Blaze AI Team" \
      version="2.1.0" \
      description="Enhanced Blaze AI with Enterprise-Grade Features" \
      org.opencontainers.image.title="Blaze AI Enhanced" \
      org.opencontainers.image.description="Advanced AI Module with Enterprise-Grade Security and Monitoring" \
      org.opencontainers.image.version="2.1.0" \
      org.opencontainers.image.vendor="Blaze AI" \
      org.opencontainers.image.licenses="MIT"

# =============================================================================
# Environment Variables
# =============================================================================
ENV APP_ENVIRONMENT=production \
    DEBUG=false \
    LOG_LEVEL=INFO \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_WORKERS=4 \
    REDIS_HOST=localhost \
    REDIS_PORT=6379 \
    REDIS_DB=0 \
    PROMETHEUS_ENABLED=true \
    PROMETHEUS_PORT=9090

# =============================================================================
# Volume Mounts
# =============================================================================
VOLUME ["/app/logs", "/app/cache", "/app/outputs"]

# =============================================================================
# Security Considerations
# =============================================================================
# - Non-root user (blazeai)
# - Minimal base image (python:3.11-slim)
# - No unnecessary packages
# - Health checks
# - Proper file permissions
# - Environment variable configuration
