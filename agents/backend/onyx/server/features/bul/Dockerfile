# Enhanced Multi-stage Dockerfile for BUL System
# Optimized for production with security, performance, and observability

# =============================================================================
# Builder Stage - Optimized for dependency installation
# =============================================================================
FROM python:3.11-slim as builder

# Set build environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_INDEX_URL=https://pypi.org/simple/ \
    PIP_EXTRA_INDEX_URL=""

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements files
COPY requirements.txt requirements-prod.txt requirements-dev.txt ./

# Install production dependencies
RUN pip install --no-deps -r requirements-prod.txt

# =============================================================================
# Security Stage - Security scanning and validation
# =============================================================================
FROM builder as security

# Install security scanning tools
RUN pip install bandit safety

# Copy source code for security scanning
COPY . /app
WORKDIR /app

# Run security scans
RUN bandit -r . -f json -o /tmp/bandit-report.json || true
RUN safety check --json --output /tmp/safety-report.json || true

# =============================================================================
# Production Stage - Optimized runtime image
# =============================================================================
FROM python:3.11-slim as production

# Set production environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    BUL_ENV=production \
    BUL_LOG_LEVEL=INFO \
    BUL_WORKERS=4 \
    BUL_HOST=0.0.0.0 \
    BUL_PORT=8000

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user with specific UID/GID for security
RUN groupadd -r -g 1001 bul && \
    useradd -r -u 1001 -g bul -d /app -s /bin/bash bul

# Create application directory structure
WORKDIR /app
RUN mkdir -p /app/logs /app/tmp /app/cache /app/data && \
    chown -R bul:bul /app

# Copy application code with proper ownership
COPY --chown=bul:bul . .

# Copy security reports from security stage
COPY --from=security /tmp/bandit-report.json /app/security/
COPY --from=security /tmp/safety-report.json /app/security/

# Set proper permissions
RUN chmod -R 755 /app && \
    chmod -R 777 /app/logs /app/tmp /app/cache && \
    chmod 644 /app/security/*.json

# Switch to non-root user
USER bul

# Create startup script for better process management
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Wait for dependencies\n\
echo "Waiting for dependencies..."\n\
sleep 2\n\
\n\
# Start the application\n\
echo "Starting BUL API..."\n\
exec python main.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose ports
EXPOSE 8000 8001

# Add labels for better container management
LABEL maintainer="BUL Team" \
      version="2.0.0" \
      description="Business Universal Language - Document Generation API" \
      org.opencontainers.image.title="BUL API" \
      org.opencontainers.image.description="AI-powered document generation for SMEs" \
      org.opencontainers.image.version="2.0.0" \
      org.opencontainers.image.created="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      org.opencontainers.image.source="https://github.com/bul-system/bul" \
      org.opencontainers.image.licenses="MIT"

# Enhanced health check with multiple endpoints
HEALTHCHECK --interval=30s --timeout=15s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || \
        curl -f http://localhost:8000/ || exit 1

# Use startup script for better process management
CMD ["/app/start.sh"]

# =============================================================================
# Development Stage - For development and testing
# =============================================================================
FROM production as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN pip install -r requirements-dev.txt

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    net-tools \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Switch back to app user
USER bul

# Override environment for development
ENV BUL_ENV=development \
    BUL_LOG_LEVEL=DEBUG \
    BUL_WORKERS=1 \
    BUL_RELOAD=true

# Development command with hot reload
CMD ["python", "main.py"]

# =============================================================================
# Testing Stage - For running tests
# =============================================================================
FROM development as testing

# Install testing dependencies
RUN pip install pytest pytest-asyncio pytest-cov pytest-mock

# Copy test configuration
COPY pytest.ini ./

# Set testing environment
ENV BUL_ENV=testing \
    BUL_LOG_LEVEL=WARNING

# Default command for testing
CMD ["pytest", "tests/", "-v", "--cov=.", "--cov-report=html"]




