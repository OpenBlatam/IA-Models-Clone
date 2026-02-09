# Multi-stage Dockerfile for AWS deployment
# Optimized for production with minimal image size

# Stage 1: Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
# Use requirements-core.txt for smaller image, or requirements.txt for full features
ARG REQUIREMENTS_FILE=requirements-core.txt
COPY ${REQUIREMENTS_FILE} requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the application
# Use workers in production, single worker with reload in development
CMD ["sh", "-c", "if [ \"$ENVIRONMENT\" = \"prod\" ]; then uvicorn manuales_hogar_ai.main:app --host 0.0.0.0 --port 8000 --workers ${WORKERS:-4}; else uvicorn manuales_hogar_ai.main:app --host 0.0.0.0 --port 8000 --reload; fi"]

