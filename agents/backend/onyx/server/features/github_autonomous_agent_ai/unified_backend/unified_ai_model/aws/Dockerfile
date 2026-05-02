FROM python:3.11-slim

LABEL maintainer="Onyx Team"
LABEL description="Unified AI Model API - FastAPI backend for AI chat and agents"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Environment variables
ENV UNIFIED_AI_HOST=0.0.0.0
ENV UNIFIED_AI_PORT=8050
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8050/api/v1/health || exit 1

# Run the application
CMD ["python", "run.py"]
