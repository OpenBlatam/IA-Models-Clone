# Production Dockerfile for Diffusion System
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements
COPY requirements_production.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements_production.txt

# Copy application code
COPY production_diffusion_system.py .
COPY production_config.yaml .

# Create necessary directories
RUN mkdir -p /var/log/diffusion_system \
    /var/cache/diffusion_system/models \
    /var/checkpoints/diffusion_system \
    /tmp/diffusion_system

# Create non-root user
RUN useradd -m -u 1000 diffusion && \
    chown -R diffusion:diffusion /app /var/log/diffusion_system \
    /var/cache/diffusion_system /var/checkpoints/diffusion_system \
    /tmp/diffusion_system

# Switch to non-root user
USER diffusion

# Expose ports
EXPOSE 8080 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8080/health')"

# Run the application
CMD ["python3", "production_diffusion_system.py"]


